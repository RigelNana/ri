//! Frontend mode bindings over one [`CliRuntime`](crate::runtime::CliRuntime).

use std::sync::Arc;

use ri_sdk::FrontendMode;
use serde_json::Value;
use tokio::sync::broadcast;

use crate::cli::Command;
use crate::error::{CliError, Result};
use crate::input::PreparedInput;
use crate::mode::RunMode;
use crate::output::Output;
use crate::runtime::{CliRuntime, CommandOutput, PromptCompletion, PromptRequest};

/// Execute one administrative command.
///
/// # Errors
///
/// Returns the runtime command error or an output serialization/I/O error.
pub async fn run_command(
    runtime: &dyn CliRuntime,
    command: &Command,
    output: &Output,
) -> Result<()> {
    match runtime.command(command).await? {
        CommandOutput::Silent => Ok(()),
        CommandOutput::Text(text) => output.stdout_line(&text).await,
        CommandOutput::Json(value) => output.json(&value).await,
    }
}

/// Run text or JSON-event single-shot mode.
///
/// # Errors
///
/// Returns an argument, prompt, event-stream, interruption, or output error.
pub async fn run_headless(
    runtime: Arc<dyn CliRuntime>,
    mode: RunMode,
    input: PreparedInput,
    output: &Output,
) -> Result<()> {
    if !matches!(mode, RunMode::Text | RunMode::Json) {
        return Err(CliError::InvalidArguments(
            "headless runner requires text or JSON mode".to_owned(),
        ));
    }
    if input.is_empty() {
        return Err(CliError::MissingPrompt);
    }

    let mut events = runtime.subscribe();
    if mode == RunMode::Json
        && let Some(header) = runtime.session_header().await?
    {
        output.json(&header).await?;
    }

    let mut completion = PromptCompletion::default();
    if let Some(initial) = input.initial {
        completion = run_one(
            Arc::clone(&runtime),
            PromptRequest {
                text: initial,
                images: input.images,
                source: if mode == RunMode::Json {
                    FrontendMode::Json
                } else {
                    FrontendMode::Print
                },
                delivery: None,
            },
            mode,
            &mut events,
            output,
        )
        .await?;
    }
    for follow_up in input.follow_ups {
        completion = run_one(
            Arc::clone(&runtime),
            PromptRequest {
                text: follow_up,
                images: Vec::new(),
                source: if mode == RunMode::Json {
                    FrontendMode::Json
                } else {
                    FrontendMode::Print
                },
                delivery: None,
            },
            mode,
            &mut events,
            output,
        )
        .await?;
    }

    if mode == RunMode::Text && !completion.text.is_empty() {
        output.stdout_line(&completion.text).await?;
    }
    Ok(())
}

async fn run_one(
    runtime: Arc<dyn CliRuntime>,
    request: PromptRequest,
    mode: RunMode,
    events: &mut broadcast::Receiver<Value>,
    output: &Output,
) -> Result<PromptCompletion> {
    if mode == RunMode::Text {
        return tokio::select! {
            result = runtime.prompt(request) => result,
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(|source| CliError::Io {
                    operation: "listen for Ctrl+C",
                    source,
                })?;
                runtime.abort().await?;
                Err(CliError::Interrupted)
            }
        };
    }

    let operation = runtime.prompt(request);
    tokio::pin!(operation);
    let completion = loop {
        tokio::select! {
            result = &mut operation => break result?,
            event = events.recv() => {
                match event {
                    Ok(event) => output.json(&event).await?,
                    Err(broadcast::error::RecvError::Lagged(count)) => {
                        return Err(CliError::EventLagged(count));
                    }
                    Err(broadcast::error::RecvError::Closed) => break operation.await?,
                }
            }
            signal = tokio::signal::ctrl_c() => {
                signal.map_err(|source| CliError::Io {
                    operation: "listen for Ctrl+C",
                    source,
                })?;
                runtime.abort().await?;
                return Err(CliError::Interrupted);
            }
        }
    };

    loop {
        match events.try_recv() {
            Ok(event) => output.json(&event).await?,
            Err(broadcast::error::TryRecvError::Lagged(count)) => {
                return Err(CliError::EventLagged(count));
            }
            Err(broadcast::error::TryRecvError::Empty | broadcast::error::TryRecvError::Closed) => {
                break;
            }
        }
    }
    if let Some(message) = &completion.message {
        output
            .json(&serde_json::json!({
                "type": "message_end",
                "message": message,
            }))
            .await?;
    }
    Ok(completion)
}

/// Run the terminal UI until the user exits.
///
/// # Errors
///
/// Returns a terminal, runtime, prompt, or interruption error.
#[cfg(feature = "interactive")]
pub async fn run_interactive(runtime: Arc<dyn CliRuntime>, input: PreparedInput) -> Result<()> {
    use std::collections::VecDeque;
    use std::time::Duration;

    use ri_tui::{CrosstermTerminal, Tui};
    use tokio::sync::mpsc;

    use crate::interactive::{InteractiveAction, channel};
    use crate::runtime::PromptDelivery;

    let (component, handle, mut actions) = channel();
    let mut tui = Tui::new(CrosstermTerminal::stdout());
    tui.mount(component);
    let status = runtime.status().await?;
    handle.set_status(format_status(&status));

    let mut pending = VecDeque::new();
    if let Some(initial) = input.initial {
        pending.push_back(PromptRequest {
            text: initial,
            images: input.images,
            source: FrontendMode::Interactive,
            delivery: None,
        });
    }
    pending.extend(input.follow_ups.into_iter().map(|text| PromptRequest {
        text,
        images: Vec::new(),
        source: FrontendMode::Interactive,
        delivery: None,
    }));

    let (completed_tx, mut completed_rx) = mpsc::unbounded_channel();
    let mut active = 0_usize;
    tui.start()?;
    let run_result = async {
        if let Some(request) = pending.pop_front() {
            handle.user(&request.text);
            spawn_prompt(Arc::clone(&runtime), request, completed_tx.clone());
            active += 1;
            handle.set_busy(true);
        }
        tui.render()?;

        let mut events = runtime.subscribe();
        let mut quitting = false;
        while !quitting {
            let mut changed = tui.tick(Duration::from_millis(20))?;

            while let Ok(action) = actions.try_recv() {
                changed = true;
                match action {
                    InteractiveAction::Submit(text) => {
                        handle.user(&text);
                        let delivery = (active > 0).then_some(PromptDelivery::FollowUp);
                        spawn_prompt(
                            Arc::clone(&runtime),
                            PromptRequest {
                                text,
                                images: Vec::new(),
                                source: FrontendMode::Interactive,
                                delivery,
                            },
                            completed_tx.clone(),
                        );
                        active += 1;
                        handle.set_busy(true);
                    }
                    InteractiveAction::Abort => {
                        runtime.abort().await?;
                        handle.notice("Abort requested");
                    }
                    InteractiveAction::Quit => {
                        if active > 0 {
                            runtime.abort().await?;
                        }
                        quitting = true;
                    }
                }
            }

            while let Ok(result) = completed_rx.try_recv() {
                changed = true;
                active = active.saturating_sub(1);
                match result {
                    Ok(completion) => handle.assistant(completion.text),
                    Err(error) => handle.error(error.to_string()),
                }
                if active == 0
                    && let Some(request) = pending.pop_front()
                {
                    handle.user(&request.text);
                    spawn_prompt(Arc::clone(&runtime), request, completed_tx.clone());
                    active += 1;
                }
                handle.set_busy(active > 0);
                handle.set_status(format_status(&runtime.status().await?));
            }

            loop {
                match events.try_recv() {
                    Ok(event) => {
                        changed |= apply_interactive_event(&handle, &event);
                    }
                    Err(broadcast::error::TryRecvError::Lagged(count)) => {
                        handle.error(format!("runtime event stream lost {count} event(s)"));
                        changed = true;
                    }
                    Err(
                        broadcast::error::TryRecvError::Empty
                        | broadcast::error::TryRecvError::Closed,
                    ) => break,
                }
            }

            if changed {
                tui.invalidate();
                tui.render()?;
            }
            tokio::task::yield_now().await;
        }
        Ok(())
    }
    .await;
    let stop_result = tui.stop().map_err(CliError::from);
    run_result.and(stop_result)
}

#[cfg(feature = "interactive")]
fn spawn_prompt(
    runtime: Arc<dyn CliRuntime>,
    request: PromptRequest,
    completed: tokio::sync::mpsc::UnboundedSender<Result<PromptCompletion>>,
) {
    tokio::spawn(async move {
        let result = runtime.prompt(request).await;
        let _ = completed.send(result);
    });
}

#[cfg(feature = "interactive")]
fn format_status(status: &crate::runtime::RuntimeStatus) -> String {
    let model = status.model.as_deref().unwrap_or("no model");
    let thinking = if status.thinking.is_empty() {
        "off"
    } else {
        &status.thinking
    };
    format!(
        "{model} · thinking {thinking} · session {}",
        status.session_id
    )
}

#[cfg(feature = "interactive")]
fn apply_interactive_event(handle: &crate::interactive::InteractiveHandle, event: &Value) -> bool {
    match event.get("type").and_then(Value::as_str) {
        Some("auto_retry_start") => {
            handle.notice("Retrying provider request");
            true
        }
        Some("compaction_start") => {
            handle.notice("Compacting session context");
            true
        }
        Some("extension_error") => {
            let error = event
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("extension failed");
            handle.error(error);
            true
        }
        _ => false,
    }
}
