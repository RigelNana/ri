//! Invocation orchestration independent of the concrete SDK adapter.

use std::sync::Arc;

use crate::cli::{Cli, Command, ModelCommand, ModelListArgs, ProviderCommand};
use crate::error::{CliError, Result};
use crate::input::{PreparedInput, prepare, read_piped_stdin};
use crate::mode::{IoCapabilities, RunMode, StdinUse, select_mode, stdin_use};
use crate::modes::{run_command, run_headless};
use crate::output::Output;
use crate::runtime::CliRuntime;

/// Fully consumed stdin/file input paired with its selected frontend.
#[derive(Debug)]
pub struct PreparedRun {
    mode: RunMode,
    input: PreparedInput,
}

/// Validate mode-dependent constraints without constructing application state.
///
/// # Errors
///
/// Returns an argument or feature error when the invocation is inconsistent
/// with the selected frontend or terminal capabilities.
pub fn validate_invocation(cli: &Cli, io: IoCapabilities) -> Result<Option<RunMode>> {
    cli.validate()?;
    if cli.is_metadata_request() {
        return Ok(None);
    }
    let mode = select_mode(cli.mode, cli.print, io)?;
    let inputs = cli.inputs();
    if mode == RunMode::Rpc {
        if !inputs.files.is_empty() {
            return Err(CliError::InvalidArguments(
                "RPC mode does not accept @file arguments".to_owned(),
            ));
        }
        if !inputs.messages.is_empty() {
            return Err(CliError::InvalidArguments(
                "RPC mode reads commands from stdin and does not accept prompt arguments"
                    .to_owned(),
            ));
        }
    }
    if matches!(mode, RunMode::Text | RunMode::Json)
        && io.stdin_tty
        && inputs.files.is_empty()
        && inputs.messages.is_empty()
    {
        return Err(CliError::MissingPrompt);
    }
    #[cfg(not(feature = "interactive"))]
    if mode == RunMode::Interactive {
        return Err(CliError::FeatureUnavailable {
            mode: "interactive",
            feature: "interactive",
        });
    }
    #[cfg(not(feature = "rpc"))]
    if mode == RunMode::Rpc {
        return Err(CliError::FeatureUnavailable {
            mode: "RPC",
            feature: "rpc",
        });
    }
    Ok(Some(mode))
}

/// Consume any piped stdin and attached files before runtime construction.
///
/// # Errors
///
/// Returns an argument, feature, or I/O error when mode validation, stdin
/// consumption, or attachment preparation fails.
pub async fn prepare_run(cli: &Cli, io: IoCapabilities) -> Result<Option<PreparedRun>> {
    let Some(mode) = validate_invocation(cli, io)? else {
        return Ok(None);
    };
    let inputs = cli.inputs();
    let piped = match stdin_use(mode, io) {
        StdinUse::Prompt => read_piped_stdin().await?,
        StdinUse::Interactive | StdinUse::Rpc | StdinUse::None => None,
    };
    let input = if mode == RunMode::Rpc {
        PreparedInput::default()
    } else {
        prepare(inputs, piped).await?
    };
    if matches!(mode, RunMode::Text | RunMode::Json) && input.is_empty() {
        return Err(CliError::MissingPrompt);
    }
    Ok(Some(PreparedRun { mode, input }))
}

/// Execute a parsed invocation against one already-created shared runtime.
///
/// # Errors
///
/// Returns any validation, input, runtime, output, or shutdown error produced
/// while executing the invocation.
pub async fn execute(
    cli: Cli,
    io: IoCapabilities,
    runtime: Arc<dyn CliRuntime>,
    output: Output,
) -> Result<()> {
    let prepared = match prepare_run(&cli, io).await {
        Ok(prepared) => prepared,
        Err(error) => {
            let _ = runtime.shutdown().await;
            return Err(error);
        }
    };
    execute_prepared(cli, runtime, output, prepared).await
}

/// Execute an invocation whose stdin and file attachments were prepared.
///
/// # Errors
///
/// Returns any command, frontend, output, or runtime shutdown error.
pub async fn execute_prepared(
    cli: Cli,
    runtime: Arc<dyn CliRuntime>,
    output: Output,
    prepared: Option<PreparedRun>,
) -> Result<()> {
    let operation = execute_inner(&cli, Arc::clone(&runtime), &output, prepared).await;
    let shutdown = runtime.shutdown().await;
    match (operation, shutdown) {
        (Err(error), _) => Err(error),
        (Ok(()), result) => result,
    }
}

async fn execute_inner(
    cli: &Cli,
    runtime: Arc<dyn CliRuntime>,
    output: &Output,
    prepared: Option<PreparedRun>,
) -> Result<()> {
    cli.validate()?;

    if let Some(command) = &cli.command {
        return run_command(runtime.as_ref(), command, output).await;
    }
    if cli.list_providers {
        return run_command(
            runtime.as_ref(),
            &Command::Provider {
                command: ProviderCommand::List,
            },
            output,
        )
        .await;
    }
    if let Some(search) = &cli.list_models {
        return run_command(
            runtime.as_ref(),
            &Command::Model {
                command: ModelCommand::List(ModelListArgs {
                    provider: cli.provider.clone(),
                    search: (!search.is_empty()).then(|| search.clone()),
                    all: false,
                    json: false,
                }),
            },
            output,
        )
        .await;
    }

    let PreparedRun { mode, input } = prepared.ok_or_else(|| {
        CliError::InvalidArguments("run input was not prepared before execution".to_owned())
    })?;

    match mode {
        RunMode::Text | RunMode::Json => run_headless(runtime, mode, input, output).await,
        RunMode::Interactive => {
            #[cfg(feature = "interactive")]
            {
                crate::modes::run_interactive(runtime, input).await
            }
            #[cfg(not(feature = "interactive"))]
            {
                let _ = (runtime, input);
                Err(CliError::FeatureUnavailable {
                    mode: "interactive",
                    feature: "interactive",
                })
            }
        }
        RunMode::Rpc => {
            #[cfg(feature = "rpc")]
            {
                crate::rpc_mode::run_rpc(runtime).await
            }
            #[cfg(not(feature = "rpc"))]
            {
                let _ = runtime;
                Err(CliError::FeatureUnavailable {
                    mode: "RPC",
                    feature: "rpc",
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use clap::Parser as _;
    use serde_json::Value;
    use tokio::sync::broadcast;

    use super::*;
    use crate::runtime::{CommandOutput, PromptCompletion, PromptRequest, RuntimeStatus};

    #[derive(Debug)]
    struct EmptyRuntime {
        events: broadcast::Sender<Value>,
        shutdowns: AtomicUsize,
    }

    impl EmptyRuntime {
        fn new() -> Self {
            let (events, _) = broadcast::channel(8);
            Self {
                events,
                shutdowns: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl CliRuntime for EmptyRuntime {
        fn subscribe(&self) -> broadcast::Receiver<Value> {
            self.events.subscribe()
        }

        async fn session_header(&self) -> Result<Option<Value>> {
            Ok(None)
        }

        async fn prompt(&self, _request: PromptRequest) -> Result<PromptCompletion> {
            unreachable!("empty input must fail before prompting")
        }

        async fn abort(&self) -> Result<()> {
            Ok(())
        }

        async fn status(&self) -> Result<RuntimeStatus> {
            Ok(RuntimeStatus::default())
        }

        async fn command(&self, _command: &Command) -> Result<CommandOutput> {
            Ok(CommandOutput::Silent)
        }

        #[cfg(feature = "rpc")]
        async fn rpc(
            &self,
            _request: ri_rpc::Request,
            _context: ri_rpc::DispatchContext,
        ) -> Result<ri_rpc::ResponsePayload> {
            Err(CliError::unsupported("RPC", "not used by this test"))
        }

        async fn shutdown(&self) -> Result<()> {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn shutdown_runs_after_validation_failure() {
        let cli = Cli::try_parse_from(["ri", "-p"]).unwrap();
        let runtime = Arc::new(EmptyRuntime::new());
        let result = execute(
            cli,
            IoCapabilities::new(true, true),
            runtime.clone(),
            Output::stdio(),
        )
        .await;
        assert!(matches!(result, Err(CliError::MissingPrompt)));
        assert_eq!(runtime.shutdowns.load(Ordering::SeqCst), 1);
    }
}
