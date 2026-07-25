//! Typed `ri-rpc` dispatch over the shared SDK runtime.

#![cfg(feature = "rpc")]

use std::sync::Arc;
use std::sync::atomic::Ordering;

use ri_harness::{HarnessEvent, Phase, RetryOperation, SummaryKind};
use ri_rpc::{
    AgentMessage, AvailableModelsData, AvailableThinkingLevelsData, BashResult as RpcBashResult,
    CancelledData, Command as RpcCommand, CommandsData, CompactionResult as RpcCompactionResult,
    CycleModelData, EntriesData, Event, ExportHtmlData, ForkData, ForkMessage, ForkMessagesData,
    LastAssistantTextData, MessagesData, Model as RpcModel, ModelCost as RpcModelCost,
    ModelCostTier as RpcModelCostTier, ModelInput as RpcModelInput, Request, ResponsePayload,
    SessionState, SessionStats as RpcSessionStats, SessionTokenTotals,
    SessionTreeNode as RpcTreeNode, SlashCommand, SlashCommandSource, SourceInfo, SourceOrigin,
    SourceScope, SummarizationSource, ThinkingLevelData, TreeData, Usage as RpcUsage,
    UsageCost as RpcUsageCost,
};
use ri_sdk::{FrontendMode, PromptOptions, StreamingBehavior};
use ri_session::{CreateOptions, SessionEntry};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use super::SdkCliRuntime;
use super::events::duration_millis;
use crate::cli::SessionForkArgs;
use crate::error::{CliError, Result};

pub(super) async fn dispatch(
    adapter: &SdkCliRuntime,
    request: Request,
    context: ri_rpc::DispatchContext,
) -> Result<ResponsePayload> {
    ensure_event_bridge(adapter, &context);
    let request_id = request.id.map(ri_rpc::RequestId::into_inner);
    match request.command {
        RpcCommand::Prompt {
            message,
            images,
            streaming_behavior,
        } => {
            adapter
                .required_runtime("handle an RPC prompt")?
                .frontend(FrontendMode::Rpc)
                .prompt(
                    message,
                    PromptOptions {
                        images: images.into_iter().map(command_image).collect(),
                        streaming_behavior: streaming_behavior.map(|behavior| match behavior {
                            ri_rpc::StreamingBehavior::Steer => StreamingBehavior::Steer,
                            ri_rpc::StreamingBehavior::FollowUp => StreamingBehavior::FollowUp,
                        }),
                        expand_resources: true,
                        ..PromptOptions::default()
                    },
                )
                .await
                .map_err(|error| CliError::runtime("run RPC prompt", error))?;
            Ok(ResponsePayload::Prompt)
        }
        RpcCommand::Steer { message, images } => {
            adapter
                .required_runtime("queue RPC steering input")?
                .steer_with_images(message, images.into_iter().map(command_image).collect())
                .await
                .map_err(|error| CliError::runtime("queue steering input", error))?;
            Ok(ResponsePayload::Steer)
        }
        RpcCommand::FollowUp { message, images } => {
            adapter
                .required_runtime("queue RPC follow-up input")?
                .follow_up_with_images(message, images.into_iter().map(command_image).collect())
                .await
                .map_err(|error| CliError::runtime("queue follow-up input", error))?;
            Ok(ResponsePayload::FollowUp)
        }
        RpcCommand::Abort => {
            adapter
                .required_runtime("abort an RPC run")?
                .abort()
                .await
                .map_err(|error| CliError::runtime("abort RPC run", error))?;
            Ok(ResponsePayload::Abort)
        }
        RpcCommand::NewSession { parent_session } => {
            let runtime = adapter.required_runtime("create an RPC session")?;
            let mut options = CreateOptions::new(adapter.cwd.to_string_lossy().into_owned());
            options.parent_session = parent_session;
            let session = adapter.sessions.create(options).await?;
            runtime
                .replace_session(session)
                .await
                .map_err(|error| CliError::runtime("replace RPC session", error))?;
            Ok(ResponsePayload::NewSession {
                data: CancelledData { cancelled: false },
            })
        }
        RpcCommand::GetState => Ok(ResponsePayload::GetState {
            data: state(adapter).await?,
        }),
        RpcCommand::SetModel { provider, model_id } => {
            let model =
                adapter
                    .models
                    .model(&provider, &model_id)
                    .ok_or_else(|| CliError::NotFound {
                        kind: "model",
                        name: format!("{provider}/{model_id}"),
                    })?;
            adapter
                .required_runtime("set the RPC model")?
                .harness()
                .set_model(Arc::new(model.clone()))
                .await
                .map_err(|error| CliError::runtime("set model", error))?;
            Ok(ResponsePayload::SetModel {
                data: rpc_model(&model)?,
            })
        }
        RpcCommand::CycleModel => {
            let runtime = adapter.required_runtime("cycle the RPC model")?;
            let config = runtime.harness().config().await;
            let available = adapter
                .models
                .available(None)
                .await
                .map_err(|error| CliError::runtime("list models for cycling", error))?;
            let selected = next(
                &available,
                available
                    .iter()
                    .position(|model| {
                        model.provider == config.model.provider && model.id == config.model.id
                    })
                    .unwrap_or_default(),
            )
            .cloned();
            let data = if let Some(selected) = selected {
                runtime
                    .harness()
                    .set_model(Arc::new(selected.clone()))
                    .await
                    .map_err(|error| CliError::runtime("cycle model", error))?;
                let config = runtime.harness().config().await;
                Some(CycleModelData {
                    model: rpc_model(&selected)?,
                    thinking_level: config.thinking_level,
                    is_scoped: false,
                })
            } else {
                None
            };
            Ok(ResponsePayload::CycleModel { data })
        }
        RpcCommand::GetAvailableModels => {
            let models = adapter
                .models
                .available(None)
                .await
                .map_err(|error| CliError::runtime("list available RPC models", error))?
                .into_iter()
                .map(|model| rpc_model(&model))
                .collect::<Result<Vec<_>>>()?;
            Ok(ResponsePayload::GetAvailableModels {
                data: AvailableModelsData { models },
            })
        }
        RpcCommand::SetThinkingLevel { level } => {
            adapter
                .required_runtime("set RPC thinking level")?
                .harness()
                .set_thinking_level(level)
                .await
                .map_err(|error| CliError::runtime("set thinking level", error))?;
            Ok(ResponsePayload::SetThinkingLevel)
        }
        RpcCommand::CycleThinkingLevel => {
            let runtime = adapter.required_runtime("cycle RPC thinking level")?;
            let config = runtime.harness().config().await;
            let levels = ri_ai::supported_thinking_levels(&config.model);
            let selected = next(
                &levels,
                levels
                    .iter()
                    .position(|level| *level == config.thinking_level)
                    .unwrap_or_default(),
            )
            .copied();
            if let Some(selected) = selected {
                runtime
                    .harness()
                    .set_thinking_level(selected)
                    .await
                    .map_err(|error| CliError::runtime("cycle thinking level", error))?;
            }
            Ok(ResponsePayload::CycleThinkingLevel {
                data: selected.map(|level| ThinkingLevelData { level }),
            })
        }
        RpcCommand::GetAvailableThinkingLevels => {
            let config = adapter
                .required_runtime("list thinking levels")?
                .harness()
                .config()
                .await;
            Ok(ResponsePayload::GetAvailableThinkingLevels {
                data: AvailableThinkingLevelsData {
                    levels: ri_ai::supported_thinking_levels(&config.model),
                },
            })
        }
        RpcCommand::SetSteeringMode { mode } => {
            let runtime = adapter.required_runtime("set steering queue mode")?;
            let config = runtime.harness().config().await;
            runtime
                .harness()
                .set_queue_modes(mode, config.follow_up_mode)
                .await;
            Ok(ResponsePayload::SetSteeringMode)
        }
        RpcCommand::SetFollowUpMode { mode } => {
            let runtime = adapter.required_runtime("set follow-up queue mode")?;
            let config = runtime.harness().config().await;
            runtime
                .harness()
                .set_queue_modes(config.steering_mode, mode)
                .await;
            Ok(ResponsePayload::SetFollowUpMode)
        }
        RpcCommand::Compact {
            custom_instructions,
        } => {
            let result = adapter
                .required_runtime("compact the RPC session")?
                .compact(custom_instructions)
                .await
                .map_err(|error| CliError::runtime("compact session", error))?;
            Ok(ResponsePayload::Compact {
                data: rpc_compaction(result),
            })
        }
        RpcCommand::SetAutoCompaction { enabled } => {
            adapter
                .packages
                .set_auto_compaction_enabled(enabled)
                .await?;
            adapter
                .required_runtime("set automatic compaction")?
                .set_auto_compaction_enabled(enabled)
                .await;
            Ok(ResponsePayload::SetAutoCompaction)
        }
        RpcCommand::SetAutoRetry { enabled } => {
            adapter.packages.set_auto_retry_enabled(enabled).await?;
            adapter
                .required_runtime("set automatic retry")?
                .set_auto_retry_enabled(enabled)
                .await;
            Ok(ResponsePayload::SetAutoRetry)
        }
        RpcCommand::AbortRetry => {
            adapter
                .required_runtime("abort automatic retry")?
                .abort_retry()
                .await;
            Ok(ResponsePayload::AbortRetry)
        }
        RpcCommand::Bash {
            command,
            exclude_from_context: _,
        } => {
            let process_id = request_id.unwrap_or_else(|| {
                format!(
                    "bash-{}",
                    adapter.bash_sequence.fetch_add(1, Ordering::Relaxed)
                )
            });
            let cancellation = CancellationToken::new();
            adapter
                .bash_processes
                .lock()
                .await
                .insert(process_id.clone(), cancellation.clone());
            let result = ri_tools::Tools::local(adapter.cwd.clone())
                .bash_with_cancellation(
                    ri_tools::BashInput {
                        command,
                        timeout: None,
                    },
                    &cancellation,
                    None,
                )
                .await;
            adapter.bash_processes.lock().await.remove(&process_id);
            let result =
                result.map_err(|error| CliError::runtime("run RPC shell command", error))?;
            let output = result.text_content();
            let details = result.details;
            Ok(ResponsePayload::Bash {
                data: RpcBashResult {
                    output,
                    exit_code: details.as_ref().map(|details| details.exit_code),
                    cancelled: cancellation.is_cancelled(),
                    truncated: details
                        .as_ref()
                        .and_then(|details| details.truncation.as_ref())
                        .is_some(),
                    full_output_path: details
                        .and_then(|details| details.full_output_path)
                        .map(|path| path.to_string_lossy().into_owned()),
                },
            })
        }
        RpcCommand::AbortBash => {
            let processes = adapter
                .bash_processes
                .lock()
                .await
                .values()
                .cloned()
                .collect::<Vec<_>>();
            for process in processes {
                process.cancel();
            }
            Ok(ResponsePayload::AbortBash)
        }
        RpcCommand::GetSessionStats => Ok(ResponsePayload::GetSessionStats {
            data: session_stats(adapter).await?,
        }),
        RpcCommand::ExportHtml { output_path } => {
            let runtime = adapter.required_runtime("export the RPC session")?;
            let session = runtime.session().await;
            let metadata = session
                .metadata()
                .await
                .map_err(|error| CliError::runtime("read session metadata", error))?;
            let messages = session
                .context()
                .await
                .map_err(|error| CliError::runtime("project session messages", error))?
                .messages;
            let path = output_path.map_or_else(
                || adapter.cwd.join(format!("{}.html", metadata.id)),
                |path| {
                    let path = std::path::PathBuf::from(path);
                    if path.is_absolute() {
                        path
                    } else {
                        adapter.cwd.join(path)
                    }
                },
            );
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|source| CliError::Io {
                        operation: "create HTML export directory",
                        source,
                    })?;
            }
            let body =
                serde_json::to_string_pretty(&messages).map_err(|source| CliError::Json {
                    operation: "encode HTML session export",
                    source,
                })?;
            let html = format!(
                "<!doctype html><meta charset=\"utf-8\"><title>Ri session {}</title>\
                 <style>body{{font:14px/1.5 ui-monospace,monospace;max-width:1000px;margin:2rem auto;padding:0 1rem}}pre{{white-space:pre-wrap}}</style>\
                 <h1>Ri session {}</h1><pre>{}</pre>\n",
                html_escape(&metadata.id),
                html_escape(&metadata.id),
                html_escape(&body),
            );
            tokio::fs::write(&path, html)
                .await
                .map_err(|source| CliError::Io {
                    operation: "write HTML session export",
                    source,
                })?;
            Ok(ResponsePayload::ExportHtml {
                data: ExportHtmlData {
                    path: path.to_string_lossy().into_owned(),
                },
            })
        }
        RpcCommand::SwitchSession { session_path } => {
            let session = adapter.sessions.open_target(&session_path).await?;
            adapter
                .required_runtime("switch RPC session")?
                .replace_session(session)
                .await
                .map_err(|error| CliError::runtime("switch session", error))?;
            Ok(ResponsePayload::SwitchSession {
                data: CancelledData { cancelled: false },
            })
        }
        RpcCommand::Fork { entry_id } => {
            let runtime = adapter.required_runtime("fork RPC session")?;
            let source_session = runtime.session().await;
            let source = source_session
                .metadata()
                .await
                .map_err(|error| CliError::runtime("read source session metadata", error))?
                .id;
            let text = source_session
                .entry(&entry_id)
                .await
                .map_err(|error| CliError::runtime("read fork message", error))?
                .as_ref()
                .and_then(|entry| match &entry.entry {
                    SessionEntry::Message(message) => Some(message_text(&message.message)),
                    _ => None,
                })
                .unwrap_or_default();
            let fork = adapter
                .sessions
                .fork(&SessionForkArgs {
                    source,
                    entry: Some(entry_id),
                    at: false,
                    id: None,
                    cwd: Some(adapter.cwd.clone()),
                })
                .await?;
            runtime
                .replace_session(fork)
                .await
                .map_err(|error| CliError::runtime("activate forked session", error))?;
            Ok(ResponsePayload::Fork {
                data: ForkData {
                    text,
                    cancelled: false,
                },
            })
        }
        RpcCommand::Clone => {
            let runtime = adapter.required_runtime("clone RPC session")?;
            let source = runtime
                .session()
                .await
                .metadata()
                .await
                .map_err(|error| CliError::runtime("read source session metadata", error))?
                .id;
            let clone = adapter
                .sessions
                .fork(&SessionForkArgs {
                    source,
                    entry: None,
                    at: false,
                    id: None,
                    cwd: Some(adapter.cwd.clone()),
                })
                .await?;
            runtime
                .replace_session(clone)
                .await
                .map_err(|error| CliError::runtime("activate cloned session", error))?;
            Ok(ResponsePayload::Clone {
                data: CancelledData { cancelled: false },
            })
        }
        RpcCommand::GetForkMessages => {
            let entries = adapter
                .required_runtime("list RPC fork messages")?
                .session()
                .await
                .entries(None, None)
                .await
                .map_err(|error| CliError::runtime("list session entries", error))?;
            let messages = entries
                .into_iter()
                .filter_map(|entry| match entry.entry {
                    SessionEntry::Message(message)
                        if message.message.get("role").and_then(Value::as_str) == Some("user") =>
                    {
                        Some(ForkMessage {
                            entry_id: message.base.id,
                            text: message_text(&message.message),
                        })
                    }
                    _ => None,
                })
                .collect();
            Ok(ResponsePayload::GetForkMessages {
                data: ForkMessagesData { messages },
            })
        }
        RpcCommand::GetEntries { since } => {
            let session = adapter
                .required_runtime("read RPC entries")?
                .session()
                .await;
            let mut entries = session
                .entries(None, None)
                .await
                .map_err(|error| CliError::runtime("read session entries", error))?;
            if let Some(since) = since {
                let position = entries
                    .iter()
                    .position(|entry| entry.entry.id() == since)
                    .ok_or_else(|| CliError::NotFound {
                        kind: "session entry",
                        name: since.clone(),
                    })?;
                entries.drain(..=position);
            }
            let entries = entries
                .into_iter()
                .map(|entry| {
                    ri_compat::native_entry_to_pi(entry.entry).map_err(|error| {
                        CliError::runtime("convert a native session entry for RPC", error)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let leaf_id = session
                .leaf_id()
                .await
                .map_err(|error| CliError::runtime("read session leaf", error))?;
            Ok(ResponsePayload::GetEntries {
                data: EntriesData { entries, leaf_id },
            })
        }
        RpcCommand::GetTree => {
            let session = adapter.required_runtime("read RPC tree")?.session().await;
            let tree = session
                .tree()
                .await
                .map_err(|error| CliError::runtime("read session tree", error))?
                .into_iter()
                .map(rpc_tree_node)
                .collect::<Result<Vec<_>>>()?;
            let leaf_id = session
                .leaf_id()
                .await
                .map_err(|error| CliError::runtime("read session leaf", error))?;
            Ok(ResponsePayload::GetTree {
                data: TreeData { tree, leaf_id },
            })
        }
        RpcCommand::GetLastAssistantText => {
            let text = adapter
                .required_runtime("read last assistant text")?
                .session()
                .await
                .context()
                .await
                .map_err(|error| CliError::runtime("project session messages", error))?
                .messages
                .iter()
                .rev()
                .find_map(assistant_text);
            Ok(ResponsePayload::GetLastAssistantText {
                data: LastAssistantTextData { text },
            })
        }
        RpcCommand::SetSessionName { name } => {
            if name.trim().is_empty() {
                return Err(CliError::InvalidArguments(
                    "session name cannot be empty".to_owned(),
                ));
            }
            adapter
                .required_runtime("set RPC session name")?
                .session()
                .await
                .append_session_info(Some(name))
                .await
                .map_err(|error| CliError::runtime("set session name", error))?;
            Ok(ResponsePayload::SetSessionName)
        }
        RpcCommand::GetMessages => {
            let messages = adapter
                .required_runtime("read RPC messages")?
                .session()
                .await
                .context()
                .await
                .map_err(|error| CliError::runtime("project session messages", error))?
                .messages
                .into_iter()
                .map(|message| {
                    serde_json::from_value(message).map_err(|source| CliError::Json {
                        operation: "decoding an RPC agent message",
                        source,
                    })
                })
                .collect::<Result<Vec<AgentMessage>>>()?;
            Ok(ResponsePayload::GetMessages {
                data: MessagesData { messages },
            })
        }
        RpcCommand::GetCommands => Ok(ResponsePayload::GetCommands {
            data: CommandsData {
                commands: resource_commands(adapter),
            },
        }),
    }
}

fn ensure_event_bridge(adapter: &SdkCliRuntime, context: &ri_rpc::DispatchContext) {
    if adapter.rpc_events_started.swap(true, Ordering::AcqRel) {
        return;
    }
    let mut events = adapter.harness_events.subscribe();
    let context = context.clone();
    tokio::spawn(async move {
        loop {
            match events.recv().await {
                Ok(harness_event) => {
                    let Some(event) = rpc_event(&harness_event) else {
                        continue;
                    };
                    if context.emit(event).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });
}

fn rpc_event(event: &HarnessEvent) -> Option<Event> {
    match event {
        HarnessEvent::PromptAccepted { .. } => Some(Event::AgentStart),
        HarnessEvent::Settled { .. } => Some(Event::AgentSettled),
        HarnessEvent::RetryScheduled {
            operation,
            attempt,
            max_attempts,
            delay,
            error,
        } => Some(match operation {
            RetryOperation::Agent => Event::AutoRetryStart {
                attempt: *attempt,
                max_attempts: *max_attempts,
                delay_ms: duration_millis(*delay),
                error_message: error.clone(),
            },
            RetryOperation::Compaction
            | RetryOperation::TurnPrefix
            | RetryOperation::BranchSummary => Event::SummarizationRetryScheduled {
                attempt: *attempt,
                max_attempts: *max_attempts,
                delay_ms: duration_millis(*delay),
                error_message: error.clone(),
            },
        }),
        HarnessEvent::RetryAttemptStarted { kind, reason } => {
            Some(Event::SummarizationRetryAttemptStart {
                source: match kind {
                    SummaryKind::Branch => SummarizationSource::BranchSummary,
                    SummaryKind::Compaction | SummaryKind::TurnPrefix => {
                        SummarizationSource::Compaction
                    }
                },
                reason: *reason,
            })
        }
        HarnessEvent::RetryFinished {
            operation,
            success,
            attempt,
            final_error,
        } => Some(match operation {
            RetryOperation::Agent => Event::AutoRetryEnd {
                success: *success,
                attempt: *attempt,
                final_error: final_error.clone(),
            },
            RetryOperation::Compaction
            | RetryOperation::TurnPrefix
            | RetryOperation::BranchSummary => Event::SummarizationRetryFinished,
        }),
        HarnessEvent::CompactionStarted { reason, .. } => {
            Some(Event::CompactionStart { reason: *reason })
        }
        HarnessEvent::CompactionFinished {
            reason,
            result,
            aborted,
            will_retry,
            error_message,
        } => Some(Event::CompactionEnd {
            reason: *reason,
            result: result.clone().map(|result| rpc_compaction(*result)),
            aborted: *aborted,
            will_retry: *will_retry,
            error_message: error_message.clone(),
        }),
        HarnessEvent::ResourceExpanded { .. }
        | HarnessEvent::QueueUpdated(_)
        | HarnessEvent::MessagePersisted { .. }
        | HarnessEvent::SavePoint { .. }
        | HarnessEvent::BranchNavigated { .. }
        | HarnessEvent::SessionReplacing { .. }
        | HarnessEvent::SessionReplaced { .. } => None,
    }
}

async fn state(adapter: &SdkCliRuntime) -> Result<SessionState> {
    let runtime = adapter.required_runtime("read RPC state")?;
    let status = runtime.status().await;
    let config = runtime.harness().config().await;
    let session = runtime.session().await;
    let metadata = session
        .metadata()
        .await
        .map_err(|error| CliError::runtime("read RPC session metadata", error))?;
    let session_name = session
        .name()
        .await
        .map_err(|error| CliError::runtime("read RPC session name", error))?;
    let message_count = session
        .context()
        .await
        .map_err(|error| CliError::runtime("count RPC session messages", error))?
        .messages
        .len();
    Ok(SessionState {
        model: Some(rpc_model(&config.model)?),
        thinking_level: config.thinking_level,
        is_streaming: !matches!(status.phase, Phase::Idle | Phase::Settling),
        is_compacting: status.phase == Phase::Compaction,
        steering_mode: config.steering_mode,
        follow_up_mode: config.follow_up_mode,
        session_file: metadata
            .path
            .map(|path| path.to_string_lossy().into_owned()),
        session_id: metadata.id,
        session_name,
        auto_compaction_enabled: config.compaction.enabled,
        message_count,
        pending_message_count: status
            .queues
            .steer
            .saturating_add(status.queues.follow_up)
            .saturating_add(status.queues.next_turn),
    })
}

async fn session_stats(adapter: &SdkCliRuntime) -> Result<RpcSessionStats> {
    let runtime = adapter.required_runtime("read session statistics")?;
    let session = runtime.session().await;
    let metadata = session
        .metadata()
        .await
        .map_err(|error| CliError::runtime("read session metadata", error))?;
    let aggregate = session
        .stats()
        .await
        .map_err(|error| CliError::runtime("read session statistics", error))?;
    let entries = session
        .entries(None, None)
        .await
        .map_err(|error| CliError::runtime("read session messages", error))?;
    let mut user_messages = 0;
    let mut assistant_messages = 0;
    let mut tool_results = 0;
    let mut tool_calls = 0;
    for entry in entries {
        let SessionEntry::Message(message) = entry.entry else {
            continue;
        };
        match message.message.get("role").and_then(Value::as_str) {
            Some("user") => user_messages += 1,
            Some("assistant") => {
                assistant_messages += 1;
                tool_calls += message
                    .message
                    .get("content")
                    .and_then(Value::as_array)
                    .map_or(0, |content| {
                        content
                            .iter()
                            .filter(|block| {
                                block.get("type").and_then(Value::as_str) == Some("toolCall")
                            })
                            .count()
                    });
            }
            Some("toolResult") => tool_results += 1,
            _ => {}
        }
    }
    let output = aggregate
        .total_tokens
        .saturating_sub(aggregate.uncached_tokens)
        .saturating_sub(aggregate.cached_tokens);
    Ok(RpcSessionStats {
        session_file: metadata
            .path
            .map(|path| path.to_string_lossy().into_owned()),
        session_id: metadata.id,
        user_messages,
        assistant_messages,
        tool_calls,
        tool_results,
        total_messages: user_messages + assistant_messages + tool_results,
        tokens: SessionTokenTotals {
            input: aggregate.uncached_tokens,
            output,
            cache_read: aggregate.cached_tokens,
            cache_write: 0,
            total: aggregate.total_tokens,
        },
        cost: aggregate.cost_total,
        context_usage: None,
    })
}

fn rpc_model(model: &ri_ai::Model) -> Result<RpcModel> {
    let compat = model
        .compat
        .as_ref()
        .map(serde_json::to_value)
        .transpose()
        .map_err(|source| CliError::Json {
            operation: "encoding RPC model compatibility settings",
            source,
        })?;
    Ok(RpcModel {
        id: model.id.clone(),
        name: model.name.clone(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        base_url: model.base_url.clone(),
        reasoning: model.reasoning,
        input: model
            .input
            .iter()
            .map(|input| match input {
                ri_ai::ModelInput::Text => RpcModelInput::Text,
                ri_ai::ModelInput::Image => RpcModelInput::Image,
            })
            .collect(),
        context_window: model.context_window,
        max_tokens: model.max_tokens,
        cost: RpcModelCost {
            input: model.cost.rates.input,
            output: model.cost.rates.output,
            cache_read: model.cost.rates.cache_read,
            cache_write: model.cost.rates.cache_write,
            tiers: model
                .cost
                .tiers
                .iter()
                .map(|tier| RpcModelCostTier {
                    input_tokens_above: tier.input_tokens_above,
                    input: tier.rates.input,
                    output: tier.rates.output,
                    cache_read: tier.rates.cache_read,
                    cache_write: tier.rates.cache_write,
                })
                .collect(),
        },
        thinking_level_map: model.thinking_level_map.clone(),
        headers: model.headers.clone(),
        compat,
    })
}

fn rpc_compaction(result: ri_harness::CompactionResult) -> RpcCompactionResult {
    RpcCompactionResult {
        summary: result.summary,
        first_kept_entry_id: result.first_kept_entry_id,
        tokens_before: result.tokens_before,
        estimated_tokens_after: Some(result.estimated_tokens_after),
        usage: result.usage.as_ref().map(rpc_usage),
        details: result.details,
    }
}

fn rpc_tree_node(node: ri_session::SessionTreeNode) -> Result<RpcTreeNode> {
    Ok(RpcTreeNode {
        entry: ri_compat::native_entry_to_pi(node.entry).map_err(|error| {
            CliError::runtime("convert a native session tree entry for RPC", error)
        })?,
        children: node
            .children
            .into_iter()
            .map(rpc_tree_node)
            .collect::<Result<Vec<_>>>()?,
        label: node.label,
        label_timestamp: node.label_timestamp.map(|timestamp| timestamp.to_rfc3339()),
    })
}

fn assistant_text(value: &Value) -> Option<String> {
    if value.get("role").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    Some(
        value
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
    )
}

fn message_text(value: &Value) -> String {
    match value.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(content)) => content
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn resource_commands(adapter: &SdkCliRuntime) -> Vec<SlashCommand> {
    let resources = adapter.resources.resources();
    let skills = resources.skills.iter().map(|skill| SlashCommand {
        name: skill.name.clone(),
        description: Some(skill.description.clone()),
        source: SlashCommandSource::Skill,
        source_info: source_info(&skill.source, &skill.name),
    });
    let prompts = resources
        .prompt_templates
        .iter()
        .map(|prompt| SlashCommand {
            name: prompt.name.clone(),
            description: prompt.description.clone(),
            source: SlashCommandSource::Prompt,
            source_info: source_info(&prompt.source, &prompt.name),
        });
    skills.chain(prompts).collect()
}

fn source_info(path: &str, source: &str) -> SourceInfo {
    SourceInfo {
        path: path.to_owned(),
        source: source.to_owned(),
        scope: SourceScope::Temporary,
        origin: SourceOrigin::TopLevel,
        base_dir: None,
    }
}

fn next<T>(values: &[T], current: usize) -> Option<&T> {
    if values.is_empty() {
        None
    } else {
        values.get(current.saturating_add(1) % values.len())
    }
}

fn rpc_usage(usage: &ri_ai::Usage) -> RpcUsage {
    RpcUsage {
        input: usage.input,
        output: usage.output,
        cache_read: usage.cache_read,
        cache_write: usage.cache_write,
        cache_write1h: usage.cache_write_1h,
        reasoning: usage.reasoning,
        total_tokens: Some(usage.total_tokens),
        cost: Some(RpcUsageCost {
            input: usage.cost.input,
            output: usage.cost.output,
            cache_read: usage.cost.cache_read,
            cache_write: usage.cost.cache_write,
            total: usage.cost.total,
        }),
    }
}

fn command_image(image: ri_rpc::CommandImage) -> ri_ai::ImageContent {
    ri_ai::ImageContent {
        data: image.data,
        mime_type: image.mime_type,
    }
}
