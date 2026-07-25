use std::{
    collections::VecDeque,
    future::{self, Future},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use futures::StreamExt;
use parking_lot::Mutex;
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use super::*;

fn model() -> ri_ai::Model {
    ri_ai::Model::new(
        "test-provider",
        "test-model",
        "test-api",
        "https://invalid.test",
    )
}

fn user(text: &str) -> ri_ai::Message {
    let mut message = ri_ai::UserMessage::new(text);
    message.timestamp = 1;
    ri_ai::Message::User(message)
}

fn assistant(
    content: Vec<ri_ai::ContentBlock>,
    reason: ri_ai::StopReason,
) -> ri_ai::AssistantMessage {
    let mut message = ri_ai::AssistantMessage::empty("test-api", "test-provider", "test-model");
    message.content = content;
    message.stop_reason = reason;
    message.timestamp = 2;
    message
}

fn text_assistant(text: &str) -> ri_ai::AssistantMessage {
    assistant(
        vec![ri_ai::ContentBlock::Text(ri_ai::TextContent::new(text))],
        ri_ai::StopReason::Stop,
    )
}

fn call(id: &str, name: &str, arguments: Value) -> ri_ai::ToolCall {
    ri_ai::ToolCall {
        id: id.to_owned(),
        name: name.to_owned(),
        arguments,
        thought_signature: None,
    }
}

fn scripted_stream(
    messages: Vec<ri_ai::AssistantMessage>,
) -> (
    Arc<dyn StreamFn>,
    Arc<Mutex<Vec<ri_ai::Context>>>,
    Arc<AtomicUsize>,
) {
    let messages = Arc::new(Mutex::new(VecDeque::from(messages)));
    let contexts = Arc::new(Mutex::new(Vec::new()));
    let calls = Arc::new(AtomicUsize::new(0));
    let captured_contexts = Arc::clone(&contexts);
    let call_count = Arc::clone(&calls);
    let stream: Arc<dyn StreamFn> = Arc::new(
        move |_model: ri_ai::Model, context: ri_ai::Context, _options: StreamOptions| {
            captured_contexts.lock().push(context);
            call_count.fetch_add(1, Ordering::SeqCst);
            let message = messages.lock().pop_front();
            async move {
                let message =
                    message.ok_or_else(|| ri_ai::AiError::Stream("script exhausted".to_owned()))?;
                let reason = message.stop_reason;
                Ok(ri_ai::AssistantEventStream::completed(
                    ri_ai::AssistantMessageEvent::Done { reason, message },
                ))
            }
        },
    );
    (stream, contexts, calls)
}

fn recording_sink<M: Clone + Send + 'static>()
-> (SharedEventSink<M>, Arc<Mutex<Vec<AgentEvent<M>>>>) {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = Arc::clone(&events);
    let sink: SharedEventSink<M> = Arc::new(move |event| {
        captured.lock().push(event);
        future::ready(Ok(()))
    });
    (sink, events)
}

fn event_kinds<M>(events: &[AgentEvent<M>]) -> Vec<AgentEventKind> {
    events.iter().map(AgentEvent::kind).collect()
}

mod events {
    use super::*;

    #[tokio::test]
    async fn basic() {
        let stream: Arc<dyn StreamFn> = Arc::new(
            |_model: ri_ai::Model, _context: ri_ai::Context, _options: StreamOptions| async move {
                let (mut sender, stream) = ri_ai::create_assistant_message_event_stream();
                let empty = assistant(Vec::new(), ri_ai::StopReason::Stop);
                sender.send(ri_ai::AssistantMessageEvent::Start { partial: empty });
                let partial = text_assistant("hel");
                sender.send(ri_ai::AssistantMessageEvent::TextDelta {
                    content_index: 0,
                    delta: "hel".to_owned(),
                    partial,
                });
                let final_message = text_assistant("hello");
                sender.send(ri_ai::AssistantMessageEvent::Done {
                    reason: ri_ai::StopReason::Stop,
                    message: final_message,
                });
                sender.close();
                Ok(stream)
            },
        );
        let (sink, events) = recording_sink();
        let produced = run_agent_loop(
            vec![user("hi")],
            AgentContext::new("system", Vec::new()),
            AgentLoopConfig::new(model()),
            CancellationToken::new(),
            sink,
            stream,
        )
        .await
        .expect("basic run");

        assert_eq!(produced.len(), 2);
        assert_eq!(
            event_kinds(&events.lock()),
            vec![
                AgentEventKind::AgentStart,
                AgentEventKind::TurnStart,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageEnd,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageUpdate,
                AgentEventKind::MessageEnd,
                AgentEventKind::TurnEnd,
                AgentEventKind::AgentEnd,
            ]
        );
    }

    #[tokio::test]
    async fn event_stream_result_is_independently_awaitable() {
        let (stream_fn, _, _) = scripted_stream(vec![text_assistant("done")]);
        let mut stream = agent_loop(
            vec![user("hello")],
            AgentContext::new("", Vec::new()),
            AgentLoopConfig::new(model()),
            CancellationToken::new(),
            stream_fn,
        );
        let result = stream.result().await.expect("stream result");
        assert_eq!(result.len(), 2);
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            events.push(event.kind());
        }
        assert_eq!(events.last(), Some(&AgentEventKind::AgentEnd));
    }

    #[derive(Clone, Debug, PartialEq)]
    enum AppMessage {
        Llm(Box<ri_ai::Message>),
        Notice(String),
    }

    impl AgentMessage for AppMessage {
        fn as_llm_message(&self) -> Option<&ri_ai::Message> {
            match self {
                Self::Llm(message) => Some(message.as_ref()),
                Self::Notice(_) => None,
            }
        }

        fn from_llm_message(message: ri_ai::Message) -> Self {
            Self::Llm(Box::new(message))
        }
    }

    #[tokio::test]
    async fn transform_precedes_extensible_projection() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let transform_order = Arc::clone(&order);
        let convert_order = Arc::clone(&order);
        let config = AgentLoopConfig::new(model())
            .with_transform_context(move |mut messages, _| {
                transform_order.lock().push("transform");
                messages.push(AppMessage::Notice("ui-only".to_owned()));
                async move { Ok(messages) }
            })
            .with_convert_to_llm(move |messages| {
                convert_order.lock().push("convert");
                async move { Ok(messages.iter().filter_map(AgentMessage::project).collect()) }
            });
        let (stream, contexts, _) = scripted_stream(vec![text_assistant("ok")]);
        let (sink, _) = recording_sink();
        run_agent_loop(
            vec![AppMessage::Llm(Box::new(user("prompt")))],
            AgentContext {
                system_prompt: String::new(),
                messages: vec![AppMessage::Notice("old-ui".to_owned())],
                tools: Vec::new(),
            },
            config,
            CancellationToken::new(),
            sink,
            stream,
        )
        .await
        .expect("custom projection");

        assert_eq!(*order.lock(), vec!["transform", "convert"]);
        assert_eq!(contexts.lock()[0].messages, vec![user("prompt")]);
    }

    #[tokio::test]
    async fn listener_barrier_delays_prompt_and_idle() {
        let agent = Agent::<ri_ai::Message>::new(
            model(),
            |_model: ri_ai::Model, _context: ri_ai::Context, _options: StreamOptions| async move {
                Ok(ri_ai::AssistantEventStream::completed(
                    ri_ai::AssistantMessageEvent::Done {
                        reason: ri_ai::StopReason::Stop,
                        message: text_assistant("ok"),
                    },
                ))
            },
        );
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let order = Arc::new(Mutex::new(Vec::new()));
        let first_entered = Arc::clone(&entered);
        let first_release = Arc::clone(&release);
        let first_order = Arc::clone(&order);
        agent.subscribe(move |event: AgentEvent<ri_ai::Message>, _| {
            let entered = Arc::clone(&first_entered);
            let release = Arc::clone(&first_release);
            let order = Arc::clone(&first_order);
            async move {
                if matches!(event, AgentEvent::AgentEnd { .. }) {
                    order.lock().push("first-enter");
                    entered.notify_one();
                    release.notified().await;
                    order.lock().push("first-exit");
                }
            }
        });
        let second_order = Arc::clone(&order);
        agent.subscribe(move |event, _| {
            let order = Arc::clone(&second_order);
            async move {
                if matches!(event, AgentEvent::AgentEnd { .. }) {
                    order.lock().push("second");
                }
            }
        });

        let prompt_agent = agent.clone();
        let prompt = tokio::spawn(async move { prompt_agent.prompt("hello").await });
        entered.notified().await;
        assert!(agent.state().is_streaming);
        assert_eq!(*order.lock(), vec!["first-enter"]);

        let idle_agent = agent.clone();
        let idle = tokio::spawn(async move { idle_agent.wait_for_idle().await });
        tokio::task::yield_now().await;
        assert!(!prompt.is_finished());
        assert!(!idle.is_finished());

        release.notify_one();
        prompt.await.expect("prompt task").expect("prompt");
        idle.await.expect("idle task");
        assert_eq!(*order.lock(), vec!["first-enter", "first-exit", "second"]);
        assert!(!agent.state().is_streaming);
    }

    #[tokio::test]
    async fn thrown_stream_failure_completes_lifecycle() {
        let agent = Agent::<ri_ai::Message>::new(
            model(),
            |_model: ri_ai::Model, _context: ri_ai::Context, _options: StreamOptions| async move {
                Err(ri_ai::AiError::Stream("provider exploded".to_owned()))
            },
        );
        let trace = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&trace);
        agent.subscribe(move |event: AgentEvent<ri_ai::Message>, _| {
            captured.lock().push(event.kind());
            future::ready(())
        });
        agent.prompt("hello").await.expect("encoded failure");

        assert_eq!(
            *trace.lock(),
            vec![
                AgentEventKind::AgentStart,
                AgentEventKind::TurnStart,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageEnd,
                AgentEventKind::MessageStart,
                AgentEventKind::MessageEnd,
                AgentEventKind::TurnEnd,
                AgentEventKind::AgentEnd,
            ]
        );
        let state = agent.state();
        assert_eq!(
            state.error_message.as_deref(),
            Some("stream error: provider exploded")
        );
        assert!(matches!(
            state.messages.last(),
            Some(ri_ai::Message::Assistant(message))
                if message.stop_reason == ri_ai::StopReason::Error
        ));
    }
}

mod tools {
    use super::*;

    fn echo_tool<F, Fut>(handler: F) -> Arc<dyn Tool>
    where
        F: Fn(ToolCallContext, Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolResult, ToolError>> + Send + 'static,
    {
        Arc::new(FnTool::new(
            "echo",
            "Echo",
            "Echo a value",
            json!({
                "type": "object",
                "properties": {"value": {}},
                "required": ["value"]
            }),
            handler,
        ))
    }

    #[tokio::test]
    async fn parallel_completion_and_persistence_order() {
        let release_first = Arc::new(Notify::new());
        let second_end = Arc::new(Notify::new());
        let release = Arc::clone(&release_first);
        let tool = echo_tool(move |_context, arguments| {
            let release = Arc::clone(&release);
            async move {
                if arguments["value"] == "first" {
                    release.notified().await;
                }
                Ok(ToolResult::text(
                    arguments["value"].as_str().unwrap_or_default(),
                ))
            }
        });
        let first = assistant(
            vec![
                ri_ai::ContentBlock::ToolCall(call("call-1", "echo", json!({"value": "first"}))),
                ri_ai::ContentBlock::ToolCall(call("call-2", "echo", json!({"value": "second"}))),
            ],
            ri_ai::StopReason::ToolUse,
        );
        let (stream, _, _) = scripted_stream(vec![first, text_assistant("done")]);
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let observed_second = Arc::clone(&second_end);
        let sink: SharedEventSink<ri_ai::Message> = Arc::new(move |event| {
            if matches!(
                &event,
                AgentEvent::ToolExecutionEnd {
                    tool_call_id,
                    ..
                } if tool_call_id == "call-2"
            ) {
                observed_second.notify_one();
            }
            captured.lock().push(event);
            future::ready(Ok(()))
        });
        let run = tokio::spawn(run_agent_loop(
            vec![user("run")],
            AgentContext {
                system_prompt: String::new(),
                messages: Vec::new(),
                tools: vec![tool],
            },
            AgentLoopConfig::new(model()),
            CancellationToken::new(),
            sink,
            stream,
        ));
        second_end.notified().await;
        release_first.notify_one();
        run.await.expect("run task").expect("parallel run");

        let events = events.lock();
        let starts = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolExecutionStart { tool_call_id, .. } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let ends = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let persisted = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::MessageEnd {
                    message: ri_ai::Message::ToolResult(message),
                } => Some(message.tool_call_id.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(starts, vec!["call-1", "call-2"]);
        assert_eq!(ends, vec!["call-2", "call-1"]);
        assert_eq!(persisted, vec!["call-1", "call-2"]);
    }

    #[tokio::test]
    async fn one_sequential_tool_serializes_whole_batch() {
        let slow_started = Arc::new(Notify::new());
        let release_slow = Arc::new(Notify::new());
        let fast_started = Arc::new(AtomicBool::new(false));
        let entered = Arc::clone(&slow_started);
        let release = Arc::clone(&release_slow);
        let slow: Arc<dyn Tool> = Arc::new(
            FnTool::new(
                "slow",
                "Slow",
                "slow",
                json!({"type": "object"}),
                move |_context, _arguments| {
                    let entered = Arc::clone(&entered);
                    let release = Arc::clone(&release);
                    async move {
                        entered.notify_one();
                        release.notified().await;
                        Ok(ToolResult::text("slow"))
                    }
                },
            )
            .with_execution_mode(ToolExecutionMode::Sequential),
        );
        let fast_flag = Arc::clone(&fast_started);
        let fast: Arc<dyn Tool> = Arc::new(FnTool::new(
            "fast",
            "Fast",
            "fast",
            json!({"type": "object"}),
            move |_context, _arguments| {
                fast_flag.store(true, Ordering::SeqCst);
                async move { Ok(ToolResult::text("fast")) }
            },
        ));
        let first = assistant(
            vec![
                ri_ai::ContentBlock::ToolCall(call("slow-id", "slow", json!({}))),
                ri_ai::ContentBlock::ToolCall(call("fast-id", "fast", json!({}))),
            ],
            ri_ai::StopReason::ToolUse,
        );
        let (stream, _, _) = scripted_stream(vec![first, text_assistant("done")]);
        let (sink, _) = recording_sink();
        let run = tokio::spawn(run_agent_loop(
            vec![user("run")],
            AgentContext {
                system_prompt: String::new(),
                messages: Vec::new(),
                tools: vec![slow, fast],
            },
            AgentLoopConfig::new(model()),
            CancellationToken::new(),
            sink,
            stream,
        ));
        slow_started.notified().await;
        tokio::task::yield_now().await;
        assert!(!fast_started.load(Ordering::SeqCst));
        release_slow.notify_one();
        run.await.expect("run task").expect("sequential run");
        assert!(fast_started.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn prepare_validate_hooks_and_terminate() {
        let executed = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&executed);
        let tool: Arc<dyn Tool> = Arc::new(
            FnTool::new(
                "edit",
                "Edit",
                "edit",
                json!({
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": false
                }),
                move |_context, arguments| {
                    captured.lock().push(arguments);
                    async move { Ok(ToolResult::text("original")) }
                },
            )
            .with_prepare_arguments(|arguments| Ok(json!({"value": arguments["legacy"].clone()}))),
        );
        let config = AgentLoopConfig::new(model())
            .with_before_tool_call(|context, _| async move {
                assert_eq!(context.arguments, json!({"value": "prepared"}));
                Ok(BeforeToolCallResult {
                    arguments: Some(json!({"value": 123})),
                    ..BeforeToolCallResult::default()
                })
            })
            .with_after_tool_call(|context, _| async move {
                assert_eq!(context.arguments, json!({"value": 123}));
                Ok(AfterToolCallResult {
                    content: Some(vec![ri_ai::message::InputContent::Text(
                        ri_ai::TextContent::new("patched"),
                    )]),
                    terminate: Some(true),
                    ..AfterToolCallResult::default()
                })
            });
        let first = assistant(
            vec![ri_ai::ContentBlock::ToolCall(call(
                "edit-id",
                "edit",
                json!({"legacy": "prepared"}),
            ))],
            ri_ai::StopReason::ToolUse,
        );
        let (stream, _, calls) = scripted_stream(vec![first]);
        let (sink, events) = recording_sink();
        run_agent_loop(
            vec![user("edit")],
            AgentContext {
                system_prompt: String::new(),
                messages: Vec::new(),
                tools: vec![tool],
            },
            config,
            CancellationToken::new(),
            sink,
            stream,
        )
        .await
        .expect("hook run");

        assert_eq!(*executed.lock(), vec![json!({"value": 123})]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(events.lock().iter().any(|event| matches!(
            event,
            AgentEvent::MessageEnd {
                message: ri_ai::Message::ToolResult(result),
            } if matches!(
                result.content.first(),
                Some(ri_ai::message::InputContent::Text(text))
                    if text.text == "patched"
            )
        )));
    }

    #[tokio::test]
    async fn missing_invalid_and_blocked_calls_become_ordered_errors() {
        let executions = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&executions);
        let tool: Arc<dyn Tool> = Arc::new(FnTool::new(
            "echo",
            "Echo",
            "echo",
            json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            }),
            move |_context, _arguments| {
                count.fetch_add(1, Ordering::SeqCst);
                async move { Ok(ToolResult::text("unexpected")) }
            },
        ));
        let config = AgentLoopConfig::new(model()).with_before_tool_call(|context, _| async move {
            Ok(BeforeToolCallResult {
                block: context.arguments["value"] == "blocked",
                reason: Some("policy blocked this call".to_owned()),
                arguments: None,
            })
        });
        let first = assistant(
            vec![
                ri_ai::ContentBlock::ToolCall(call("missing-id", "missing", json!({}))),
                ri_ai::ContentBlock::ToolCall(call(
                    "invalid-id",
                    "echo",
                    json!({"value": {"not": "text"}}),
                )),
                ri_ai::ContentBlock::ToolCall(call(
                    "blocked-id",
                    "echo",
                    json!({"value": "blocked"}),
                )),
            ],
            ri_ai::StopReason::ToolUse,
        );
        let (stream, _, _) = scripted_stream(vec![first, text_assistant("done")]);
        let (sink, events) = recording_sink();
        run_agent_loop(
            vec![user("run")],
            AgentContext {
                system_prompt: String::new(),
                messages: Vec::new(),
                tools: vec![tool],
            },
            config,
            CancellationToken::new(),
            sink,
            stream,
        )
        .await
        .expect("error-result run");

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        let error_ids = events
            .lock()
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolExecutionEnd {
                    tool_call_id,
                    is_error: true,
                    ..
                } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            error_ids,
            vec![
                "missing-id".to_owned(),
                "invalid-id".to_owned(),
                "blocked-id".to_owned(),
            ]
        );
    }

    #[tokio::test]
    async fn length_repairs_every_call_without_execution() {
        let executions = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&executions);
        let tool = echo_tool(move |_context, _arguments| {
            count.fetch_add(1, Ordering::SeqCst);
            async move { Ok(ToolResult::text("unexpected")) }
        });
        let truncated = assistant(
            vec![
                ri_ai::ContentBlock::ToolCall(call("one", "echo", json!({"value": "cut"}))),
                ri_ai::ContentBlock::ToolCall(call("two", "echo", json!({"value": "cut"}))),
            ],
            ri_ai::StopReason::Length,
        );
        let (stream, _, calls) = scripted_stream(vec![truncated, text_assistant("repaired")]);
        let (sink, events) = recording_sink();
        run_agent_loop(
            vec![user("run")],
            AgentContext {
                system_prompt: String::new(),
                messages: Vec::new(),
                tools: vec![tool],
            },
            AgentLoopConfig::new(model()),
            CancellationToken::new(),
            sink,
            stream,
        )
        .await
        .expect("length repair");

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let failed = events
            .lock()
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    AgentEvent::ToolExecutionEnd {
                        is_error: true,
                        result,
                        ..
                    } if matches!(
                        result.content.first(),
                        Some(ri_ai::message::InputContent::Text(text))
                            if text.text.contains("output token limit")
                    )
                )
            })
            .count();
        assert_eq!(failed, 2);
    }

    #[tokio::test]
    async fn mixed_terminate_results_continue() {
        let tool = echo_tool(move |_context, arguments| async move {
            let terminate = arguments["value"] == "stop";
            Ok(ToolResult {
                terminate,
                ..ToolResult::text("ok")
            })
        });
        let first = assistant(
            vec![
                ri_ai::ContentBlock::ToolCall(call("one", "echo", json!({"value": "stop"}))),
                ri_ai::ContentBlock::ToolCall(call("two", "echo", json!({"value": "continue"}))),
            ],
            ri_ai::StopReason::ToolUse,
        );
        let (stream, _, calls) = scripted_stream(vec![first, text_assistant("done")]);
        let (sink, _) = recording_sink();
        run_agent_loop(
            vec![user("run")],
            AgentContext {
                system_prompt: String::new(),
                messages: Vec::new(),
                tools: vec![tool],
            },
            AgentLoopConfig::new(model()),
            CancellationToken::new(),
            sink,
            stream,
        )
        .await
        .expect("mixed terminate");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn settled_parallel_tool_suppresses_late_update() {
        let saved_update = Arc::new(Mutex::new(None::<ToolUpdateSink>));
        let saved = Arc::clone(&saved_update);
        let settled: Arc<dyn Tool> = Arc::new(FnTool::new(
            "settled",
            "Settled",
            "settled",
            json!({"type": "object"}),
            move |context, _arguments| {
                *saved.lock() = Some(context.updates.clone());
                async move {
                    context
                        .updates
                        .send(ToolResult::text("running"))
                        .await
                        .map_err(|error| ToolError::message(error.to_string()))?;
                    Ok(ToolResult {
                        terminate: true,
                        ..ToolResult::text("done")
                    })
                }
            },
        ));
        let slow_started = Arc::new(Notify::new());
        let release_slow = Arc::new(Notify::new());
        let started = Arc::clone(&slow_started);
        let release = Arc::clone(&release_slow);
        let slow: Arc<dyn Tool> = Arc::new(FnTool::new(
            "slow",
            "Slow",
            "slow",
            json!({"type": "object"}),
            move |_context, _arguments| {
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                async move {
                    started.notify_one();
                    release.notified().await;
                    Ok(ToolResult {
                        terminate: true,
                        ..ToolResult::text("done")
                    })
                }
            },
        ));
        let tool_message = assistant(
            vec![
                ri_ai::ContentBlock::ToolCall(call("settled-id", "settled", json!({}))),
                ri_ai::ContentBlock::ToolCall(call("slow-id", "slow", json!({}))),
            ],
            ri_ai::StopReason::ToolUse,
        );
        let (stream, _, _) = scripted_stream(vec![tool_message]);
        let settled_end = Arc::new(Notify::new());
        let observed_end = Arc::clone(&settled_end);
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let sink: SharedEventSink<ri_ai::Message> = Arc::new(move |event| {
            if matches!(
                &event,
                AgentEvent::ToolExecutionEnd {
                    tool_call_id,
                    ..
                } if tool_call_id == "settled-id"
            ) {
                observed_end.notify_one();
            }
            captured.lock().push(event);
            future::ready(Ok(()))
        });
        let run = tokio::spawn(run_agent_loop(
            vec![user("run")],
            AgentContext {
                system_prompt: String::new(),
                messages: Vec::new(),
                tools: vec![settled, slow],
            },
            AgentLoopConfig::new(model()),
            CancellationToken::new(),
            sink,
            stream,
        ));
        settled_end.notified().await;
        slow_started.notified().await;

        let update = saved_update.lock().clone().expect("captured update sink");
        assert!(
            !update
                .send(ToolResult::text("late"))
                .await
                .expect("late update is harmless")
        );
        assert_eq!(
            events
                .lock()
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolExecutionUpdate { .. }))
                .count(),
            1
        );
        release_slow.notify_one();
        run.await.expect("run task").expect("late-update run");
    }

    #[tokio::test]
    async fn next_turn_snapshot_applies_before_provider_request() {
        let tool = echo_tool(move |_context, _arguments| async move { Ok(ToolResult::text("ok")) });
        let updated = Arc::new(AtomicBool::new(false));
        let update_once = Arc::clone(&updated);
        let config = AgentLoopConfig::new(model()).with_prepare_next_turn(move |turn, _| {
            let first = !update_once.swap(true, Ordering::SeqCst);
            async move {
                if first {
                    let mut context = turn.context;
                    context.system_prompt = "next-system".to_owned();
                    Ok(Some(AgentLoopTurnUpdate {
                        context: Some(context),
                        ..AgentLoopTurnUpdate::default()
                    }))
                } else {
                    Ok(None)
                }
            }
        });
        let first = assistant(
            vec![ri_ai::ContentBlock::ToolCall(call(
                "one",
                "echo",
                json!({"value": "x"}),
            ))],
            ri_ai::StopReason::ToolUse,
        );
        let (stream, contexts, _) = scripted_stream(vec![first, text_assistant("done")]);
        let (sink, _) = recording_sink();
        run_agent_loop(
            vec![user("run")],
            AgentContext {
                system_prompt: "first-system".to_owned(),
                messages: Vec::new(),
                tools: vec![tool],
            },
            config,
            CancellationToken::new(),
            sink,
            stream,
        )
        .await
        .expect("snapshot run");
        assert_eq!(
            contexts.lock()[1].system_prompt.as_deref(),
            Some("next-system")
        );
    }
}

mod queue {
    use super::*;

    fn user_texts(context: &ri_ai::Context) -> Vec<String> {
        context
            .messages
            .iter()
            .filter_map(|message| match message {
                ri_ai::Message::User(message) => match &message.content {
                    ri_ai::UserContent::Text(text) => Some(text.clone()),
                    ri_ai::UserContent::Blocks(_) => None,
                },
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn steer_then_follow_one_at_a_time() {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&contexts);
        let agent = Agent::<ri_ai::Message>::new(
            model(),
            move |_model: ri_ai::Model, context: ri_ai::Context, _options: StreamOptions| {
                captured.lock().push(context);
                async move {
                    Ok(ri_ai::AssistantEventStream::completed(
                        ri_ai::AssistantMessageEvent::Done {
                            reason: ri_ai::StopReason::Stop,
                            message: text_assistant("ok"),
                        },
                    ))
                }
            },
        );
        agent.steer(user("steer-1"));
        agent.steer(user("steer-2"));
        agent.follow_up(user("follow"));
        agent.prompt("root").await.expect("queued run");

        let contexts = contexts.lock();
        assert_eq!(contexts.len(), 3);
        assert_eq!(user_texts(&contexts[0]), vec!["root", "steer-1"]);
        assert_eq!(user_texts(&contexts[1]), vec!["root", "steer-1", "steer-2"]);
        assert_eq!(
            user_texts(&contexts[2]),
            vec!["root", "steer-1", "steer-2", "follow"]
        );
    }

    #[tokio::test]
    async fn all_mode_drains_one_provider_turn() {
        let calls = Arc::new(AtomicUsize::new(0));
        let count = Arc::clone(&calls);
        let agent = Agent::<ri_ai::Message>::new(
            model(),
            move |_model: ri_ai::Model, _context: ri_ai::Context, _options: StreamOptions| {
                count.fetch_add(1, Ordering::SeqCst);
                async move {
                    Ok(ri_ai::AssistantEventStream::completed(
                        ri_ai::AssistantMessageEvent::Done {
                            reason: ri_ai::StopReason::Stop,
                            message: text_assistant("ok"),
                        },
                    ))
                }
            },
        );
        agent.set_steering_mode(QueueMode::All);
        agent.steer(user("one"));
        agent.steer(user("two"));
        agent.prompt("root").await.expect("all-mode run");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn assistant_tail_continue_prioritizes_steering() {
        let contexts = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&contexts);
        let agent = Agent::<ri_ai::Message>::new(
            model(),
            move |_model: ri_ai::Model, context: ri_ai::Context, _options: StreamOptions| {
                captured.lock().push(context);
                async move {
                    Ok(ri_ai::AssistantEventStream::completed(
                        ri_ai::AssistantMessageEvent::Done {
                            reason: ri_ai::StopReason::Stop,
                            message: text_assistant("ok"),
                        },
                    ))
                }
            },
        );
        agent.set_messages(vec![
            user("initial"),
            ri_ai::Message::Assistant(text_assistant("initial response")),
        ]);
        agent.steer(user("steer-1"));
        agent.steer(user("steer-2"));
        agent.follow_up(user("follow"));
        agent.continue_run().await.expect("continue queued work");

        let contexts = contexts.lock();
        assert_eq!(contexts.len(), 3);
        let first = user_texts(&contexts[0]);
        let second = user_texts(&contexts[1]);
        let third = user_texts(&contexts[2]);
        assert_eq!(first.last().map(String::as_str), Some("steer-1"));
        assert_eq!(second.last().map(String::as_str), Some("steer-2"));
        assert_eq!(third.last().map(String::as_str), Some("follow"));
    }
}

mod cancel {
    use super::*;

    #[tokio::test]
    async fn abort_idle_is_noop_and_active_abort_completes() {
        let agent = Agent::<ri_ai::Message>::new(
            model(),
            |_model: ri_ai::Model, _context: ri_ai::Context, options: StreamOptions| async move {
                let (mut sender, stream) = ri_ai::create_assistant_message_event_stream();
                sender.send(ri_ai::AssistantMessageEvent::Start {
                    partial: assistant(Vec::new(), ri_ai::StopReason::Stop),
                });
                tokio::spawn(async move {
                    options.cancellation.cancelled().await;
                    let mut aborted = text_assistant("");
                    aborted.stop_reason = ri_ai::StopReason::Aborted;
                    aborted.error_message = Some("cancelled".to_owned());
                    sender.send(ri_ai::AssistantMessageEvent::Error {
                        reason: ri_ai::StopReason::Aborted,
                        error: aborted,
                    });
                    sender.close();
                });
                Ok(stream)
            },
        );
        agent.abort();
        let assistant_started = Arc::new(Notify::new());
        let observed = Arc::clone(&assistant_started);
        let trace = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&trace);
        agent.subscribe(
            move |event: AgentEvent<ri_ai::Message>, token: CancellationToken| {
                if matches!(
                    &event,
                    AgentEvent::MessageStart {
                        message: ri_ai::Message::Assistant(_),
                    }
                ) {
                    observed.notify_one();
                }
                captured.lock().push((event.kind(), token.is_cancelled()));
                future::ready(())
            },
        );

        let running_agent = agent.clone();
        let run = tokio::spawn(async move { running_agent.prompt("run").await });
        assistant_started.notified().await;
        agent.abort();
        run.await.expect("prompt task").expect("aborted lifecycle");

        let state = agent.state();
        assert!(!state.is_streaming);
        assert!(matches!(
            state.messages.last(),
            Some(ri_ai::Message::Assistant(message))
                if message.stop_reason == ri_ai::StopReason::Aborted
        ));
        assert_eq!(
            trace.lock().last().map(|entry| entry.0),
            Some(AgentEventKind::AgentEnd)
        );
        assert!(trace.lock().last().is_some_and(|entry| entry.1));
    }
}
