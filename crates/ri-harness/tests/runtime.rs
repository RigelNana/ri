//! Deterministic end-to-end coverage for the unified harness lifecycle.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use ri_ai::{
    AssistantMessage, ContentBlock, ImageContent, InputContent, Message, Model, StopReason,
    TextContent, ThinkingLevel, Usage, UserContent, UserMessage,
};
use ri_harness::{
    BackendError, BackendErrorKind, BeforeAgentStart, BeforeAgentStartResult, CompactionSettings,
    Harness, HarnessBackend, HarnessConfig, HarnessEvent, HarnessHooks, HarnessObserver,
    HookContext, InputAction, InputEvent, NavigateOptions, Phase, PromptOptions, PromptOutcome,
    PromptTemplate, QueueMode, RequestOptions, Resources, RetryOperation, RetryPolicy,
    SessionWrite, SummaryKind, SummaryRequest, SummaryResponse, TurnOutput, TurnRequest,
};
use ri_session::{CreateOptions, MemoryRepository, Repository, SessionEntry};
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;

#[derive(Debug)]
struct ScriptBackend {
    turns: Mutex<VecDeque<Result<TurnOutput, BackendError>>>,
    summaries: Mutex<VecDeque<Result<SummaryResponse, BackendError>>>,
    summary_requests: Mutex<Vec<SummaryRequest>>,
    requests: Mutex<Vec<TurnRequest>>,
    block_first: bool,
    first_started: Notify,
    first_release: Notify,
    first_was_blocked: AtomicBool,
}

impl ScriptBackend {
    fn new(turns: impl IntoIterator<Item = Result<TurnOutput, BackendError>>) -> Self {
        Self {
            turns: Mutex::new(turns.into_iter().collect()),
            summaries: Mutex::new(VecDeque::new()),
            summary_requests: Mutex::new(Vec::new()),
            requests: Mutex::new(Vec::new()),
            block_first: false,
            first_started: Notify::new(),
            first_release: Notify::new(),
            first_was_blocked: AtomicBool::new(false),
        }
    }

    fn blocking(turns: impl IntoIterator<Item = Result<TurnOutput, BackendError>>) -> Self {
        Self {
            block_first: true,
            ..Self::new(turns)
        }
    }

    async fn requests(&self) -> Vec<TurnRequest> {
        self.requests.lock().await.clone()
    }
}

#[async_trait]
impl HarnessBackend for ScriptBackend {
    async fn preflight(&self, _model: &Model) -> Result<(), BackendError> {
        Ok(())
    }

    async fn execute_turn(
        &self,
        request: TurnRequest,
        _cancellation: CancellationToken,
    ) -> Result<TurnOutput, BackendError> {
        let first = {
            let mut requests = self.requests.lock().await;
            let first = requests.is_empty();
            requests.push(request);
            first
        };
        if first && self.block_first {
            self.first_was_blocked.store(true, Ordering::Release);
            self.first_started.notify_one();
            self.first_release.notified().await;
        }
        self.turns.lock().await.pop_front().expect("scripted turn")
    }

    async fn summarize(
        &self,
        request: SummaryRequest,
        _cancellation: CancellationToken,
    ) -> Result<SummaryResponse, BackendError> {
        self.summary_requests.lock().await.push(request);
        self.summaries
            .lock()
            .await
            .pop_front()
            .expect("scripted summary")
    }
}

fn assistant(model: &Model, text: &str) -> AssistantMessage {
    let mut message = AssistantMessage::empty(&model.api, &model.provider, &model.id);
    message
        .content
        .push(ContentBlock::Text(TextContent::new(text)));
    message.stop_reason = StopReason::Stop;
    message.usage = Usage::from_parts(10, 2, 0, 0);
    message
}

fn output(model: &Model, text: &str, continue_after_tools: bool) -> TurnOutput {
    TurnOutput {
        messages: vec![Message::Assistant(assistant(model, text))],
        continue_after_tools,
    }
}

fn model() -> Model {
    let mut model = Model::new("test", "model", "test-api", "https://example.test");
    model.context_window = 200;
    model.max_tokens = 50;
    model
}

fn config(model: &Model) -> HarnessConfig {
    HarnessConfig {
        model: Arc::new(model.clone()),
        thinking_level: ThinkingLevel::Off,
        system_prompt: "old system".to_owned(),
        tools: Arc::new([]),
        active_tool_names: Arc::new([]),
        resources: Resources::default(),
        request_options: RequestOptions::default(),
        steering_mode: QueueMode::OneAtATime,
        follow_up_mode: QueueMode::OneAtATime,
        retry: RetryPolicy {
            base_delay: Duration::ZERO,
            max_delay: Duration::ZERO,
            ..RetryPolicy::default()
        },
        compaction: CompactionSettings {
            reserve_tokens: 40,
            keep_recent_tokens: 40,
            ..CompactionSettings::default()
        },
    }
}

async fn session(id: &str) -> ri_session::Session {
    MemoryRepository::default()
        .create(CreateOptions {
            id: Some(id.to_owned()),
            ..CreateOptions::new(".")
        })
        .await
        .expect("session")
}

#[derive(Debug)]
struct SettledWriter {
    harness: Harness,
    called: AtomicBool,
}

#[derive(Debug, Default)]
struct EventRecorder {
    events: Mutex<Vec<HarnessEvent>>,
}

impl EventRecorder {
    async fn events(&self) -> Vec<HarnessEvent> {
        self.events.lock().await.clone()
    }
}

#[derive(Debug, Default)]
struct PipelineHooks {
    stages: Mutex<Vec<String>>,
}

#[async_trait]
impl HarnessHooks for PipelineHooks {
    async fn command(&self, _context: &HookContext, input: &str) -> ri_harness::Result<bool> {
        self.stages.lock().await.push(format!("command:{input}"));
        Ok(input == "/handled")
    }

    async fn input(
        &self,
        _context: &HookContext,
        event: InputEvent,
    ) -> ri_harness::Result<InputAction> {
        self.stages
            .lock()
            .await
            .push(format!("input:{}", event.text));
        Ok(if event.text == "/alias" {
            InputAction::Transform {
                text: "/greet world".into(),
                images: None,
            }
        } else {
            InputAction::Continue
        })
    }

    async fn before_agent_start(
        &self,
        _context: &HookContext,
        event: BeforeAgentStart,
    ) -> ri_harness::Result<BeforeAgentStartResult> {
        self.stages
            .lock()
            .await
            .push(format!("before:{}", event.prompt));
        Ok(BeforeAgentStartResult::default())
    }
}

#[async_trait]
impl HarnessObserver for SettledWriter {
    async fn on_event(&self, event: &HarnessEvent) -> ri_harness::Result<()> {
        if matches!(event, HarnessEvent::Settled { .. }) {
            self.harness
                .write_session(SessionWrite::Custom {
                    kind: "settled-observer".to_owned(),
                    data: Some(serde_json::json!({"done": true})),
                })
                .await?;
            tokio::time::sleep(Duration::from_millis(5)).await;
            self.called.store(true, Ordering::Release);
        }
        Ok(())
    }
}

#[async_trait]
impl HarnessObserver for EventRecorder {
    async fn on_event(&self, event: &HarnessEvent) -> ri_harness::Result<()> {
        self.events.lock().await.push(event.clone());
        Ok(())
    }
}

#[tokio::test]
async fn snapshots_refresh_only_at_save_points_and_settlement_flushes_callbacks() {
    let model = model();
    let backend = Arc::new(ScriptBackend::blocking([
        Ok(output(&model, "first", false)),
        Ok(output(&model, "second", false)),
    ]));
    let harness = Harness::new(
        session("snapshots").await,
        config(&model),
        backend.clone(),
        None,
    )
    .await
    .expect("harness");
    let observer = Arc::new(SettledWriter {
        harness: harness.clone(),
        called: AtomicBool::new(false),
    });
    harness.add_observer(observer.clone()).await;

    let running = {
        let harness = harness.clone();
        tokio::spawn(async move {
            harness
                .prompt("initial", PromptOptions::interactive())
                .await
        })
    };
    backend.first_started.notified().await;
    assert!(backend.first_was_blocked.load(Ordering::Acquire));
    harness.set_system_prompt("new system").await;
    harness.steer("steered").await.expect("steer");
    backend.first_release.notify_one();

    assert!(matches!(
        running.await.expect("task").expect("prompt"),
        PromptOutcome::Completed(_)
    ));
    assert!(observer.called.load(Ordering::Acquire));
    let requests = backend.requests().await;
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].snapshot.system_prompt.as_ref(), "old system");
    assert_eq!(requests[1].snapshot.system_prompt.as_ref(), "new system");
    assert!(requests[1].continuation);
    assert!(
        requests[1].snapshot.context.messages.len() > requests[0].snapshot.context.messages.len()
    );

    let branch = harness.session().await.branch().await.expect("branch");
    assert!(branch.iter().any(|entry| {
        matches!(
            &entry.entry,
            SessionEntry::Custom(entry) if entry.custom_type == "settled-observer"
        )
    }));
}

#[tokio::test]
async fn transient_retry_excludes_the_failed_assistant_from_the_next_snapshot() {
    let model = model();
    let backend = Arc::new(ScriptBackend::new([
        Err(BackendError::new(
            BackendErrorKind::Transient,
            "temporarily overloaded",
        )),
        Ok(output(&model, "recovered", false)),
    ]));
    let harness = Harness::new(
        session("retry").await,
        config(&model),
        backend.clone(),
        None,
    )
    .await
    .expect("harness");
    let events = Arc::new(EventRecorder::default());
    harness.add_observer(events.clone()).await;
    let outcome = harness
        .prompt("hello", PromptOptions::interactive())
        .await
        .expect("retry succeeds");
    assert!(
        matches!(outcome, PromptOutcome::Completed(message) if message.error_message.is_none())
    );
    let requests = backend.requests().await;
    assert_eq!(requests.len(), 2);
    assert!(requests[1].continuation);
    assert!(!requests[1].snapshot.context.messages.iter().any(|message| {
        matches!(message, Message::Assistant(message) if message.stop_reason == StopReason::Error)
    }));
    let retry_events = events
        .events()
        .await
        .into_iter()
        .filter(|event| {
            matches!(
                event,
                HarnessEvent::RetryScheduled { .. } | HarnessEvent::RetryFinished { .. }
            )
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        retry_events.as_slice(),
        [
            HarnessEvent::RetryScheduled {
                operation: RetryOperation::Agent,
                attempt: 1,
                max_attempts: 3,
                ..
            },
            HarnessEvent::RetryFinished {
                operation: RetryOperation::Agent,
                success: true,
                attempt: 1,
                final_error: None,
            }
        ]
    ));
}

#[tokio::test]
async fn abort_retry_cancels_only_the_backoff_and_keeps_the_prompt_result() {
    let model = model();
    let backend = Arc::new(ScriptBackend::new([Err(BackendError::new(
        BackendErrorKind::Transient,
        "temporary failure",
    ))]));
    let mut harness_config = config(&model);
    harness_config.retry.base_delay = Duration::from_secs(30);
    harness_config.retry.max_delay = Duration::from_secs(30);
    let harness = Harness::new(
        session("abort-retry").await,
        harness_config,
        backend.clone(),
        None,
    )
    .await
    .expect("harness");
    let events = Arc::new(EventRecorder::default());
    harness.add_observer(events.clone()).await;

    let prompt = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.prompt("hello", PromptOptions::interactive()).await })
    };
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if harness.status().await.phase == Phase::Retry {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("retry phase");

    assert!(harness.abort_retry().await);
    let outcome = tokio::time::timeout(Duration::from_secs(1), prompt)
        .await
        .expect("prompt settles without waiting for the backoff")
        .expect("prompt task")
        .expect("retry cancellation is not a prompt abort");
    let PromptOutcome::Completed(message) = outcome else {
        panic!("expected completed prompt");
    };
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(backend.requests().await.len(), 1);
    assert_eq!(harness.status().await.phase, Phase::Idle);
    assert!(!harness.abort_retry().await);
    assert!(events.events().await.iter().any(|event| matches!(
        event,
        HarnessEvent::RetryFinished {
            operation: RetryOperation::Agent,
            success: false,
            attempt: 1,
            final_error: Some(error),
        } if error == "Retry cancelled"
    )));
}

#[tokio::test]
async fn prompt_pipeline_orders_commands_input_expansion_and_agent_start() {
    let model = model();
    let backend = Arc::new(ScriptBackend::new([Ok(output(&model, "done", false))]));
    let hooks = Arc::new(PipelineHooks::default());
    let mut harness_config = config(&model);
    harness_config.resources = Resources::new(
        Vec::new(),
        vec![PromptTemplate {
            name: "greet".into(),
            description: Some("Greets a subject".into()),
            content: "Hello $1".into(),
            source: "test".into(),
        }],
        Vec::new(),
    );
    let harness = Harness::new(
        session("pipeline").await,
        harness_config,
        backend.clone(),
        Some(hooks.clone()),
    )
    .await
    .expect("harness");

    assert_eq!(
        harness
            .prompt("/handled", PromptOptions::interactive())
            .await
            .expect("handled"),
        PromptOutcome::Handled
    );
    assert!(matches!(
        harness
            .prompt("/alias", PromptOptions::interactive())
            .await
            .expect("prompt"),
        PromptOutcome::Completed(_)
    ));

    let requests = backend.requests().await;
    assert_eq!(requests.len(), 1);
    let Message::User(user) = requests[0]
        .snapshot
        .context
        .messages
        .last()
        .expect("user message")
    else {
        panic!("expected expanded user message");
    };
    assert_eq!(ri_harness::user_text(user), "Hello world");
    assert_eq!(
        hooks.stages.lock().await.as_slice(),
        [
            "command:/handled",
            "command:/alias",
            "input:/alias",
            "before:Hello world"
        ]
    );
}

#[tokio::test]
async fn one_at_a_time_queues_drain_steering_before_followups() {
    let model = model();
    let backend = Arc::new(ScriptBackend::blocking([
        Ok(output(&model, "one", false)),
        Ok(output(&model, "two", false)),
        Ok(output(&model, "three", false)),
        Ok(output(&model, "four", false)),
    ]));
    let harness = Harness::new(
        session("queues").await,
        config(&model),
        backend.clone(),
        None,
    )
    .await
    .expect("harness");
    let running = {
        let harness = harness.clone();
        tokio::spawn(async move { harness.prompt_text("initial").await })
    };
    backend.first_started.notified().await;
    harness
        .steer_with_images(
            "steer-one",
            vec![ImageContent {
                data: "aW1hZ2U=".to_owned(),
                mime_type: "image/png".to_owned(),
            }],
        )
        .await
        .expect("steer one");
    harness.steer("steer-two").await.expect("steer two");
    harness.follow_up("follow").await.expect("follow up");
    backend.first_release.notify_one();
    running.await.expect("join").expect("prompt");

    let requests = backend.requests().await;
    assert_eq!(requests.len(), 4);
    let latest_users = requests
        .iter()
        .map(|request| {
            request
                .snapshot
                .context
                .messages
                .iter()
                .rev()
                .find_map(|message| match message {
                    Message::User(message) => Some(ri_harness::user_text(message)),
                    Message::Assistant(_) | Message::ToolResult(_) => None,
                })
                .expect("latest user")
        })
        .collect::<Vec<_>>();
    assert_eq!(
        latest_users,
        ["initial", "steer-one", "steer-two", "follow"]
    );
    assert!(requests.iter().skip(1).any(|request| {
        request.snapshot.context.messages.iter().any(|message| {
            let Message::User(message) = message else {
                return false;
            };
            let UserContent::Blocks(blocks) = &message.content else {
                return false;
            };
            blocks.iter().any(|block| {
                matches!(
                    block,
                    InputContent::Image(image)
                        if image.data == "aW1hZ2U=" && image.mime_type == "image/png"
                )
            })
        })
    }));
}

#[tokio::test]
async fn live_auto_compaction_and_retry_switches_update_future_config() {
    let model = model();
    let harness = Harness::new(
        session("live-policy").await,
        config(&model),
        Arc::new(ScriptBackend::new([])),
        None,
    )
    .await
    .expect("harness");

    harness.set_auto_compaction_enabled(false).await;
    harness.set_auto_retry_enabled(false).await;
    let config = harness.config().await;
    assert!(!config.compaction.enabled);
    assert!(!config.retry.enabled);
}

#[tokio::test]
async fn navigation_summarizes_the_abandoned_branch_and_restores_user_input() {
    let model = model();
    let session = session("navigation").await;
    session
        .append_message(
            serde_json::to_value(Message::User(UserMessage::new("root"))).expect("serialize user"),
        )
        .await
        .expect("append root");
    session
        .append_message(
            serde_json::to_value(Message::Assistant(assistant(&model, "answer")))
                .expect("serialize assistant"),
        )
        .await
        .expect("append answer");
    let abandoned_user_id = session
        .append_message(
            serde_json::to_value(Message::User(UserMessage::new("abandoned request")))
                .expect("serialize user"),
        )
        .await
        .expect("append abandoned user");
    session
        .append_message(
            serde_json::to_value(Message::Assistant(assistant(&model, "abandoned answer")))
                .expect("serialize assistant"),
        )
        .await
        .expect("append abandoned answer");

    let backend = Arc::new(ScriptBackend::new(
        Vec::<Result<TurnOutput, BackendError>>::new(),
    ));
    backend
        .summaries
        .lock()
        .await
        .push_back(Err(BackendError::new(
            BackendErrorKind::Transient,
            "summary temporarily unavailable",
        )));
    backend
        .summaries
        .lock()
        .await
        .push_back(Ok(SummaryResponse {
            text: "Branch work was summarized.".into(),
            usage: Usage::default(),
        }));
    let harness = Harness::new(session, config(&model), backend.clone(), None)
        .await
        .expect("harness");
    let events = Arc::new(EventRecorder::default());
    harness.add_observer(events.clone()).await;
    let result = harness
        .navigate(
            abandoned_user_id,
            NavigateOptions {
                summarize: true,
                label: Some("return point".into()),
                ..NavigateOptions::default()
            },
        )
        .await
        .expect("navigate");

    assert!(!result.cancelled);
    assert_eq!(result.editor_text.as_deref(), Some("abandoned request"));
    assert!(result.summary_entry_id.is_some());
    let summaries = backend.summary_requests.lock().await;
    assert_eq!(summaries.len(), 2);
    assert!(summaries[0].prompt.contains("abandoned answer"));
    assert!(
        harness
            .session()
            .await
            .snapshot()
            .await
            .expect("snapshot")
            .active_path()
            .expect("active path")
            .iter()
            .any(|entry| matches!(entry.entry, SessionEntry::BranchSummary(_)))
    );
    let retry_events = events
        .events()
        .await
        .into_iter()
        .filter(|event| {
            matches!(
                event,
                HarnessEvent::RetryScheduled { .. }
                    | HarnessEvent::RetryAttemptStarted { .. }
                    | HarnessEvent::RetryFinished { .. }
            )
        })
        .collect::<Vec<_>>();
    assert!(matches!(
        retry_events.as_slice(),
        [
            HarnessEvent::RetryScheduled {
                operation: RetryOperation::BranchSummary,
                attempt: 1,
                max_attempts: 3,
                ..
            },
            HarnessEvent::RetryAttemptStarted {
                kind: SummaryKind::Branch,
                reason: None,
            },
            HarnessEvent::RetryFinished {
                operation: RetryOperation::BranchSummary,
                success: true,
                attempt: 1,
                final_error: None,
            }
        ]
    ));
}

#[tokio::test]
async fn manual_compaction_persists_a_summary_with_custom_focus() {
    let model = model();
    let session = session("manual-compaction").await;
    for index in 0..4 {
        session
            .append_message(
                serde_json::to_value(Message::User(UserMessage::new(format!(
                    "request {index} {}",
                    "x".repeat(80)
                ))))
                .expect("serialize user"),
            )
            .await
            .expect("append user");
        session
            .append_message(
                serde_json::to_value(Message::Assistant(assistant(
                    &model,
                    &format!("answer {index} {}", "y".repeat(80)),
                )))
                .expect("serialize assistant"),
            )
            .await
            .expect("append assistant");
    }
    let backend = Arc::new(ScriptBackend::new(
        Vec::<Result<TurnOutput, BackendError>>::new(),
    ));
    for text in ["history summary", "split turn summary"] {
        backend
            .summaries
            .lock()
            .await
            .push_back(Ok(SummaryResponse {
                text: text.into(),
                usage: Usage::default(),
            }));
    }
    let mut harness_config = config(&model);
    harness_config.compaction.keep_recent_tokens = 20;
    let harness = Harness::new(session, harness_config, backend.clone(), None)
        .await
        .expect("harness");
    let result = harness
        .compact(Some("Preserve API decisions.".into()))
        .await
        .expect("compact");

    assert!(!result.summary.is_empty());
    assert!(!result.first_kept_entry_id.is_empty());
    assert!(
        backend
            .summary_requests
            .lock()
            .await
            .iter()
            .any(|request| request.prompt.contains("Preserve API decisions."))
    );
    assert!(
        harness
            .session()
            .await
            .snapshot()
            .await
            .expect("snapshot")
            .active_path()
            .expect("active path")
            .iter()
            .any(|entry| matches!(entry.entry, SessionEntry::Compaction(_)))
    );
}

#[tokio::test]
async fn overflow_compacts_then_retries_from_projected_session_state() {
    let model = model();
    let session = session("overflow").await;
    for index in 0..6 {
        session
            .append_message(
                serde_json::to_value(Message::User(UserMessage::new(format!(
                    "long user message {index} {}",
                    "x".repeat(80)
                ))))
                .expect("json"),
            )
            .await
            .expect("user");
        session
            .append_message(
                serde_json::to_value(Message::Assistant(assistant(
                    &model,
                    &format!("long assistant {index} {}", "y".repeat(80)),
                )))
                .expect("json"),
            )
            .await
            .expect("assistant");
    }
    let backend = Arc::new(ScriptBackend::new([
        Err(BackendError::new(
            BackendErrorKind::ContextOverflow,
            "prompt is too long",
        )),
        Ok(output(&model, "after compaction", false)),
    ]));
    for text in ["historical summary", "turn prefix", "overflow summary"] {
        backend
            .summaries
            .lock()
            .await
            .push_back(Ok(SummaryResponse {
                text: text.to_owned(),
                usage: Usage::from_parts(20, 5, 0, 0),
            }));
    }
    let mut harness_config = config(&model);
    harness_config.compaction.enabled = false;
    harness_config.compaction.reserve_tokens = 20;
    harness_config.compaction.keep_recent_tokens = 20;
    let harness = Harness::new(session, harness_config, backend.clone(), None)
        .await
        .expect("harness");
    let outcome = harness
        .prompt("continue", PromptOptions::interactive())
        .await
        .expect("overflow recovery");
    assert!(matches!(outcome, PromptOutcome::Completed(_)));
    assert_eq!(backend.requests().await.len(), 2);
    assert!(
        harness
            .session()
            .await
            .branch()
            .await
            .expect("branch")
            .iter()
            .any(|entry| matches!(entry.entry, SessionEntry::Compaction(_)))
    );
}

#[derive(Debug, Default)]
struct BindingHooks {
    calls: Mutex<Vec<String>>,
}

#[async_trait]
impl ri_harness::HarnessHooks for BindingHooks {
    async fn unbind_session(&self, context: &HookContext) -> ri_harness::Result<()> {
        self.calls.lock().await.push(format!(
            "unbind:{}:{}",
            context.session_id, context.generation
        ));
        Ok(())
    }

    async fn bind_session(&self, context: &HookContext) -> ri_harness::Result<()> {
        self.calls.lock().await.push(format!(
            "bind:{}:{}",
            context.session_id, context.generation
        ));
        Ok(())
    }
}

#[tokio::test]
async fn replacement_unbinds_then_binds_and_invalidates_captured_contexts() {
    let model = model();
    let hooks = Arc::new(BindingHooks::default());
    let backend = Arc::new(ScriptBackend::new(std::iter::empty::<
        Result<TurnOutput, BackendError>,
    >()));
    let harness = Harness::new(
        session("old").await,
        config(&model),
        backend,
        Some(hooks.clone()),
    )
    .await
    .expect("harness");
    let stale = HookContext {
        session_id: "old".into(),
        generation: 1,
    };
    harness
        .replace_session(session("new").await, config(&model))
        .await
        .expect("replace");
    assert!(harness.validate_hook_context(&stale).await.is_err());
    assert_eq!(
        hooks.calls.lock().await.as_slice(),
        ["bind:old:1", "unbind:old:1", "bind:new:2"]
    );
}
