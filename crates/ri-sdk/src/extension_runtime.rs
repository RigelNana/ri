//! Adapters from `ri-ext` reducers to harness and agent hook boundaries.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use ri_agent::{
    AfterToolCallContext, AfterToolCallResult, AgentError, AgentEvent, BeforeToolCallContext,
    BeforeToolCallResult, Tool as AgentTool, ToolCallContext, ToolError, ToolResult,
};
use ri_ai::{
    ContentBlock, ImageContent, Message, TextContent, ToolResultMessage, UserContent, UserMessage,
    message::InputContent,
};
use ri_ext::{
    ActionError, ContentPart, ContextActions, ContextBinding, ContextMessage, CustomMessage,
    EventBus, Extension, ExtensionHost, GenerationClock, InputResult, InputSource, MessageRole,
    NativeTool, NotificationEvent, SessionBeforeEvent, SessionOverride,
    StreamingBehavior as ExtStreamingBehavior, ToolCallEvent, ToolResultEvent, ToolUsage,
};
use ri_harness::{
    AgentBackendHooks, BeforeAgentStart, BeforeAgentStartResult, BeforeCompactionResult,
    BeforeNavigation, BeforeNavigationResult, BranchSummaryOverride, CompactionOverride,
    CompactionPreparation, CompactionReason, Error as HarnessError, HarnessEvent, HarnessHooks,
    HookContext, InputAction, InputEvent, Phase, PromptOptions, PromptSource,
    Result as HarnessResult, SessionWrite, StreamingBehavior,
};
use serde_json::{Value, json};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

/// Shared native extension host adapted to both high- and low-level hooks.
pub struct ExtensionRuntime {
    host: RwLock<ExtensionHost>,
    extensions: Arc<[Arc<dyn Extension>]>,
    actions: Option<Arc<HarnessContextActions>>,
    pending_session_start: AtomicBool,
    turn_index: AtomicUsize,
}

impl std::fmt::Debug for ExtensionRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExtensionRuntime")
            .field("extension_count", &self.extensions.len())
            .finish_non_exhaustive()
    }
}

impl ExtensionRuntime {
    /// Wraps an already loaded host and its reload set.
    pub fn new(host: ExtensionHost, extensions: Vec<Arc<dyn Extension>>) -> Self {
        Self {
            host: RwLock::new(host),
            extensions: extensions.into(),
            actions: None,
            pending_session_start: AtomicBool::new(false),
            turn_index: AtomicUsize::new(0),
        }
    }

    /// Loads extensions with context actions that are bound to the harness
    /// created by [`ri_sdk::SessionBuilder`](crate::SessionBuilder).
    pub async fn load(binding: ContextBinding, extensions: Vec<Arc<dyn Extension>>) -> Arc<Self> {
        let actions = Arc::new(HarnessContextActions::default());
        let host = ExtensionHost::load(
            GenerationClock::default(),
            binding,
            actions.clone(),
            EventBus::default(),
            &extensions,
        )
        .await;
        Arc::new(Self {
            host: RwLock::new(host),
            extensions: extensions.into(),
            actions: Some(actions),
            pending_session_start: AtomicBool::new(false),
            turn_index: AtomicUsize::new(0),
        })
    }

    /// Runs a closure against the serialized extension host.
    pub async fn with_host<T>(&self, action: impl FnOnce(&ExtensionHost) -> T) -> T {
        let host = self.host.read().await;
        action(&host)
    }

    /// Produces `ri-agent` adapters for all extension-registered native tools.
    pub async fn agent_tools(self: &Arc<Self>) -> Vec<Arc<dyn AgentTool>> {
        let tools = {
            let host = self.host.read().await;
            let names = host
                .registries()
                .tool_names()
                .map(str::to_owned)
                .collect::<Vec<_>>();
            names
                .into_iter()
                .filter_map(|name| host.registries().tool(&name).map(|(tool, _)| (name, tool)))
                .collect::<Vec<_>>()
        };
        tools
            .into_iter()
            .map(|(name, tool)| {
                Arc::new(NativeToolAdapter::new(name, tool, self.clone())) as Arc<dyn AgentTool>
            })
            .collect()
    }

    pub(crate) async fn bind_harness(&self, harness: ri_harness::Harness) {
        if let Some(actions) = &self.actions {
            actions.bind(harness).await;
            if self.pending_session_start.swap(false, Ordering::AcqRel) {
                self.emit_session_start().await;
            }
        }
    }

    async fn emit_session_start(&self) {
        self.host
            .read()
            .await
            .runner()
            .emit_notification(&NotificationEvent::SessionStart {
                reason: "session bound".to_owned(),
            })
            .await;
    }
}

#[async_trait]
impl HarnessHooks for ExtensionRuntime {
    async fn command(&self, _context: &HookContext, input: &str) -> HarnessResult<bool> {
        let Some(command_text) = input.strip_prefix('/') else {
            return Ok(false);
        };
        let (name, arguments) = command_text
            .split_once(char::is_whitespace)
            .map_or((command_text, ""), |(name, arguments)| {
                (name, arguments.trim_start())
            });
        let (command, context) = {
            let host = self.host.read().await;
            let command = host
                .registries()
                .commands()
                .into_iter()
                .find(|command| command.invocation_name == name);
            (command, host.runner().context())
        };
        let Some(command) = command else {
            return Ok(false);
        };
        command
            .handler
            .execute(arguments, &context)
            .await
            .map_err(|error| HarnessError::Hook(error.to_string()))?;
        Ok(true)
    }

    async fn input(&self, _context: &HookContext, event: InputEvent) -> HarnessResult<InputAction> {
        let source = input_source(event.source);
        let behavior = event.streaming_behavior.map(|behavior| match behavior {
            ri_harness::StreamingBehavior::Steer => ExtStreamingBehavior::Steer,
            ri_harness::StreamingBehavior::FollowUp => ExtStreamingBehavior::FollowUp,
        });
        let reduced = self
            .host
            .read()
            .await
            .runner()
            .emit_input(ri_ext::InputEvent {
                text: event.text,
                images: event.images.iter().map(image_to_part).collect(),
                source,
                streaming_behavior: behavior,
            })
            .await;
        match reduced {
            InputResult::Continue => Ok(InputAction::Continue),
            InputResult::Handled => Ok(InputAction::Handled),
            InputResult::Transform { text, images } => Ok(InputAction::Transform {
                text,
                images: images
                    .map(|parts| {
                        parts
                            .into_iter()
                            .map(part_to_image)
                            .collect::<HarnessResult<Vec<_>>>()
                    })
                    .transpose()?,
            }),
        }
    }

    async fn before_agent_start(
        &self,
        _context: &HookContext,
        event: BeforeAgentStart,
    ) -> HarnessResult<BeforeAgentStartResult> {
        let reduced = self
            .host
            .read()
            .await
            .runner()
            .emit_before_agent_start(
                event.prompt,
                event.images.iter().map(image_to_part).collect(),
                event.system_prompt,
            )
            .await;
        let messages = reduced
            .messages
            .into_iter()
            .map(|message| {
                Message::User(UserMessage::new(format!(
                    "<extension-message type={:?}>\n{}\n</extension-message>",
                    message.custom_type, message.content
                )))
            })
            .collect();
        Ok(BeforeAgentStartResult {
            messages,
            system_prompt: reduced.system_prompt,
        })
    }

    async fn context(
        &self,
        _context: &HookContext,
        messages: Vec<Message>,
    ) -> HarnessResult<Vec<Message>> {
        let projected = messages.iter().map(to_context_message).collect::<Vec<_>>();
        self.host
            .read()
            .await
            .runner()
            .emit_context(&projected)
            .await
            .into_iter()
            .map(from_context_message)
            .collect()
    }

    async fn before_compaction(
        &self,
        _context: &HookContext,
        preparation: &CompactionPreparation,
        reason: CompactionReason,
        _will_retry: bool,
        _custom_instructions: Option<&str>,
        _cancellation: CancellationToken,
    ) -> HarnessResult<BeforeCompactionResult> {
        let value = json!({
            "firstKeptEntryId": preparation.first_kept_entry_id,
            "tokensBefore": preparation.tokens_before,
            "splitTurn": preparation.is_split_turn,
            "fileOperations": preparation.file_operations.lists(),
        });
        let result = self
            .host
            .read()
            .await
            .runner()
            .emit_session_before(&SessionBeforeEvent::Compact {
                reason,
                preparation: value,
            })
            .await;
        let Some(result) = result else {
            return Ok(BeforeCompactionResult::default());
        };
        let replacement = match result.override_value {
            Some(SessionOverride::Compaction(value)) => {
                Some(parse_compaction_override(value, preparation)?)
            }
            _ => None,
        };
        Ok(BeforeCompactionResult {
            cancel: result.cancel,
            replacement,
        })
    }

    async fn before_navigation(
        &self,
        _context: &HookContext,
        event: BeforeNavigation,
        _cancellation: CancellationToken,
    ) -> HarnessResult<BeforeNavigationResult> {
        let preparation = json!({
            "oldLeafId": event.old_leaf_id,
            "commonAncestorId": event.common_ancestor_id,
            "entryIds": event.entries.iter().map(|entry| entry.entry.id()).collect::<Vec<_>>(),
        });
        let result = self
            .host
            .read()
            .await
            .runner()
            .emit_session_before(&SessionBeforeEvent::Tree {
                target_id: event.target_id,
                preparation,
            })
            .await;
        let Some(result) = result else {
            return Ok(BeforeNavigationResult::default());
        };
        let mut output = BeforeNavigationResult {
            cancel: result.cancel,
            ..BeforeNavigationResult::default()
        };
        if let Some(SessionOverride::TreeSummary {
            summary,
            details,
            custom_instructions,
            replace_instructions,
            label,
        }) = result.override_value
        {
            output.summary = Some(BranchSummaryOverride {
                summary,
                details,
                usage: None,
            });
            output.custom_instructions = custom_instructions;
            output.replace_instructions = replace_instructions;
            output.label = label;
        }
        Ok(output)
    }

    async fn event(&self, _context: &HookContext, event: &HarnessEvent) -> HarnessResult<()> {
        let notification = match event {
            HarnessEvent::PromptAccepted { .. } => {
                self.turn_index.store(0, Ordering::Release);
                Some(NotificationEvent::AgentStart)
            }
            HarnessEvent::Settled { .. } => Some(NotificationEvent::AgentSettled),
            _ => None,
        };
        if let Some(notification) = notification {
            self.host
                .read()
                .await
                .runner()
                .emit_notification(&notification)
                .await;
        }
        Ok(())
    }

    async fn unbind_session(&self, _context: &HookContext) -> HarnessResult<()> {
        self.host
            .write()
            .await
            .reload("session replacement", &self.extensions)
            .await;
        Ok(())
    }

    async fn bind_session(&self, _context: &HookContext) -> HarnessResult<()> {
        if let Some(actions) = &self.actions
            && !actions.is_bound().await
        {
            self.pending_session_start.store(true, Ordering::Release);
            return Ok(());
        }
        self.emit_session_start().await;
        Ok(())
    }
}

#[async_trait]
impl AgentBackendHooks for ExtensionRuntime {
    async fn event(
        &self,
        event: &AgentEvent<Message>,
        _cancellation: CancellationToken,
    ) -> Result<(), AgentError> {
        let notification = match event {
            // High-level lifecycle and message completion own these notifications.
            AgentEvent::AgentStart | AgentEvent::MessageEnd { .. } => None,
            AgentEvent::AgentEnd { messages } => Some(NotificationEvent::AgentEnd {
                messages: messages.iter().map(to_context_message).collect(),
            }),
            AgentEvent::TurnStart => Some(NotificationEvent::TurnStart {
                index: self.turn_index.fetch_add(1, Ordering::AcqRel),
                timestamp_ms: u64::try_from(ri_ai::message::now_millis()).unwrap_or_default(),
            }),
            AgentEvent::TurnEnd { .. } => Some(NotificationEvent::TurnEnd {
                index: self.turn_index.load(Ordering::Acquire).saturating_sub(1),
            }),
            AgentEvent::MessageStart { message } => Some(NotificationEvent::MessageStart {
                message: to_context_message(message),
            }),
            AgentEvent::MessageUpdate {
                message,
                assistant_event,
            } => Some(NotificationEvent::MessageUpdate {
                message: to_context_message(message),
                delta: serde_json::to_value(assistant_event).unwrap_or(Value::Null),
            }),
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                ..
            } => Some(NotificationEvent::ToolExecutionStart {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
            }),
            AgentEvent::ToolExecutionUpdate {
                tool_call_id,
                tool_name,
                partial_result,
                ..
            } => Some(NotificationEvent::ToolExecutionUpdate {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                partial_result: serde_json::to_value(partial_result).unwrap_or(Value::Null),
            }),
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                is_error,
                ..
            } => Some(NotificationEvent::ToolExecutionEnd {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                is_error: *is_error,
            }),
        };
        if let Some(notification) = notification {
            self.host
                .read()
                .await
                .runner()
                .emit_notification(&notification)
                .await;
        }
        Ok(())
    }

    async fn before_tool_call(
        &self,
        context: BeforeToolCallContext<Message>,
        _cancellation: CancellationToken,
    ) -> Result<BeforeToolCallResult, AgentError> {
        let arguments =
            context.arguments.as_object().cloned().ok_or_else(|| {
                AgentError::Callback("tool arguments must be an object".to_owned())
            })?;
        let reduced = self
            .host
            .read()
            .await
            .runner()
            .emit_tool_call(ToolCallEvent {
                tool_call_id: context.tool_call.id,
                tool_name: context.tool_call.name,
                input: arguments,
            })
            .await;
        let result = reduced.result.unwrap_or_default();
        Ok(BeforeToolCallResult {
            block: result.block,
            reason: result.reason,
            arguments: Some(Value::Object(reduced.event.input)),
        })
    }

    async fn after_tool_call(
        &self,
        context: AfterToolCallContext<Message>,
        _cancellation: CancellationToken,
    ) -> Result<AfterToolCallResult, AgentError> {
        let arguments = context.arguments.as_object().cloned().unwrap_or_default();
        let event = ToolResultEvent {
            tool_call_id: context.tool_call.id,
            tool_name: context.tool_call.name,
            input: arguments,
            content: context.result.content.iter().map(input_to_part).collect(),
            details: context.result.details,
            is_error: context.is_error,
            usage: context.result.usage.as_ref().map(|usage| ToolUsage {
                input_tokens: usage.input,
                output_tokens: usage.output,
            }),
        };
        let Some(reduced) = self
            .host
            .read()
            .await
            .runner()
            .emit_tool_result(event)
            .await
        else {
            return Ok(AfterToolCallResult::default());
        };
        Ok(AfterToolCallResult {
            content: Some(
                reduced
                    .content
                    .into_iter()
                    .map(part_to_input)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            details: reduced.details,
            usage: reduced.usage.map(|usage| {
                ri_ai::Usage::from_parts(usage.input_tokens, usage.output_tokens, 0, 0)
            }),
            is_error: Some(reduced.is_error),
            terminate: None,
        })
    }

    async fn message_end(&self, message: &Message) -> Result<Option<Message>, AgentError> {
        self.host
            .read()
            .await
            .runner()
            .emit_message_end(to_context_message(message))
            .await
            .map(from_context_message)
            .transpose()
            .map_err(|error| AgentError::Callback(error.to_string()))
    }
}

#[derive(Debug, Default)]
struct HarnessContextActions {
    harness: RwLock<Option<ri_harness::Harness>>,
}

impl HarnessContextActions {
    async fn bind(&self, harness: ri_harness::Harness) {
        *self.harness.write().await = Some(harness);
    }

    async fn is_bound(&self) -> bool {
        self.harness.read().await.is_some()
    }

    async fn harness(&self) -> Result<ri_harness::Harness, ActionError> {
        self.harness
            .read()
            .await
            .clone()
            .ok_or_else(|| ActionError::new("extension runtime is not bound to a session"))
    }
}

#[async_trait]
impl ContextActions for HarnessContextActions {
    async fn send_message(&self, message: CustomMessage) -> Result<(), ActionError> {
        self.harness()
            .await?
            .write_session(SessionWrite::CustomMessage {
                kind: message.custom_type,
                content: Value::String(message.content),
                display: message.display,
                details: message.details,
            })
            .await
            .map_err(|error| action_error(&error))
    }

    async fn send_user_message(&self, text: String) -> Result<(), ActionError> {
        let harness = self.harness().await?;
        let phase = harness.status().await.phase;
        let options = PromptOptions {
            source: PromptSource::Extension,
            streaming_behavior: (phase != Phase::Idle).then_some(StreamingBehavior::Steer),
            expand_resources: false,
            ..PromptOptions::default()
        };
        harness
            .prompt(text, options)
            .await
            .map(|_| ())
            .map_err(|error| action_error(&error))
    }

    async fn append_entry(&self, custom_type: String, data: Value) -> Result<(), ActionError> {
        self.harness()
            .await?
            .write_session(SessionWrite::Custom {
                kind: custom_type,
                data: Some(data),
            })
            .await
            .map_err(|error| action_error(&error))
    }

    async fn active_tools(&self) -> Result<Vec<String>, ActionError> {
        Ok(self
            .harness()
            .await?
            .config()
            .await
            .active_tool_names
            .to_vec())
    }

    async fn set_active_tools(&self, names: Vec<String>) -> Result<(), ActionError> {
        self.harness()
            .await?
            .set_active_tools(names)
            .await
            .map_err(|error| action_error(&error))
    }

    async fn shutdown(&self) -> Result<(), ActionError> {
        self.harness()
            .await?
            .abort()
            .await
            .map(|_| ())
            .map_err(|error| action_error(&error))
    }
}

struct NativeToolAdapter {
    definition: ri_ai::Tool,
    label: String,
    tool: Arc<dyn NativeTool>,
    runtime: Arc<ExtensionRuntime>,
}

impl NativeToolAdapter {
    fn new(name: String, tool: Arc<dyn NativeTool>, runtime: Arc<ExtensionRuntime>) -> Self {
        let descriptor = tool.descriptor();
        Self {
            definition: ri_ai::Tool::new(
                name,
                descriptor.description.clone(),
                descriptor.parameter_schema.clone(),
            ),
            label: descriptor.label.clone(),
            tool,
            runtime,
        }
    }
}

impl std::fmt::Debug for NativeToolAdapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeToolAdapter")
            .field("definition", &self.definition)
            .field("label", &self.label)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl AgentTool for NativeToolAdapter {
    fn definition(&self) -> &ri_ai::Tool {
        &self.definition
    }

    fn label(&self) -> &str {
        &self.label
    }

    async fn execute(
        &self,
        context: ToolCallContext,
        arguments: Value,
    ) -> Result<ToolResult, ToolError> {
        let arguments = arguments.as_object().cloned().ok_or_else(|| {
            ToolError::Arguments("extension tool arguments must be an object".into())
        })?;
        let extension_context = self.runtime.host.read().await.runner().context();
        let output = self
            .tool
            .execute(&context.tool_call_id, arguments, &extension_context)
            .await
            .map_err(|error| ToolError::message(error.to_string()))?;
        if output.is_error {
            let message = output
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            return Err(ToolError::message(if message.is_empty() {
                "extension tool reported an error".to_owned()
            } else {
                message
            }));
        }
        Ok(ToolResult {
            content: output
                .content
                .into_iter()
                .map(|part| {
                    part_to_input(part).map_err(|error| ToolError::message(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?,
            details: output.details,
            usage: output.usage.map(|usage| {
                ri_ai::Usage::from_parts(usage.input_tokens, usage.output_tokens, 0, 0)
            }),
            ..ToolResult::default()
        })
    }
}

fn action_error(error: &ri_harness::Error) -> ActionError {
    ActionError::new(error.to_string())
}

const fn input_source(source: PromptSource) -> InputSource {
    match source {
        PromptSource::Rpc => InputSource::Rpc,
        PromptSource::Extension => InputSource::Extension,
        PromptSource::Interactive
        | PromptSource::Print
        | PromptSource::Json
        | PromptSource::Sdk => InputSource::Interactive,
    }
}

fn parse_compaction_override(
    value: Value,
    preparation: &CompactionPreparation,
) -> HarnessResult<CompactionOverride> {
    #[derive(serde::Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Override {
        summary: String,
        first_kept_entry_id: Option<String>,
        tokens_before: Option<u64>,
        details: Option<Value>,
        retained_tail: Option<Vec<Value>>,
    }
    let value: Override = serde_json::from_value(value)
        .map_err(|error| HarnessError::Hook(format!("invalid compaction override: {error}")))?;
    Ok(CompactionOverride {
        summary: value.summary,
        first_kept_entry_id: value
            .first_kept_entry_id
            .unwrap_or_else(|| preparation.first_kept_entry_id.clone()),
        tokens_before: value.tokens_before.unwrap_or(preparation.tokens_before),
        details: value.details,
        usage: None,
        retained_tail: value.retained_tail,
    })
}

fn image_to_part(image: &ImageContent) -> ContentPart {
    ContentPart::Image {
        media_type: image.mime_type.clone(),
        source: json!({"data": image.data}),
    }
}

fn part_to_image(part: ContentPart) -> HarnessResult<ImageContent> {
    match part {
        ContentPart::Image { media_type, source } => Ok(ImageContent {
            data: source
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| HarnessError::Hook("extension image source lacks data".to_owned()))?
                .to_owned(),
            mime_type: media_type,
        }),
        ContentPart::Text { .. } => Err(HarnessError::Hook(
            "input image replacement contained text".to_owned(),
        )),
    }
}

fn input_to_part(content: &InputContent) -> ContentPart {
    match content {
        InputContent::Text(text) => ContentPart::Text {
            text: text.text.clone(),
        },
        InputContent::Image(image) => image_to_part(image),
    }
}

fn part_to_input(part: ContentPart) -> Result<InputContent, AgentError> {
    match part {
        ContentPart::Text { text } => Ok(InputContent::Text(TextContent::new(text))),
        ContentPart::Image { media_type, source } => Ok(InputContent::Image(ImageContent {
            data: source
                .get("data")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    AgentError::Callback("extension image source lacks data".to_owned())
                })?
                .to_owned(),
            mime_type: media_type,
        })),
    }
}

fn to_context_message(message: &Message) -> ContextMessage {
    let (role, content) = match message {
        Message::User(message) => {
            let content = match &message.content {
                UserContent::Text(text) => vec![ContentPart::Text { text: text.clone() }],
                UserContent::Blocks(blocks) => blocks.iter().map(input_to_part).collect(),
            };
            (MessageRole::User, content)
        }
        Message::Assistant(message) => (
            MessageRole::Assistant,
            message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text(text) => Some(ContentPart::Text {
                        text: text.text.clone(),
                    }),
                    ContentBlock::Thinking(thinking) => Some(ContentPart::Text {
                        text: thinking.thinking.clone(),
                    }),
                    ContentBlock::ToolCall(_) => None,
                })
                .collect(),
        ),
        Message::ToolResult(message) => (
            MessageRole::ToolResult,
            message.content.iter().map(input_to_part).collect(),
        ),
    };
    ContextMessage {
        role,
        content,
        metadata: Some(json!({"ri.original": message})),
    }
}

fn from_context_message(message: ContextMessage) -> HarnessResult<Message> {
    let original = message
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("ri.original"))
        .cloned()
        .map(serde_json::from_value::<Message>)
        .transpose()
        .map_err(|error| HarnessError::Hook(error.to_string()))?;
    if let Some(original) = &original
        && to_context_message(original).content == message.content
    {
        return Ok(original.clone());
    }
    match message.role {
        MessageRole::User | MessageRole::Custom => {
            let blocks = message
                .content
                .into_iter()
                .map(|part| {
                    part_to_input(part).map_err(|error| HarnessError::Hook(error.to_string()))
                })
                .collect::<HarnessResult<Vec<_>>>()?;
            Ok(Message::User(UserMessage {
                content: UserContent::Blocks(blocks),
                timestamp: ri_ai::message::now_millis(),
            }))
        }
        MessageRole::Assistant => {
            let Some(Message::Assistant(mut original)) = original else {
                return Err(HarnessError::Hook(
                    "extensions cannot inject an assistant context message without metadata"
                        .to_owned(),
                ));
            };
            original.content = message
                .content
                .into_iter()
                .map(|part| match part {
                    ContentPart::Text { text } => Ok(ContentBlock::Text(TextContent::new(text))),
                    ContentPart::Image { .. } => Err(HarnessError::Hook(
                        "assistant context images are unsupported".to_owned(),
                    )),
                })
                .collect::<HarnessResult<Vec<_>>>()?;
            Ok(Message::Assistant(original))
        }
        MessageRole::ToolResult => {
            let Some(Message::ToolResult(original)) = original else {
                return Err(HarnessError::Hook(
                    "extensions cannot inject a tool result without metadata".to_owned(),
                ));
            };
            Ok(Message::ToolResult(ToolResultMessage {
                content: message
                    .content
                    .into_iter()
                    .map(|part| {
                        part_to_input(part).map_err(|error| HarnessError::Hook(error.to_string()))
                    })
                    .collect::<HarnessResult<Vec<_>>>()?,
                ..original
            }))
        }
    }
}
