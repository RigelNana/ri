//! Strongly typed construction for a reusable agent base.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use ri_agent::{AgentEvent as LoopEvent, Tool};
use ri_ai::{
    ApiKeyCredential, AssistantMessage, Credential, InMemoryCredentialStore, Message, Models,
    SystemAuthContext, ThinkingLevel, builtin_providers,
};
use ri_harness::{
    AgentBackendHooks, HarnessEvent, HarnessObserver, PromptTemplate, Resources, Skill,
};
use ri_session::{CreateOptions, JsonlRepository};
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use url::Url;

use crate::{Error, ModelRuntime, ResourceRuntime, Result, SessionBuilder, SessionRuntime};

/// A provider included in Ri's built-in text-model catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinProvider {
    /// Amazon Bedrock.
    AmazonBedrock,
    /// Ant Ling.
    AntLing,
    /// Anthropic Messages API.
    Anthropic,
    /// Azure `OpenAI` Responses API.
    AzureOpenAiResponses,
    /// Cerebras.
    Cerebras,
    /// Cloudflare AI Gateway.
    CloudflareAiGateway,
    /// Cloudflare Workers AI.
    CloudflareWorkersAi,
    /// `DeepSeek`.
    DeepSeek,
    /// Fireworks.
    Fireworks,
    /// GitHub Copilot.
    GitHubCopilot,
    /// Google Generative AI API.
    Google,
    /// Google Vertex AI.
    GoogleVertex,
    /// Groq.
    Groq,
    /// Hugging Face inference routing.
    HuggingFace,
    /// Kimi For Coding.
    KimiCoding,
    /// `MiniMax` international endpoint.
    MiniMax,
    /// `MiniMax` China endpoint.
    MiniMaxCn,
    /// Mistral Conversations API.
    Mistral,
    /// Moonshot AI international endpoint.
    MoonshotAi,
    /// Moonshot AI China endpoint.
    MoonshotAiCn,
    /// NVIDIA NIM.
    Nvidia,
    /// `OpenAI` APIs.
    OpenAi,
    /// `OpenAI` Codex Responses API.
    OpenAiCodex,
    /// `OpenCode` Zen.
    OpenCode,
    /// `OpenCode` Zen Go.
    OpenCodeGo,
    /// `OpenRouter`.
    OpenRouter,
    /// Qwen Token Plan international endpoint.
    QwenTokenPlan,
    /// Qwen Token Plan China endpoint.
    QwenTokenPlanCn,
    /// Together AI.
    Together,
    /// Vercel AI Gateway.
    VercelAiGateway,
    /// xAI.
    Xai,
    /// Xiaomi `MiMo`.
    Xiaomi,
    /// Xiaomi Token Plan Amsterdam endpoint.
    XiaomiTokenPlanAms,
    /// Xiaomi Token Plan China endpoint.
    XiaomiTokenPlanCn,
    /// Xiaomi Token Plan Singapore endpoint.
    XiaomiTokenPlanSgp,
    /// Z.AI.
    Zai,
    /// Z.AI Coding China endpoint.
    ZaiCodingCn,
}

impl BuiltinProvider {
    /// Every provider in Pi's generated static model catalog.
    pub const ALL: [Self; 37] = [
        Self::AmazonBedrock,
        Self::AntLing,
        Self::Anthropic,
        Self::AzureOpenAiResponses,
        Self::Cerebras,
        Self::CloudflareAiGateway,
        Self::CloudflareWorkersAi,
        Self::DeepSeek,
        Self::Fireworks,
        Self::GitHubCopilot,
        Self::Google,
        Self::GoogleVertex,
        Self::Groq,
        Self::HuggingFace,
        Self::KimiCoding,
        Self::MiniMax,
        Self::MiniMaxCn,
        Self::Mistral,
        Self::MoonshotAi,
        Self::MoonshotAiCn,
        Self::Nvidia,
        Self::OpenAi,
        Self::OpenAiCodex,
        Self::OpenCode,
        Self::OpenCodeGo,
        Self::OpenRouter,
        Self::QwenTokenPlan,
        Self::QwenTokenPlanCn,
        Self::Together,
        Self::VercelAiGateway,
        Self::Xai,
        Self::Xiaomi,
        Self::XiaomiTokenPlanAms,
        Self::XiaomiTokenPlanCn,
        Self::XiaomiTokenPlanSgp,
        Self::Zai,
        Self::ZaiCodingCn,
    ];

    /// Returns the provider id used by the model catalog and session format.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AmazonBedrock => "amazon-bedrock",
            Self::AntLing => "ant-ling",
            Self::Anthropic => "anthropic",
            Self::AzureOpenAiResponses => "azure-openai-responses",
            Self::Cerebras => "cerebras",
            Self::CloudflareAiGateway => "cloudflare-ai-gateway",
            Self::CloudflareWorkersAi => "cloudflare-workers-ai",
            Self::DeepSeek => "deepseek",
            Self::Fireworks => "fireworks",
            Self::GitHubCopilot => "github-copilot",
            Self::Google => "google",
            Self::GoogleVertex => "google-vertex",
            Self::Groq => "groq",
            Self::HuggingFace => "huggingface",
            Self::KimiCoding => "kimi-coding",
            Self::MiniMax => "minimax",
            Self::MiniMaxCn => "minimax-cn",
            Self::Mistral => "mistral",
            Self::MoonshotAi => "moonshotai",
            Self::MoonshotAiCn => "moonshotai-cn",
            Self::Nvidia => "nvidia",
            Self::OpenAi => "openai",
            Self::OpenAiCodex => "openai-codex",
            Self::OpenCode => "opencode",
            Self::OpenCodeGo => "opencode-go",
            Self::OpenRouter => "openrouter",
            Self::QwenTokenPlan => "qwen-token-plan",
            Self::QwenTokenPlanCn => "qwen-token-plan-cn",
            Self::Together => "together",
            Self::VercelAiGateway => "vercel-ai-gateway",
            Self::Xai => "xai",
            Self::Xiaomi => "xiaomi",
            Self::XiaomiTokenPlanAms => "xiaomi-token-plan-ams",
            Self::XiaomiTokenPlanCn => "xiaomi-token-plan-cn",
            Self::XiaomiTokenPlanSgp => "xiaomi-token-plan-sgp",
            Self::Zai => "zai",
            Self::ZaiCodingCn => "zai-coding-cn",
        }
    }
}

impl fmt::Display for BuiltinProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A non-empty API key whose debug representation never exposes the secret.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(Box<str>);

impl ApiKey {
    /// Creates an API key while enforcing its non-empty invariant.
    ///
    /// # Errors
    /// Returns [`InvalidApiKey`] when `value` is empty.
    pub fn new(value: impl Into<String>) -> std::result::Result<Self, InvalidApiKey> {
        let value = value.into();
        if value.is_empty() {
            return Err(InvalidApiKey);
        }
        Ok(Self(value.into_boxed_str()))
    }

    fn into_string(self) -> String {
        self.0.into()
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApiKey([REDACTED])")
    }
}

impl TryFrom<String> for ApiKey {
    type Error = InvalidApiKey;

    fn try_from(value: String) -> std::result::Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<&str> for ApiKey {
    type Error = InvalidApiKey;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        Self::new(value)
    }
}

/// Error returned when an API key is empty.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[error("API key must not be empty")]
pub struct InvalidApiKey;

/// One event from either the durable harness or the low-level streaming loop.
#[derive(Clone, Debug)]
pub enum AgentEvent {
    /// Session lifecycle, retry, compaction, queue, and persistence event.
    Runtime(HarnessEvent),
    /// Model delta, message, turn, and tool-execution event.
    Loop(Arc<LoopEvent<Message>>),
}

/// Receiver for one agent's ordered event stream.
#[derive(Debug)]
pub struct AgentEvents {
    receiver: broadcast::Receiver<AgentEvent>,
}

impl AgentEvents {
    /// Receives the next event.
    ///
    /// # Errors
    /// Returns a lag error when the consumer falls behind the bounded event
    /// buffer, or a closed error after every agent sender is dropped.
    pub async fn recv(&mut self) -> std::result::Result<AgentEvent, broadcast::error::RecvError> {
        self.receiver.recv().await
    }
}

#[derive(Debug)]
struct EventHub {
    sender: broadcast::Sender<AgentEvent>,
}

impl EventHub {
    fn new() -> Self {
        let (sender, _) = broadcast::channel(1_024);
        Self { sender }
    }

    fn subscribe(&self) -> AgentEvents {
        AgentEvents {
            receiver: self.sender.subscribe(),
        }
    }

    fn send(&self, event: AgentEvent) {
        // Having no subscribers is a normal state, not an event-delivery failure.
        drop(self.sender.send(event));
    }
}

#[async_trait]
impl AgentBackendHooks for EventHub {
    async fn event(
        &self,
        event: &LoopEvent<Message>,
        _cancellation: CancellationToken,
    ) -> std::result::Result<(), ri_agent::AgentError> {
        self.send(AgentEvent::Loop(Arc::new(event.clone())));
        Ok(())
    }
}

#[async_trait]
impl HarnessObserver for EventHub {
    async fn on_event(&self, event: &HarnessEvent) -> ri_harness::Result<()> {
        self.send(AgentEvent::Runtime(event.clone()));
        Ok(())
    }
}

/// Builder for an agent backed by one built-in provider and model.
///
/// Provider, model, and authentication are constructor arguments because an
/// agent cannot be valid without them. The endpoint remains an explicit,
/// strongly typed override. The working directory defaults to the process
/// working directory, matching Pi, and sessions default to `.ri/sessions`
/// beneath that directory.
#[derive(Clone, Debug)]
pub struct AgentBuilder {
    provider: BuiltinProvider,
    model: String,
    api_key: ApiKey,
    endpoint: Option<Url>,
    workspace: Option<PathBuf>,
    session_dir: Option<PathBuf>,
    system_prompt: Option<String>,
    thinking_level: ThinkingLevel,
    tools: Vec<Arc<dyn Tool>>,
    coding_tools: bool,
    skills: Vec<Skill>,
    prompt_templates: Vec<PromptTemplate>,
    context: Vec<String>,
}

impl AgentBuilder {
    fn new(provider: BuiltinProvider, model: impl Into<String>, api_key: ApiKey) -> Self {
        Self {
            provider,
            model: model.into(),
            api_key,
            endpoint: None,
            workspace: None,
            session_dir: None,
            system_prompt: None,
            thinking_level: ThinkingLevel::Off,
            tools: Vec::new(),
            coding_tools: false,
            skills: Vec::new(),
            prompt_templates: Vec::new(),
            context: Vec::new(),
        }
    }

    /// Overrides the selected provider model's API base URL.
    #[must_use]
    pub fn endpoint(mut self, endpoint: Url) -> Self {
        self.endpoint = Some(endpoint);
        self
    }

    /// Sets the project working directory used by tools and the session.
    #[must_use]
    pub fn workspace(mut self, workspace: impl Into<PathBuf>) -> Self {
        self.workspace = Some(workspace.into());
        self
    }

    /// Stores new JSONL sessions in `directory`.
    #[must_use]
    pub fn session_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.session_dir = Some(directory.into());
        self
    }

    /// Sets the model-visible system instruction.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the requested reasoning level, clamped to model support at build time.
    #[must_use]
    pub fn thinking_level(mut self, level: ThinkingLevel) -> Self {
        self.thinking_level = level;
        self
    }

    /// Registers one application tool. Agents start with no tools.
    #[must_use]
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Registers application tools in iteration order.
    #[must_use]
    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Registers one concrete tool without requiring the caller to allocate it.
    #[must_use]
    pub fn owned_tool(mut self, tool: impl Tool) -> Self {
        self.tools.push(Arc::new(tool));
        self
    }

    /// Adds Ri's seven workspace coding tools after application tools.
    #[must_use]
    pub const fn coding_tools(mut self) -> Self {
        self.coding_tools = true;
        self
    }

    /// Registers one model-visible skill and its direct `/skill:name` invocation.
    #[must_use]
    pub fn skill(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Registers one slash-invoked prompt template.
    #[must_use]
    pub fn prompt_template(mut self, template: PromptTemplate) -> Self {
        self.prompt_templates.push(template);
        self
    }

    /// Adds a model-visible context fragment.
    #[must_use]
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    /// Builds a durable agent from the explicitly registered capabilities.
    ///
    /// # Errors
    /// Returns an error when the model is unknown, authentication cannot be
    /// installed, paths are invalid, or the session runtime cannot be created.
    pub async fn build(self) -> Result<Agent> {
        let workspace = match self.workspace {
            Some(workspace) => workspace,
            None => std::env::current_dir()?,
        };
        let cwd = utf8_path(&workspace)?.to_owned();
        let session_dir = self
            .session_dir
            .unwrap_or_else(|| workspace.join(".ri").join("sessions"));

        let credentials = Arc::new(InMemoryCredentialStore::default());
        credentials
            .set(
                self.provider.as_str(),
                Credential::ApiKey(ApiKeyCredential {
                    key: Some(self.api_key.into_string()),
                    env: BTreeMap::default(),
                }),
            )
            .await?;
        let models = Arc::new(ModelRuntime::new(Models::with_providers(
            credentials,
            Arc::new(SystemAuthContext),
            builtin_providers(),
        )));
        let mut model = models
            .model(self.provider.as_str(), &self.model)
            .ok_or_else(|| Error::ModelNotFound {
                provider: self.provider.to_string(),
                model: self.model,
            })?;
        if let Some(endpoint) = self.endpoint {
            model.base_url = endpoint.into();
        }

        let mut tools = self.tools;
        if self.coding_tools {
            tools.extend(crate::local_tools(workspace)?);
        }
        let resources = ResourceRuntime::new(
            Resources::new(self.skills, self.prompt_templates, self.context),
            tools,
        );
        let repository = Arc::new(JsonlRepository::new(session_dir));
        let events = Arc::new(EventHub::new());
        let mut builder = SessionBuilder::new(models)
            .model(model)
            .create_session(repository, CreateOptions::new(cwd))
            .resources(resources)
            .thinking_level(self.thinking_level)
            .agent_hooks(events.clone());
        if let Some(prompt) = self.system_prompt {
            builder = builder.system_prompt(prompt);
        }
        let runtime = builder.build().await?;
        runtime.add_observer(events.clone()).await;
        Ok(Agent { runtime, events })
    }
}

/// A ready-to-use, multi-turn agent with durable session state.
#[derive(Clone, Debug)]
pub struct Agent {
    runtime: SessionRuntime,
    events: Arc<EventHub>,
}

impl Agent {
    /// Starts a builder with every logically required choice.
    pub fn builder(
        provider: BuiltinProvider,
        model: impl Into<String>,
        api_key: ApiKey,
    ) -> AgentBuilder {
        AgentBuilder::new(provider, model, api_key)
    }

    /// Runs one prompt and returns its terminal assistant message.
    ///
    /// # Errors
    /// Returns an error when the agent is busy or model, tool, or persistence work fails.
    pub async fn prompt(&self, text: impl Into<String>) -> Result<AssistantMessage> {
        Ok(self.runtime.harness().prompt_message(text).await?)
    }

    /// Subscribes to model deltas, tool execution, and durable lifecycle events.
    ///
    /// Subscribe before calling [`prompt`](Self::prompt) so initial events are
    /// not missed.
    pub fn subscribe(&self) -> AgentEvents {
        self.events.subscribe()
    }

    /// Accesses the durable session runtime for advanced operations.
    pub const fn session(&self) -> &SessionRuntime {
        &self.runtime
    }

    /// Consumes the agent and returns its durable session runtime.
    pub fn into_session(self) -> SessionRuntime {
        self.runtime
    }
}

fn utf8_path(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| Error::NonUtf8Path(path.to_path_buf()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolve_model_auth;

    #[test]
    fn api_key_debug_is_redacted() {
        let key = ApiKey::new("top-secret").expect("valid key");
        assert_eq!(format!("{key:?}"), "ApiKey([REDACTED])");
        assert_eq!(ApiKey::new(""), Err(InvalidApiKey));
    }

    #[test]
    fn provider_enum_matches_generated_catalog() {
        let catalog_provider_ids = ri_ai::builtin_models()
            .into_iter()
            .map(|model| model.provider)
            .collect::<std::collections::BTreeSet<_>>();
        let typed_provider_ids = BuiltinProvider::ALL
            .map(BuiltinProvider::as_str)
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(catalog_provider_ids.len(), 37);
        assert_eq!(
            catalog_provider_ids,
            typed_provider_ids.into_iter().map(str::to_owned).collect()
        );
    }

    #[tokio::test]
    async fn builder_installs_key_and_endpoint() {
        let directory = tempfile::tempdir().expect("temporary workspace");
        let endpoint = Url::parse("https://llm.example.test/api").expect("valid URL");
        let agent = Agent::builder(
            BuiltinProvider::Anthropic,
            "claude-sonnet-4-6",
            ApiKey::new("test-key").expect("valid key"),
        )
        .endpoint(endpoint.clone())
        .workspace(directory.path())
        .build()
        .await
        .expect("agent builds");

        let config = agent.session().harness().config().await;
        assert_eq!(config.model.base_url, endpoint.as_str());
        assert!(config.tools.is_empty());
        let auth = resolve_model_auth(agent.session().models(), &config.model)
            .await
            .expect("auth resolves")
            .expect("auth is configured");
        assert_eq!(auth.auth.api_key.as_deref(), Some("test-key"));
    }

    #[tokio::test]
    async fn builder_injects_tools_skills_prompts_and_context() {
        let directory = tempfile::tempdir().expect("temporary workspace");
        let tool = ri_agent::FnTool::new(
            "ping",
            "Ping",
            "Returns pong.",
            serde_json::json!({"type": "object", "additionalProperties": false}),
            |_, _| async { Ok(ri_agent::ToolResult::text("pong")) },
        );
        let agent = Agent::builder(
            BuiltinProvider::Anthropic,
            "claude-sonnet-4-6",
            ApiKey::new("test-key").expect("valid key"),
        )
        .workspace(directory.path())
        .owned_tool(tool)
        .skill(Skill::new(
            "audit",
            "Audit code",
            "Inspect evidence.",
            "inline:audit",
        ))
        .prompt_template(PromptTemplate::new("review", "Review $1", "inline:review"))
        .context("Project context")
        .build()
        .await
        .expect("agent builds");

        let config = agent.session().harness().config().await;
        assert_eq!(config.tools.len(), 1);
        assert_eq!(config.tools[0].name, "ping");
        assert_eq!(config.resources.skills[0].name, "audit");
        assert_eq!(config.resources.prompt_templates[0].name, "review");
        assert_eq!(config.resources.context.as_ref(), ["Project context"]);
        assert_eq!(
            ri_harness::expand_resources("/review src/lib.rs", &config.resources).text,
            "Review src/lib.rs"
        );
        assert!(config.system_prompt.contains("<name>audit</name>"));
    }

    #[tokio::test]
    async fn event_subscriptions_receive_typed_events() {
        let hub = EventHub::new();
        let mut events = hub.subscribe();
        hub.send(AgentEvent::Runtime(HarnessEvent::PromptAccepted {
            operation: 7,
        }));
        assert!(matches!(
            events.recv().await.expect("event"),
            AgentEvent::Runtime(HarnessEvent::PromptAccepted { operation: 7 })
        ));
    }
}
