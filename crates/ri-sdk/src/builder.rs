//! Ergonomic construction of the shared session runtime.

use std::sync::Arc;

use ri_ai::{Model, ThinkingLevel, clamp_thinking_level};
use ri_harness::{
    AgentBackend, AgentBackendHooks, CompactionSettings, Harness, HarnessConfig, HarnessHooks,
    QueueMode, RequestOptions, RetryPolicy,
};
use ri_session::{CreateOptions, Repository, Session};

use crate::error::{Error, Result};
use crate::{ExtensionRuntime, ModelRuntime, ResourceRuntime, SessionRuntime};

#[derive(Clone)]
enum SessionSource {
    Existing(Session),
    Create {
        repository: Arc<dyn Repository>,
        options: CreateOptions,
    },
    Open {
        repository: Arc<dyn Repository>,
        id: String,
    },
}

impl std::fmt::Debug for SessionSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Existing(_) => formatter.write_str("Existing(Session)"),
            Self::Create {
                repository,
                options,
            } => formatter
                .debug_struct("Create")
                .field("repository", repository)
                .field("options", options)
                .finish(),
            Self::Open { repository, id } => formatter
                .debug_struct("Open")
                .field("repository", repository)
                .field("id", id)
                .finish(),
        }
    }
}

#[derive(Clone, Debug)]
enum ModelSelection {
    Exact(Box<Model>),
    Catalog { provider: String, model: String },
}

/// Builder for one durable high-level runtime.
///
/// A model and session source are deliberately required: the SDK never creates
/// a fake provider or silently falls back to in-memory storage.
#[derive(Clone)]
pub struct SessionBuilder {
    models: Arc<ModelRuntime>,
    model: Option<ModelSelection>,
    session: Option<SessionSource>,
    resources: ResourceRuntime,
    system_prompt: Option<String>,
    thinking_level: ThinkingLevel,
    request_options: RequestOptions,
    steering_mode: QueueMode,
    follow_up_mode: QueueMode,
    retry: RetryPolicy,
    compaction: CompactionSettings,
    hooks: Option<Arc<dyn HarnessHooks>>,
    agent_hooks: Option<Arc<dyn AgentBackendHooks>>,
    extensions: Option<Arc<ExtensionRuntime>>,
}

impl std::fmt::Debug for SessionBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionBuilder")
            .field("models", &self.models)
            .field("model", &self.model)
            .field("session", &self.session)
            .field("resources", &self.resources)
            .field("system_prompt", &self.system_prompt)
            .field("thinking_level", &self.thinking_level)
            .field("request_options", &self.request_options)
            .field("steering_mode", &self.steering_mode)
            .field("follow_up_mode", &self.follow_up_mode)
            .field("retry", &self.retry)
            .field("compaction", &self.compaction)
            .field("has_hooks", &self.hooks.is_some())
            .field("has_agent_hooks", &self.agent_hooks.is_some())
            .field("has_extensions", &self.extensions.is_some())
            .finish()
    }
}

impl SessionBuilder {
    /// Starts a builder from an explicit provider/auth runtime.
    pub fn new(models: Arc<ModelRuntime>) -> Self {
        Self {
            models,
            model: None,
            session: None,
            resources: ResourceRuntime::default(),
            system_prompt: None,
            thinking_level: ThinkingLevel::Off,
            request_options: RequestOptions::default(),
            steering_mode: QueueMode::OneAtATime,
            follow_up_mode: QueueMode::OneAtATime,
            retry: RetryPolicy::default(),
            compaction: CompactionSettings::default(),
            hooks: None,
            agent_hooks: None,
            extensions: None,
        }
    }

    /// Uses exact model metadata, including custom/dynamic endpoint fields.
    #[must_use]
    pub fn model(mut self, model: Model) -> Self {
        self.model = Some(ModelSelection::Exact(Box::new(model)));
        self
    }

    /// Resolves a model from the shared catalog during `build`.
    #[must_use]
    pub fn catalog_model(mut self, provider: impl Into<String>, model: impl Into<String>) -> Self {
        self.model = Some(ModelSelection::Catalog {
            provider: provider.into(),
            model: model.into(),
        });
        self
    }

    /// Attaches an already-open session.
    #[must_use]
    pub fn session(mut self, session: Session) -> Self {
        self.session = Some(SessionSource::Existing(session));
        self
    }

    /// Creates a session in the supplied real repository.
    #[must_use]
    pub fn create_session(
        mut self,
        repository: Arc<dyn Repository>,
        options: CreateOptions,
    ) -> Self {
        self.session = Some(SessionSource::Create {
            repository,
            options,
        });
        self
    }

    /// Opens a durable session by repository id.
    #[must_use]
    pub fn open_session(mut self, repository: Arc<dyn Repository>, id: impl Into<String>) -> Self {
        self.session = Some(SessionSource::Open {
            repository,
            id: id.into(),
        });
        self
    }

    /// Installs a resolved resource and executable-tool snapshot.
    #[must_use]
    pub fn resources(mut self, resources: ResourceRuntime) -> Self {
        self.resources = resources;
        self
    }

    /// Sets the base model-visible system instruction.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets requested reasoning effort (clamped to model support at build).
    #[must_use]
    pub fn thinking_level(mut self, level: ThinkingLevel) -> Self {
        self.thinking_level = level;
        self
    }

    /// Sets provider request policy.
    #[must_use]
    pub fn request_options(mut self, options: RequestOptions) -> Self {
        self.request_options = options;
        self
    }

    /// Sets the steering safe-point drain policy.
    #[must_use]
    pub fn steering_mode(mut self, mode: QueueMode) -> Self {
        self.steering_mode = mode;
        self
    }

    /// Sets the follow-up safe-point drain policy.
    #[must_use]
    pub fn follow_up_mode(mut self, mode: QueueMode) -> Self {
        self.follow_up_mode = mode;
        self
    }

    /// Sets high-level transient retry policy.
    #[must_use]
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    /// Sets automatic and manual compaction policy.
    #[must_use]
    pub fn compaction(mut self, settings: CompactionSettings) -> Self {
        self.compaction = settings;
        self
    }

    /// Installs prompt/session lifecycle hooks.
    #[must_use]
    pub fn hooks(mut self, hooks: Arc<dyn HarnessHooks>) -> Self {
        self.hooks = Some(hooks);
        self
    }

    /// Installs low-level agent/tool hooks.
    #[must_use]
    pub fn agent_hooks(mut self, hooks: Arc<dyn AgentBackendHooks>) -> Self {
        self.agent_hooks = Some(hooks);
        self
    }

    /// Installs one native extension runtime at both lifecycle boundaries.
    #[must_use]
    pub fn extensions(mut self, extensions: Arc<ExtensionRuntime>) -> Self {
        self.hooks = Some(extensions.clone());
        self.agent_hooks = Some(extensions.clone());
        self.extensions = Some(extensions);
        self
    }

    /// Resolves dependencies and creates one shared runtime instance.
    ///
    /// # Errors
    /// Returns an error when required inputs are missing or model, session, tool, or harness setup fails.
    pub async fn build(self) -> Result<SessionRuntime> {
        let model = match self.model.ok_or(Error::Missing("model"))? {
            ModelSelection::Exact(model) => *model,
            ModelSelection::Catalog { provider, model } => self
                .models
                .model(&provider, &model)
                .ok_or(Error::ModelNotFound { provider, model })?,
        };
        let session = match self.session.ok_or(Error::Missing("session source"))? {
            SessionSource::Existing(session) => session,
            SessionSource::Create {
                repository,
                options,
            } => repository.create(options).await?,
            SessionSource::Open { repository, id } => repository.open(&id).await?,
        };
        let mut tools = self.resources.tools().to_vec();
        let mut active_tool_names = self.resources.active_tool_names().to_vec();
        if let Some(extensions) = &self.extensions {
            let extension_tools = extensions.agent_tools().await;
            active_tool_names.extend(
                extension_tools
                    .iter()
                    .map(|tool| tool.definition().name.clone()),
            );
            tools.extend(extension_tools);
        }
        let mut backend = AgentBackend::new(self.models.clone(), self.models.clone(), tools)?;
        if let Some(hooks) = self.agent_hooks {
            backend = backend.with_hooks(hooks);
        }
        let tool_definitions = backend.tool_definitions();
        let config = HarnessConfig {
            model: Arc::new(model.clone()),
            thinking_level: clamp_thinking_level(&model, self.thinking_level),
            system_prompt: self
                .resources
                .resolve_system_prompt(self.system_prompt.as_deref()),
            tools: tool_definitions.into(),
            active_tool_names: active_tool_names.into(),
            resources: self.resources.resources().as_ref().clone(),
            request_options: self.request_options,
            steering_mode: self.steering_mode,
            follow_up_mode: self.follow_up_mode,
            retry: self.retry,
            compaction: self.compaction,
        };
        let harness = Harness::new(session, config, Arc::new(backend), self.hooks).await?;
        if let Some(extensions) = self.extensions {
            extensions.bind_harness(harness.clone()).await;
        }
        Ok(SessionRuntime::new(harness, self.models, self.resources))
    }
}
