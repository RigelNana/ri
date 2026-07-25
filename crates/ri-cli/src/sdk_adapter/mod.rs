//! Localized construction and adaptation of the high-level SDK runtime.

mod events;
#[cfg(feature = "rpc")]
mod rpc;
mod sessions;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(feature = "rpc")]
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::Duration;

use async_trait::async_trait;
use ri_ai::{
    ApiKeyCredential, ContentBlock, Credential, CredentialStore, Model, Models, SystemAuthContext,
    ThinkingLevel,
};
use ri_sdk::{
    ModelRuntime, PromptOptions, PromptOutcome, ResourceRuntime, SessionBuilder, SessionRuntime,
    StreamingBehavior,
};
use ri_session::{CreateOptions, Session, SessionMetadata};
use serde_json::{Value, json};
use tokio::sync::{Mutex, broadcast};
#[cfg(feature = "rpc")]
use tokio_util::sync::CancellationToken;

use crate::cli::{
    Cli, Command, LoginArgs, ModelCommand, ModelListArgs, PackageCommand, PackageListArgs,
    ProviderCommand, ResourceCommand, ResourceKind, ResourceListArgs, ResourceMutationArgs,
    ResourceScope, SessionCommand, SessionExportArgs, SessionListArgs, SessionOpenArgs,
    ThinkingOption,
};
use crate::credential_store::{FileCredentialStore, OverlayCredentialStore};
use crate::error::{CliError, Result};
use crate::mode::IoCapabilities;
use crate::output::Output;
use crate::package_runtime::PackageRuntime;
use crate::runtime::{
    CliRuntime, CommandOutput, PromptCompletion, PromptDelivery, PromptRequest, RuntimeStatus,
};

use self::events::EventObserver;
use self::sessions::SessionRepository;

/// Concrete frontend adapter over exactly one optional SDK session runtime.
///
/// Metadata-only invocations intentionally avoid constructing a session or
/// requiring a configured model, but still share the production model,
/// credential, resource, and repository services.
#[derive(Debug)]
pub struct SdkCliRuntime {
    runtime: Option<SessionRuntime>,
    models: Arc<ModelRuntime>,
    resources: ResourceRuntime,
    packages: PackageRuntime,
    sessions: SessionRepository,
    events: broadcast::Sender<Value>,
    harness_events: broadcast::Sender<ri_sdk::HarnessEvent>,
    observer_id: Mutex<Option<u64>>,
    cwd: PathBuf,
    #[cfg(feature = "rpc")]
    bash_processes: Mutex<BTreeMap<String, CancellationToken>>,
    #[cfg(feature = "rpc")]
    bash_sequence: AtomicU64,
    #[cfg(feature = "rpc")]
    rpc_events_started: AtomicBool,
}

/// Build application services and, for agent modes, one shared session runtime.
///
/// # Errors
///
/// Returns a configuration, resource, credential, model, session, or SDK
/// construction error.
pub async fn build(cli: &Cli, io: IoCapabilities, output: &Output) -> Result<Arc<SdkCliRuntime>> {
    let cwd = std::env::current_dir().map_err(|source| CliError::Io {
        operation: "resolve working directory",
        source,
    })?;
    let agent_dir = agent_dir()?;
    let project_trust_override = cli
        .project_trust_override()
        .or_else(|| explicitly_mutates_project_settings(cli).then_some(true));
    let packages = PackageRuntime::new(
        cwd.clone(),
        agent_dir.clone(),
        cli.offline,
        project_trust_override,
    )
    .await?;
    let project_trusted = packages.project_trusted();
    let session_root = cli
        .session_dir
        .clone()
        .or(packages.configured_session_dir().await)
        .unwrap_or_else(|| agent_dir.join("sessions"));
    tokio::fs::create_dir_all(&session_root)
        .await
        .map_err(|source| CliError::Io {
            operation: "create session directory",
            source,
        })?;

    let credentials: Arc<dyn CredentialStore> =
        Arc::new(FileCredentialStore::new(agent_dir.join("auth.json")));
    let (configured_provider, _) = packages.configured_model().await;
    let credentials = credential_overlay(cli, credentials, configured_provider.as_deref())?;
    let catalog = Models::with_providers(
        credentials,
        Arc::new(SystemAuthContext),
        ri_ai::builtin_providers(),
    );
    if !cli.offline && needs_model_refresh(cli) {
        let refresh = catalog
            .refresh(ri_ai::RefreshOptions {
                allow_network: true,
                ..ri_ai::RefreshOptions::default()
            })
            .await;
        if cli.verbose {
            for (provider, error) in refresh.errors {
                output
                    .stderr_line(&format!(
                        "warning: model refresh for {provider} failed: {error}"
                    ))
                    .await?;
            }
        }
    }
    let models = Arc::new(ModelRuntime::new(catalog));
    let resources = if needs_resources(cli) {
        let package_resources = packages.resolve(false).await?.resources;
        let disabled_resources = packages.disabled_resources().await?;
        let configured_resources = packages.configured_resource_paths().await;
        load_resources(
            cli,
            &cwd,
            &agent_dir,
            project_trusted,
            &package_resources,
            &configured_resources,
            &disabled_resources,
        )
        .await?
    } else {
        ResourceRuntime::default()
    };
    let sessions = if cli.no_session {
        SessionRepository::memory()
    } else {
        SessionRepository::durable(session_root)
    };

    let (events, _) = broadcast::channel(1024);
    let (harness_events, _) = broadcast::channel(1024);
    if cli.is_metadata_request() {
        return Ok(Arc::new(SdkCliRuntime {
            runtime: None,
            models,
            resources,
            packages,
            sessions,
            events,
            harness_events,
            observer_id: Mutex::new(None),
            cwd,
            #[cfg(feature = "rpc")]
            bash_processes: Mutex::new(BTreeMap::new()),
            #[cfg(feature = "rpc")]
            bash_sequence: AtomicU64::new(1),
            #[cfg(feature = "rpc")]
            rpc_events_started: AtomicBool::new(false),
        }));
    }

    let model = select_model(cli, &models, &packages).await?;
    if cli.set_default {
        packages
            .set_default_model(model.provider.as_str(), model.id.as_str())
            .await?;
    }
    let session = select_session(cli, io, output, &sessions, &cwd).await?;
    if let Some(name) = &cli.name {
        session
            .append_session_info(Some(name.clone()))
            .await
            .map_err(|error| CliError::runtime("set session name", error))?;
    }
    let thinking = if let Some(thinking) = cli.thinking {
        thinking_level(thinking)
    } else if let Some(thinking) = cli
        .model
        .as_deref()
        .or_else(|| cli.models.first().map(String::as_str))
        .and_then(|selector| split_model_selector(selector).1)
    {
        thinking
    } else if let Some(thinking) = packages.configured_thinking_level().await {
        thinking
    } else {
        ThinkingLevel::Off
    };
    let settings = packages.configured_runtime().await;
    let runtime = SessionBuilder::new(Arc::clone(&models))
        .model(model)
        .session(session)
        .resources(resources.clone())
        .thinking_level(thinking)
        .steering_mode(settings.steering_mode)
        .follow_up_mode(settings.follow_up_mode)
        .compaction(ri_harness::CompactionSettings {
            enabled: settings.compaction.enabled,
            reserve_tokens: settings.compaction.reserve_tokens,
            keep_recent_tokens: settings.compaction.keep_recent_tokens,
        })
        .retry(ri_harness::RetryPolicy {
            enabled: settings.retry.enabled,
            max_retries: settings.retry.max_retries,
            base_delay: Duration::from_millis(settings.retry.base_delay_ms),
            max_delay: Duration::from_millis(settings.retry.provider_max_retry_delay_ms),
        })
        .request_options(ri_harness::RequestOptions {
            timeout: settings
                .retry
                .provider_timeout_ms
                .map(Duration::from_millis),
            transport_retries: settings.retry.provider_max_retries,
            max_transport_retry_delay: Some(Duration::from_millis(
                settings.retry.provider_max_retry_delay_ms,
            )),
            ..ri_harness::RequestOptions::default()
        })
        .build()
        .await
        .map_err(|error| CliError::runtime("build session runtime", error))?;
    let observer_id = runtime
        .add_observer(Arc::new(EventObserver::new(
            events.clone(),
            harness_events.clone(),
        )))
        .await;

    Ok(Arc::new(SdkCliRuntime {
        runtime: Some(runtime),
        models,
        resources,
        packages,
        sessions,
        events,
        harness_events,
        observer_id: Mutex::new(Some(observer_id)),
        cwd,
        #[cfg(feature = "rpc")]
        bash_processes: Mutex::new(BTreeMap::new()),
        #[cfg(feature = "rpc")]
        bash_sequence: AtomicU64::new(1),
        #[cfg(feature = "rpc")]
        rpc_events_started: AtomicBool::new(false),
    }))
}

#[async_trait]
impl CliRuntime for SdkCliRuntime {
    fn subscribe(&self) -> broadcast::Receiver<Value> {
        self.events.subscribe()
    }

    async fn session_header(&self) -> Result<Option<Value>> {
        let Some(runtime) = &self.runtime else {
            return Ok(None);
        };
        let header = runtime
            .session()
            .await
            .header()
            .await
            .map_err(|error| CliError::runtime("read session header", error))?;
        serde_json::to_value(header)
            .map(Some)
            .map_err(|source| CliError::Json {
                operation: "encoding the session header",
                source,
            })
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptCompletion> {
        let runtime = self.required_runtime("submit a prompt")?;
        let frontend = runtime.frontend(request.source);
        let outcome = frontend
            .prompt(
                request.text,
                PromptOptions {
                    images: request
                        .images
                        .into_iter()
                        .map(|image| ri_ai::ImageContent {
                            data: image.data,
                            mime_type: image.mime_type,
                        })
                        .collect(),
                    streaming_behavior: request.delivery.map(|delivery| match delivery {
                        PromptDelivery::Steer => StreamingBehavior::Steer,
                        PromptDelivery::FollowUp => StreamingBehavior::FollowUp,
                    }),
                    expand_resources: true,
                    ..PromptOptions::default()
                },
            )
            .await
            .map_err(|error| CliError::runtime("run prompt", error))?;
        match outcome {
            PromptOutcome::Completed(message) => {
                let text = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::Text(text) => Some(text.text.as_str()),
                        ContentBlock::Thinking(_) | ContentBlock::ToolCall(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let message = serde_json::to_value(message).map_err(|source| CliError::Json {
                    operation: "encoding the assistant message",
                    source,
                })?;
                Ok(PromptCompletion {
                    text,
                    message: Some(message),
                })
            }
            PromptOutcome::Handled | PromptOutcome::Queued(_) => Ok(PromptCompletion::default()),
        }
    }

    async fn abort(&self) -> Result<()> {
        self.required_runtime("abort the active run")?
            .abort()
            .await
            .map(|_| ())
            .map_err(|error| CliError::runtime("abort active run", error))
    }

    async fn status(&self) -> Result<RuntimeStatus> {
        let runtime = self.required_runtime("read runtime status")?;
        let session_id = runtime
            .session()
            .await
            .metadata()
            .await
            .map_err(|error| CliError::runtime("read session metadata", error))?
            .id;
        let config = runtime.harness().config().await;
        Ok(RuntimeStatus {
            session_id,
            model: Some(format!("{}/{}", config.model.provider, config.model.id)),
            thinking: config.thinking_level.as_str().to_owned(),
        })
    }

    async fn command(&self, command: &Command) -> Result<CommandOutput> {
        match command {
            Command::Provider { command } => self.provider_command(command).await,
            Command::Model { command } => self.model_command(command).await,
            Command::Session { command } => self.session_command(command).await,
            Command::Resource { command } => self.resource_command(command).await,
            Command::Package { command } => self.package_command(command).await,
            Command::Login(arguments) => self.login(arguments).await,
            Command::Logout(arguments) => self.logout(&arguments.provider).await,
            Command::Install(arguments) => {
                self.package_install(
                    &arguments.source,
                    arguments.local,
                    arguments.checksum.as_deref(),
                )
                .await
            }
            Command::Remove(arguments) | Command::Uninstall(arguments) => {
                self.package_remove(&arguments.source, arguments.local)
                    .await
            }
            Command::Update(arguments) => {
                self.package_update(arguments.source.as_deref(), arguments.all, arguments.force)
                    .await
            }
            Command::List(arguments) => self.package_list(arguments).await,
            Command::Config(arguments) => self.resource_list(arguments).await,
        }
    }

    #[cfg(feature = "rpc")]
    async fn rpc(
        &self,
        request: ri_rpc::Request,
        context: ri_rpc::DispatchContext,
    ) -> Result<ri_rpc::ResponsePayload> {
        rpc::dispatch(self, request, context).await
    }

    async fn shutdown(&self) -> Result<()> {
        if let Some(runtime) = &self.runtime {
            runtime.wait_settled().await;
            if let Some(observer_id) = self.observer_id.lock().await.take() {
                runtime.remove_observer(observer_id).await;
            }
        }
        Ok(())
    }
}

impl SdkCliRuntime {
    fn required_runtime(&self, operation: &'static str) -> Result<&SessionRuntime> {
        self.runtime
            .as_ref()
            .ok_or_else(|| CliError::unsupported(operation, "no session runtime was constructed"))
    }

    async fn provider_command(&self, command: &ProviderCommand) -> Result<CommandOutput> {
        match command {
            ProviderCommand::List => {
                let mut lines = Vec::new();
                for provider in self.models.models().providers() {
                    let auth = self
                        .models
                        .models()
                        .check_auth(provider.id())
                        .await
                        .map_err(|error| {
                            CliError::runtime("check provider authentication", error)
                        })?;
                    let status = auth.map_or("unavailable", |auth| {
                        if auth.configured {
                            "authenticated"
                        } else {
                            "not authenticated"
                        }
                    });
                    lines.push(format!("{}\t{}\t{status}", provider.id(), provider.name()));
                }
                Ok(CommandOutput::Text(lines.join("\n")))
            }
            ProviderCommand::Login(arguments) => self.login(arguments).await,
            ProviderCommand::Logout(arguments) => self.logout(&arguments.provider).await,
        }
    }

    async fn model_command(&self, command: &ModelCommand) -> Result<CommandOutput> {
        match command {
            ModelCommand::List(arguments) => self.model_list(arguments).await,
        }
    }

    async fn model_list(&self, arguments: &ModelListArgs) -> Result<CommandOutput> {
        let mut models = if arguments.all {
            self.models.models().models(arguments.provider.as_deref())
        } else {
            self.models
                .available(arguments.provider.as_deref())
                .await
                .map_err(|error| CliError::runtime("list available models", error))?
        };
        if let Some(search) = arguments.search.as_ref().map(|value| value.to_lowercase()) {
            models.retain(|model| {
                format!("{}/{} {}", model.provider, model.id, model.name)
                    .to_lowercase()
                    .contains(&search)
            });
        }
        models.sort_by(|left, right| (&left.provider, &left.id).cmp(&(&right.provider, &right.id)));
        if arguments.json {
            return serde_json::to_value(models)
                .map(CommandOutput::Json)
                .map_err(|source| CliError::Json {
                    operation: "encoding model list",
                    source,
                });
        }
        let lines = models
            .into_iter()
            .map(|model| {
                format!(
                    "{}/{}\t{}\t{} tokens{}",
                    model.provider,
                    model.id,
                    model.name,
                    model.context_window,
                    if model.reasoning { "\treasoning" } else { "" }
                )
            })
            .collect::<Vec<_>>();
        Ok(CommandOutput::Text(lines.join("\n")))
    }

    async fn login(&self, arguments: &LoginArgs) -> Result<CommandOutput> {
        let provider = self
            .models
            .models()
            .provider(&arguments.provider)
            .ok_or_else(|| CliError::NotFound {
                kind: "provider",
                name: arguments.provider.clone(),
            })?;
        let key = if let Some(key) = &arguments.api_key {
            key.trim().to_owned()
        } else if arguments.api_key_stdin {
            crate::input::read_piped_stdin()
                .await?
                .map_or_else(String::new, |key| key.trim().to_owned())
        } else {
            return Err(CliError::unsupported(
                "interactive provider login",
                if provider.auth().oauth.is_some() {
                    "the SDK exposes OAuth refresh but no authorization-flow initiator; pass --api-key when the provider accepts one"
                } else {
                    "pass --api-key or --api-key-stdin"
                },
            ));
        };
        if key.is_empty() {
            return Err(CliError::InvalidArguments(
                "provider API key cannot be empty".to_owned(),
            ));
        }
        if provider.auth().api_key.is_none() {
            return Err(CliError::unsupported(
                "API-key login",
                format!(
                    "provider `{}` does not expose API-key authentication",
                    provider.id()
                ),
            ));
        }
        self.models
            .models()
            .store_credential(
                provider.id(),
                Credential::ApiKey(ApiKeyCredential {
                    key: Some(key),
                    env: BTreeMap::new(),
                }),
            )
            .await
            .map_err(|error| CliError::runtime("store provider credential", error))?;
        Ok(CommandOutput::Text(format!(
            "stored credentials for {}",
            provider.id()
        )))
    }

    async fn logout(&self, provider: &str) -> Result<CommandOutput> {
        if self.models.models().provider(provider).is_none() {
            return Err(CliError::NotFound {
                kind: "provider",
                name: provider.to_owned(),
            });
        }
        self.models
            .models()
            .logout(provider)
            .await
            .map_err(|error| CliError::runtime("remove provider credential", error))?;
        Ok(CommandOutput::Text(format!(
            "removed stored credentials for {provider}"
        )))
    }

    async fn session_command(&self, command: &SessionCommand) -> Result<CommandOutput> {
        match command {
            SessionCommand::List(arguments) => self.session_list(arguments).await,
            SessionCommand::Open(arguments) => self.session_open(arguments).await,
            SessionCommand::Fork(arguments) => {
                let session = self.sessions.fork(arguments).await?;
                let metadata = session
                    .metadata()
                    .await
                    .map_err(|error| CliError::runtime("read fork metadata", error))?;
                Ok(CommandOutput::Text(metadata.id))
            }
            SessionCommand::Export(arguments) => self.session_export(arguments).await,
            SessionCommand::Import(arguments) => {
                let session = self.sessions.import(arguments).await?;
                let metadata = session
                    .metadata()
                    .await
                    .map_err(|error| CliError::runtime("read import metadata", error))?;
                Ok(CommandOutput::Text(metadata.id))
            }
        }
    }

    async fn session_list(&self, arguments: &SessionListArgs) -> Result<CommandOutput> {
        let cwd = if arguments.all {
            None
        } else {
            Some(
                arguments
                    .cwd
                    .as_deref()
                    .unwrap_or(&self.cwd)
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        let mut sessions = self.sessions.list(cwd).await?;
        if let Some(limit) = arguments.limit {
            sessions.truncate(limit);
        }
        if arguments.json {
            return serde_json::to_value(sessions)
                .map(CommandOutput::Json)
                .map_err(|source| CliError::Json {
                    operation: "encoding session list",
                    source,
                });
        }
        let lines = sessions
            .into_iter()
            .map(|session| {
                format!(
                    "{}\t{}\t{}",
                    session.id,
                    session.created_at.to_rfc3339(),
                    session.cwd
                )
            })
            .collect::<Vec<_>>();
        Ok(CommandOutput::Text(lines.join("\n")))
    }

    async fn session_open(&self, arguments: &SessionOpenArgs) -> Result<CommandOutput> {
        let session = self.sessions.open_target(&arguments.target).await?;
        let header = session
            .header()
            .await
            .map_err(|error| CliError::runtime("read session header", error))?;
        let entries = session
            .entries(None, None)
            .await
            .map_err(|error| CliError::runtime("read session entries", error))?;
        let leaf_id = session
            .leaf_id()
            .await
            .map_err(|error| CliError::runtime("read session leaf", error))?;
        let name = session
            .name()
            .await
            .map_err(|error| CliError::runtime("read session name", error))?;
        if arguments.json || arguments.tree {
            return Ok(CommandOutput::Json(json!({
                "header": header,
                "name": name,
                "leafId": leaf_id,
                "entries": entries,
            })));
        }
        Ok(CommandOutput::Text(format!(
            "{}\ncreated: {}\ncwd: {}\nentries: {}\nleaf: {}{}",
            header.id,
            header.timestamp.to_rfc3339(),
            header.cwd,
            entries.len(),
            leaf_id.as_deref().unwrap_or("-"),
            name.map_or_else(String::new, |name| format!("\nname: {name}")),
        )))
    }

    async fn session_export(&self, arguments: &SessionExportArgs) -> Result<CommandOutput> {
        let session = self.sessions.open_target(&arguments.source).await?;
        let content = sessions::export(&session, arguments.format).await?;
        if let Some(path) = &arguments.output {
            tokio::fs::write(path, content)
                .await
                .map_err(|source| CliError::Io {
                    operation: "write session export",
                    source,
                })?;
            Ok(CommandOutput::Silent)
        } else {
            Ok(CommandOutput::Text(content))
        }
    }

    async fn resource_command(&self, command: &ResourceCommand) -> Result<CommandOutput> {
        match command {
            ResourceCommand::List(arguments) => self.resource_list(arguments).await,
            ResourceCommand::Enable(arguments) => self.resource_mutation(arguments, true).await,
            ResourceCommand::Disable(arguments) => self.resource_mutation(arguments, false).await,
            ResourceCommand::Reload => Ok(CommandOutput::Text("resources reloaded".to_owned())),
        }
    }

    async fn resource_mutation(
        &self,
        arguments: &ResourceMutationArgs,
        enabled: bool,
    ) -> Result<CommandOutput> {
        self.packages
            .set_resource_enabled(arguments.kind, &arguments.name, arguments.local, enabled)
            .await?;
        Ok(CommandOutput::Text(format!(
            "{} {} `{}`{}",
            if enabled { "enabled" } else { "disabled" },
            resource_kind_label(arguments.kind),
            arguments.name,
            if arguments.local {
                " for this project"
            } else {
                ""
            }
        )))
    }

    async fn resource_list(&self, arguments: &ResourceListArgs) -> Result<CommandOutput> {
        if arguments.scope != ResourceScope::All {
            let local = arguments.scope == ResourceScope::Project;
            let scope = if local {
                "project settings"
            } else {
                "global settings"
            };
            let records = self
                .packages
                .resource_overrides(local)
                .await?
                .into_iter()
                .filter(|(kind, _, _)| {
                    arguments
                        .kind
                        .is_none_or(|requested| resource_kind_label(requested) == kind)
                })
                .map(|(kind, name, enabled)| {
                    json!({"kind": kind, "name": name, "enabled": enabled, "source": scope})
                })
                .collect();
            return Ok(resource_records_output(records, arguments.json));
        }
        let resources = self.resources.resources();
        let mut records = Vec::new();
        if arguments
            .kind
            .is_none_or(|kind| kind == ResourceKind::Skill)
        {
            records.extend(
                resources.skills.iter().map(
                    |skill| json!({"kind": "skill", "name": skill.name, "source": skill.source, "enabled": true}),
                ),
            );
        }
        if arguments
            .kind
            .is_none_or(|kind| kind == ResourceKind::Prompt)
        {
            records.extend(resources.prompt_templates.iter().map(
                |prompt| json!({"kind": "prompt", "name": prompt.name, "source": prompt.source, "enabled": true}),
            ));
        }
        if arguments
            .kind
            .is_none_or(|kind| kind == ResourceKind::Context)
        {
            records.extend(
                resources.context.iter().enumerate().map(
                    |(index, _)| json!({"kind": "context", "name": format!("context-{index}"), "enabled": true}),
                ),
            );
        }
        if arguments.kind.is_none_or(|kind| kind == ResourceKind::Tool) {
            records.extend(self.resources.tools().iter().map(
                |tool| json!({"kind": "tool", "name": tool.definition().name, "enabled": true}),
            ));
        }
        let disabled = self.packages.disabled_resources().await?;
        for (kind, names) in disabled {
            if arguments
                .kind
                .is_some_and(|requested| resource_kind_label(requested) != kind)
            {
                continue;
            }
            records.extend(names.into_iter().map(
                |name| json!({"kind": kind.clone(), "name": name, "enabled": false, "source": "settings"}),
            ));
        }
        Ok(resource_records_output(records, arguments.json))
    }

    async fn package_command(&self, command: &PackageCommand) -> Result<CommandOutput> {
        match command {
            PackageCommand::Install(arguments) => {
                self.package_install(
                    &arguments.source,
                    arguments.local,
                    arguments.checksum.as_deref(),
                )
                .await
            }
            PackageCommand::Remove(arguments) | PackageCommand::Uninstall(arguments) => {
                self.package_remove(&arguments.source, arguments.local)
                    .await
            }
            PackageCommand::Update(arguments) => {
                self.package_update(arguments.source.as_deref(), arguments.all, arguments.force)
                    .await
            }
            PackageCommand::List(arguments) => self.package_list(arguments).await,
        }
    }

    async fn package_install(
        &self,
        source: &str,
        local: bool,
        checksum: Option<&str>,
    ) -> Result<CommandOutput> {
        let identity = self.packages.install(source, local, checksum).await?;
        Ok(CommandOutput::Text(format!("installed {identity}")))
    }

    async fn package_remove(&self, source: &str, local: bool) -> Result<CommandOutput> {
        let identity = self.packages.remove(source, local).await?;
        Ok(CommandOutput::Text(format!("removed {identity}")))
    }

    async fn package_update(
        &self,
        source: Option<&str>,
        all: bool,
        force: bool,
    ) -> Result<CommandOutput> {
        let count = self.packages.update(source, all, force).await?;
        Ok(CommandOutput::Text(format!("updated {count} package(s)")))
    }

    async fn package_list(&self, arguments: &PackageListArgs) -> Result<CommandOutput> {
        let records = self.packages.records().await?;
        if arguments.json {
            return Ok(CommandOutput::Json(Value::Array(records)));
        }
        Ok(CommandOutput::Text(
            records
                .into_iter()
                .map(|record| {
                    let name = record.get("name").and_then(Value::as_str).unwrap_or("-");
                    let version = record.get("version").and_then(Value::as_str).unwrap_or("-");
                    let root = record.get("root").and_then(Value::as_str).unwrap_or("-");
                    format!("{name}\t{version}\t{root}")
                })
                .collect::<Vec<_>>()
                .join("\n"),
        ))
    }
}

fn credential_overlay(
    cli: &Cli,
    base: Arc<dyn CredentialStore>,
    configured_provider: Option<&str>,
) -> Result<Arc<dyn CredentialStore>> {
    let mut overrides = BTreeMap::new();
    if let Some(api_key) = &cli.api_key {
        let api_key = api_key.trim();
        if api_key.is_empty() {
            return Err(CliError::InvalidArguments(
                "--api-key cannot be empty".to_owned(),
            ));
        }
        let provider = explicit_provider(cli)
            .or_else(|| configured_provider.map(str::to_owned))
            .ok_or_else(|| {
                CliError::InvalidArguments(
                    "--api-key requires --provider or a provider/model selector".to_owned(),
                )
            })?;
        overrides.insert(
            provider,
            Credential::ApiKey(ApiKeyCredential {
                key: Some(api_key.to_owned()),
                env: BTreeMap::new(),
            }),
        );
    }
    if overrides.is_empty() {
        Ok(base)
    } else {
        Ok(Arc::new(OverlayCredentialStore::new(base, overrides)))
    }
}

async fn select_model(
    cli: &Cli,
    models: &ModelRuntime,
    packages: &PackageRuntime,
) -> Result<Model> {
    let (default_provider, default_model) = packages.configured_model().await;
    let provider = cli.provider.as_deref().or(default_provider.as_deref());
    let selector = cli
        .model
        .as_deref()
        .or_else(|| cli.models.first().map(String::as_str))
        .or(default_model.as_deref());
    if let Some(selector) = selector {
        let selector = split_model_selector(selector).0;
        if let Some((provider, model)) = selector.split_once('/') {
            return models
                .model(provider, model)
                .ok_or_else(|| CliError::NotFound {
                    kind: "model",
                    name: selector.to_owned(),
                });
        }
        if let Some(provider) = provider {
            return models
                .model(provider, selector)
                .ok_or_else(|| CliError::NotFound {
                    kind: "model",
                    name: format!("{provider}/{selector}"),
                });
        }
        let matches = models
            .models()
            .models(None)
            .into_iter()
            .filter(|model| model.id == selector)
            .collect::<Vec<_>>();
        return match matches.as_slice() {
            [model] => Ok(model.clone()),
            [] => Err(CliError::NotFound {
                kind: "model",
                name: selector.to_owned(),
            }),
            _ => Err(CliError::InvalidArguments(format!(
                "model id `{selector}` exists in multiple providers; use provider/model"
            ))),
        };
    }
    if let Some(provider) = provider {
        return models
            .available(Some(provider))
            .await
            .map_err(|error| CliError::runtime("resolve provider models", error))?
            .into_iter()
            .next()
            .ok_or_else(|| CliError::NotFound {
                kind: "authenticated model for provider",
                name: provider.to_owned(),
            });
    }
    models
        .available(None)
        .await
        .map_err(|error| CliError::runtime("resolve available models", error))?
        .into_iter()
        .next()
        .ok_or_else(|| {
            CliError::unsupported(
                "start an agent session",
                "no authenticated model is available; configure credentials and select --model",
            )
        })
}

async fn select_session(
    cli: &Cli,
    io: IoCapabilities,
    output: &Output,
    sessions: &SessionRepository,
    cwd: &Path,
) -> Result<Session> {
    let cwd_text = cwd.to_string_lossy().into_owned();
    if cli.no_session {
        return sessions.create(CreateOptions::new(cwd_text)).await;
    }
    if cli.continue_session {
        if let Some(session) = sessions.list(Some(cwd_text.clone())).await?.first() {
            return sessions.open_target(&session.id).await;
        }
        return sessions.create(CreateOptions::new(cwd_text)).await;
    }
    if cli.resume {
        if !io.stdin_tty || !io.stdout_tty {
            return Err(CliError::InvalidArguments(
                "--resume requires terminal stdin and stdout".to_owned(),
            ));
        }
        let available = sessions.list(None).await?;
        let selected = choose_session(&available, output).await?;
        return sessions.open_target(&selected).await;
    }
    if let Some(target) = &cli.session {
        return sessions.open_target(target).await;
    }
    if let Some(id) = &cli.session_id {
        match sessions.open_target(id).await {
            Ok(session) => return Ok(session),
            Err(CliError::NotFound { .. }) => {
                let mut options = CreateOptions::new(cwd_text);
                options.id = Some(id.clone());
                return sessions.create(options).await;
            }
            Err(error) => return Err(error),
        }
    }
    if let Some(source) = &cli.fork {
        return sessions
            .fork(&crate::cli::SessionForkArgs {
                source: source.clone(),
                entry: None,
                at: false,
                id: None,
                cwd: Some(cwd.to_owned()),
            })
            .await;
    }
    sessions.create(CreateOptions::new(cwd_text)).await
}

async fn choose_session(sessions: &[SessionMetadata], output: &Output) -> Result<String> {
    if sessions.is_empty() {
        return Err(CliError::NotFound {
            kind: "session",
            name: "any saved session".to_owned(),
        });
    }
    if sessions.len() == 1 {
        return Ok(sessions[0].id.clone());
    }
    output.stderr_line("Select a session:").await?;
    for (index, session) in sessions.iter().take(50).enumerate() {
        output
            .stderr_line(&format!(
                "{:>2}. {}  {}  {}",
                index + 1,
                session.id,
                session.created_at.to_rfc3339(),
                session.cwd
            ))
            .await?;
    }
    output.stderr_line("Enter number or id prefix:").await?;
    let answer = tokio::task::spawn_blocking(|| {
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer).map(|_| answer)
    })
    .await
    .map_err(|error| CliError::InvalidArguments(format!("session selector failed: {error}")))?
    .map_err(|source| CliError::Io {
        operation: "read session selection",
        source,
    })?;
    let answer = answer.trim();
    if let Ok(index) = answer.parse::<usize>()
        && let Some(session) = index.checked_sub(1).and_then(|index| sessions.get(index))
    {
        return Ok(session.id.clone());
    }
    let matches = sessions
        .iter()
        .filter(|session| session.id.starts_with(answer))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [session] => Ok(session.id.clone()),
        [] => Err(CliError::NotFound {
            kind: "session",
            name: answer.to_owned(),
        }),
        _ => Err(CliError::InvalidArguments(format!(
            "session prefix `{answer}` is ambiguous"
        ))),
    }
}

async fn load_resources(
    cli: &Cli,
    cwd: &Path,
    agent_dir: &Path,
    project_trusted: bool,
    package_resources: &[ri_ext::PackageResource],
    configured_resources: &crate::package_runtime::ConfiguredResourcePaths,
    disabled_resources: &BTreeMap<String, BTreeSet<String>>,
) -> Result<ResourceRuntime> {
    if !cli.extensions.is_empty() {
        return Err(CliError::unsupported(
            "load native extensions from paths",
            "the SDK accepts registered extension hooks but does not expose a dynamic native loader",
        ));
    }
    if !cli.themes.is_empty() {
        return Err(CliError::unsupported(
            "load terminal themes from paths",
            "the current ri-tui API does not expose theme-file decoding",
        ));
    }
    if !cli.no_extensions && !configured_resources.extensions.is_empty() {
        return Err(CliError::unsupported(
            "load configured native extensions",
            "the SDK accepts registered extension hooks but does not expose a dynamic native loader",
        ));
    }
    if !cli.no_themes && !configured_resources.themes.is_empty() {
        return Err(CliError::unsupported(
            "load configured terminal themes",
            "the current ri-tui API does not expose theme-file decoding",
        ));
    }
    let mut tools = if cli.no_tools || cli.no_builtin_tools {
        Vec::new()
    } else {
        ri_sdk::local_tools(cwd.to_owned())
            .map_err(|error| CliError::runtime("construct built-in tools", error))?
    };
    let available = tools
        .iter()
        .map(|tool| tool.definition().name.clone())
        .collect::<HashSet<_>>();
    if !cli.tools.is_empty() {
        for requested in &cli.tools {
            if !available.contains(requested) {
                return Err(CliError::NotFound {
                    kind: "tool",
                    name: requested.clone(),
                });
            }
        }
        let requested = cli.tools.iter().map(String::as_str).collect::<HashSet<_>>();
        tools.retain(|tool| requested.contains(tool.definition().name.as_str()));
    }
    let excluded = cli
        .exclude_tools
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    tools.retain(|tool| !excluded.contains(tool.definition().name.as_str()));
    tools.retain(|tool| {
        !resource_is_disabled(
            disabled_resources,
            "tool",
            [tool.definition().name.as_str()],
        )
    });

    let mut options = ri_ext::ResourceLoaderOptions::new(cwd, agent_dir);
    options.home_dir = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from);
    options.project_trusted = project_trusted;
    options.discover_skills = !cli.no_skills;
    options.discover_prompts = !cli.no_prompt_templates;
    options.discover_context = !cli.no_context_files;
    options.explicit_system_prompt = cli.system_prompt.clone().map(ri_ext::PromptInput::Literal);
    options.explicit_append_system = cli
        .append_system_prompt
        .iter()
        .cloned()
        .map(ri_ext::PromptInput::Literal)
        .collect();
    options.additional_skill_paths = if cli.no_skills {
        Vec::new()
    } else {
        configured_resources.skills.clone()
    };
    options.additional_skill_paths.extend(
        cli.skills
            .iter()
            .cloned()
            .map(|path| ri_ext::ResourcePath::configured(path, ri_ext::SourceScope::Temporary))
            .collect::<Vec<_>>(),
    );
    options.additional_prompt_paths = if cli.no_prompt_templates {
        Vec::new()
    } else {
        configured_resources.prompts.clone()
    };
    options.additional_prompt_paths.extend(
        cli.prompt_templates
            .iter()
            .cloned()
            .map(|path| ri_ext::ResourcePath::configured(path, ri_ext::SourceScope::Temporary))
            .collect::<Vec<_>>(),
    );

    for resource in package_resources.iter().filter(|resource| {
        resource.enabled
            && !resource_is_disabled(
                disabled_resources,
                package_resource_kind(resource.kind),
                resource_identifiers(resource),
            )
    }) {
        match resource.kind {
            ri_ext::PackageResourceKind::Extension if !cli.no_extensions => {
                return Err(CliError::unsupported(
                    "load native package extensions",
                    format!(
                        "the SDK has no dynamic native loader for `{}`",
                        resource.path.display()
                    ),
                ));
            }
            ri_ext::PackageResourceKind::Skill if !cli.no_skills => {
                options.package_skill_paths.push(ri_ext::ResourcePath {
                    path: resource.path.clone(),
                    source: resource.source.clone(),
                });
            }
            ri_ext::PackageResourceKind::Prompt if !cli.no_prompt_templates => {
                options.package_prompt_paths.push(ri_ext::ResourcePath {
                    path: resource.path.clone(),
                    source: resource.source.clone(),
                });
            }
            ri_ext::PackageResourceKind::Context if !cli.no_context_files => {}
            ri_ext::PackageResourceKind::Extension
            | ri_ext::PackageResourceKind::Skill
            | ri_ext::PackageResourceKind::Prompt
            | ri_ext::PackageResourceKind::Context => {}
        }
    }
    let mut snapshot = ri_ext::load_resource_snapshot(&options, 1);
    snapshot.skills.retain(|skill| {
        !resource_is_disabled(
            disabled_resources,
            "skill",
            [
                skill.name.as_str(),
                skill.file_path.to_string_lossy().as_ref(),
            ],
        )
    });
    snapshot.prompts.retain(|prompt| {
        !resource_is_disabled(
            disabled_resources,
            "prompt",
            [
                prompt.name.as_str(),
                prompt.file_path.to_string_lossy().as_ref(),
            ],
        )
    });
    let mut context_index = 0_usize;
    snapshot.context.retain(|context| {
        let indexed = format!("context-{context_index}");
        context_index += 1;
        !resource_is_disabled(
            disabled_resources,
            "context",
            [indexed.as_str(), context.path.to_string_lossy().as_ref()],
        )
    });
    if snapshot.system_prompt.is_none() {
        snapshot.system_prompt = Some(default_system_prompt());
    }
    if !cli.no_context_files {
        for resource in package_resources.iter().filter(|resource| {
            resource.enabled
                && resource.kind == ri_ext::PackageResourceKind::Context
                && !resource_is_disabled(
                    disabled_resources,
                    "context",
                    resource_identifiers(resource),
                )
        }) {
            let content = tokio::fs::read_to_string(&resource.path)
                .await
                .map_err(|source| CliError::Io {
                    operation: "read package context",
                    source,
                })?;
            snapshot.context.push(ri_ext::ContextResource {
                path: resource.path.clone(),
                content,
                source: resource.source.clone(),
            });
        }
    }
    ResourceRuntime::from_snapshot(snapshot, tools)
        .map_err(|error| CliError::runtime("materialize resources", error))
}

fn resource_is_disabled<I, S>(
    disabled: &BTreeMap<String, BTreeSet<String>>,
    kind: &str,
    identifiers: I,
) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let Some(names) = disabled.get(kind) else {
        return false;
    };
    if names.contains("*") {
        return true;
    }
    identifiers.into_iter().any(|identifier| {
        let identifier = identifier.as_ref();
        names.contains(identifier)
            || names
                .iter()
                .any(|name| name.replace('\\', "/") == identifier.replace('\\', "/"))
    })
}

fn resource_records_output(records: Vec<Value>, json: bool) -> CommandOutput {
    let mut by_identity = BTreeMap::new();
    for record in records {
        let identity = (
            record
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("-")
                .to_owned(),
            record
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("-")
                .to_owned(),
        );
        by_identity.insert(identity, record);
    }
    let records = by_identity.into_values().collect::<Vec<_>>();
    if json {
        return CommandOutput::Json(Value::Array(records));
    }
    CommandOutput::Text(
        records
            .into_iter()
            .map(|record| {
                let kind = record.get("kind").and_then(Value::as_str).unwrap_or("-");
                let name = record.get("name").and_then(Value::as_str).unwrap_or("-");
                let source = record
                    .get("source")
                    .and_then(Value::as_str)
                    .map_or_else(String::new, |source| format!("\t{source}"));
                let status = if record
                    .get("enabled")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
                {
                    "enabled"
                } else {
                    "disabled"
                };
                format!("{kind}\t{name}\t{status}{source}")
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

const fn resource_kind_label(kind: ResourceKind) -> &'static str {
    match kind {
        ResourceKind::Extension => "extension",
        ResourceKind::Skill => "skill",
        ResourceKind::Prompt => "prompt",
        ResourceKind::Theme => "theme",
        ResourceKind::Context => "context",
        ResourceKind::Tool => "tool",
    }
}

const fn package_resource_kind(kind: ri_ext::PackageResourceKind) -> &'static str {
    match kind {
        ri_ext::PackageResourceKind::Extension => "extension",
        ri_ext::PackageResourceKind::Skill => "skill",
        ri_ext::PackageResourceKind::Prompt => "prompt",
        ri_ext::PackageResourceKind::Context => "context",
    }
}

fn resource_identifiers(resource: &ri_ext::PackageResource) -> Vec<String> {
    let mut identifiers = vec![
        resource.path.to_string_lossy().into_owned(),
        resource.relative_path.to_string_lossy().into_owned(),
    ];
    if let Some(name) = resource.path.file_name().and_then(|name| name.to_str()) {
        identifiers.push(name.to_owned());
    }
    if let Some(stem) = resource.path.file_stem().and_then(|name| name.to_str()) {
        identifiers.push(stem.to_owned());
    }
    if let ri_ext::SourceKind::Package { name, .. } = &resource.source.source {
        identifiers.push(name.clone());
    }
    identifiers
}

fn explicit_provider(cli: &Cli) -> Option<String> {
    cli.provider.clone().or_else(|| {
        cli.model
            .as_deref()
            .or_else(|| cli.models.first().map(String::as_str))
            .and_then(|selector| selector.split_once('/'))
            .map(|(provider, _)| provider.to_owned())
    })
}

fn needs_resources(cli: &Cli) -> bool {
    if !cli.is_metadata_request() {
        return true;
    }
    match cli.command.as_ref() {
        Some(
            Command::Resource {
                command: ResourceCommand::List(arguments),
            }
            | Command::Config(arguments),
        ) => arguments.scope == ResourceScope::All,
        Some(Command::Resource {
            command: ResourceCommand::Reload,
        }) => true,
        _ => false,
    }
}

fn needs_model_refresh(cli: &Cli) -> bool {
    !cli.is_metadata_request()
        || cli.list_models.is_some()
        || matches!(cli.command.as_ref(), Some(Command::Model { .. }))
}

fn explicitly_mutates_project_settings(cli: &Cli) -> bool {
    matches!(
        cli.command.as_ref(),
        Some(
            Command::Package {
                command: PackageCommand::Install(crate::cli::PackageInstallArgs {
                    local: true,
                    ..
                }) | PackageCommand::Remove(crate::cli::PackageRemoveArgs {
                    local: true,
                    ..
                }) | PackageCommand::Uninstall(crate::cli::PackageRemoveArgs {
                    local: true,
                    ..
                }),
            } | Command::Install(crate::cli::PackageInstallArgs { local: true, .. })
                | Command::Remove(crate::cli::PackageRemoveArgs { local: true, .. })
                | Command::Uninstall(crate::cli::PackageRemoveArgs { local: true, .. })
                | Command::Resource {
                    command: ResourceCommand::Enable(ResourceMutationArgs { local: true, .. })
                        | ResourceCommand::Disable(ResourceMutationArgs { local: true, .. }),
                }
        )
    )
}

const fn thinking_level(option: ThinkingOption) -> ThinkingLevel {
    match option {
        ThinkingOption::Off => ThinkingLevel::Off,
        ThinkingOption::Minimal => ThinkingLevel::Minimal,
        ThinkingOption::Low => ThinkingLevel::Low,
        ThinkingOption::Medium => ThinkingLevel::Medium,
        ThinkingOption::High => ThinkingLevel::High,
        ThinkingOption::Xhigh => ThinkingLevel::Xhigh,
        ThinkingOption::Max => ThinkingLevel::Max,
    }
}

fn split_model_selector(selector: &str) -> (&str, Option<ThinkingLevel>) {
    let Some((model, suffix)) = selector.rsplit_once(':') else {
        return (selector, None);
    };
    let thinking = match suffix {
        "off" => ThinkingLevel::Off,
        "minimal" => ThinkingLevel::Minimal,
        "low" => ThinkingLevel::Low,
        "medium" => ThinkingLevel::Medium,
        "high" => ThinkingLevel::High,
        "xhigh" => ThinkingLevel::Xhigh,
        "max" => ThinkingLevel::Max,
        _ => return (selector, None),
    };
    (model, Some(thinking))
}

fn agent_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("RI_AGENT_DIR") {
        return Ok(PathBuf::from(path));
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .map(|home| home.join(".ri").join("agent"))
        .ok_or_else(|| {
            CliError::InvalidArguments(
                "cannot locate the user home directory; set RI_AGENT_DIR".to_owned(),
            )
        })
}

fn default_system_prompt() -> String {
    "You are Ri, a careful coding agent. Inspect the repository before changing it, make the smallest correct edits, use tools when evidence is needed, preserve unrelated work, and report verification honestly.".to_owned()
}

#[cfg(test)]
mod tests {
    use clap::Parser as _;

    use super::*;

    #[test]
    fn model_selector_thinking_suffix_is_optional() {
        assert_eq!(
            split_model_selector("anthropic/sonnet:high"),
            ("anthropic/sonnet", Some(ThinkingLevel::High))
        );
        assert_eq!(
            split_model_selector("provider/model:custom"),
            ("provider/model:custom", None)
        );
    }

    #[test]
    fn metadata_commands_build_only_the_services_they_need() {
        let packages = Cli::try_parse_from(["ri", "package", "list"]).unwrap();
        assert!(!needs_resources(&packages));
        assert!(!needs_model_refresh(&packages));

        let scoped_resources =
            Cli::try_parse_from(["ri", "resource", "list", "--scope", "global"]).unwrap();
        assert!(!needs_resources(&scoped_resources));

        let resources = Cli::try_parse_from(["ri", "resource", "list"]).unwrap();
        assert!(needs_resources(&resources));

        let models = Cli::try_parse_from(["ri", "model", "list"]).unwrap();
        assert!(needs_model_refresh(&models));
    }
}
