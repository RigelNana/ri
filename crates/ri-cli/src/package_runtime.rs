//! CLI-owned package settings over `ri-ext`'s resolver and lockfile.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use ri_ext::{
    GenerationClock, PackageFilter, PackageManager, PackageManagerOptions, PackageScope,
    PackageSnapshot, PackageSource, PackageSpec, QueueMode, ResolvedCompactionSettings,
    ResolvedRetrySettings, ResourcePath, SettingsManager, SourceScope,
};
use serde_json::{Value, json};

use crate::cli::ResourceKind;
use crate::error::{CliError, Result};

const RESOURCE_OVERRIDES_KEY: &str = "ri_cli_resource_overrides";

#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeSettings {
    pub(crate) steering_mode: QueueMode,
    pub(crate) follow_up_mode: QueueMode,
    pub(crate) compaction: ResolvedCompactionSettings,
    pub(crate) retry: ResolvedRetrySettings,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ConfiguredResourcePaths {
    pub(crate) extensions: Vec<ResourcePath>,
    pub(crate) skills: Vec<ResourcePath>,
    pub(crate) prompts: Vec<ResourcePath>,
    pub(crate) themes: Vec<ResourcePath>,
}

#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
#[serde(default)]
struct ResourceOverrides {
    disabled: BTreeMap<String, BTreeSet<String>>,
    enabled: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Clone, Debug)]
pub(crate) struct PackageRuntime {
    cwd: PathBuf,
    agent_dir: PathBuf,
    offline: bool,
    project_trusted: bool,
    settings: SettingsManager,
}

impl PackageRuntime {
    pub(crate) async fn new(
        cwd: PathBuf,
        agent_dir: PathBuf,
        offline: bool,
        project_trust_override: Option<bool>,
    ) -> Result<Self> {
        let initially_trusted = project_trust_override.unwrap_or(false);
        let settings = SettingsManager::open_files(&cwd, &agent_dir, initially_trusted).await;
        let errors = settings.drain_errors();
        if !errors.is_empty() {
            return Err(CliError::InvalidConfig {
                message: errors
                    .into_iter()
                    .map(|error| format!("{:?}: {}", error.scope, error.message))
                    .collect::<Vec<_>>()
                    .join("; "),
            });
        }
        let project_trusted = match project_trust_override {
            Some(trusted) => trusted,
            None => {
                settings.global().await.default_project_trust
                    == Some(ri_ext::DefaultProjectTrust::Always)
            }
        };
        if project_trusted != initially_trusted {
            settings
                .set_project_trusted(project_trusted)
                .await
                .map_err(|error| CliError::runtime("apply project trust setting", error))?;
            let errors = settings.drain_errors();
            if !errors.is_empty() {
                return Err(CliError::InvalidConfig {
                    message: errors
                        .into_iter()
                        .map(|error| format!("{:?}: {}", error.scope, error.message))
                        .collect::<Vec<_>>()
                        .join("; "),
                });
            }
        }
        Ok(Self {
            cwd,
            agent_dir,
            offline,
            project_trusted,
            settings,
        })
    }

    pub(crate) const fn project_trusted(&self) -> bool {
        self.project_trusted
    }

    pub(crate) async fn configured_model(&self) -> (Option<String>, Option<String>) {
        let settings = self.settings.effective().await;
        (settings.default_provider, settings.default_model)
    }

    pub(crate) async fn configured_session_dir(&self) -> Option<PathBuf> {
        self.settings.effective().await.session_dir
    }

    pub(crate) async fn configured_thinking_level(&self) -> Option<ri_ai::ThinkingLevel> {
        self.settings.effective().await.default_thinking_level
    }

    pub(crate) async fn configured_runtime(&self) -> RuntimeSettings {
        let settings = self.settings.effective().await;
        RuntimeSettings {
            steering_mode: settings.steering_mode.unwrap_or_default(),
            follow_up_mode: settings.follow_up_mode.unwrap_or_default(),
            compaction: settings.resolved_compaction(),
            retry: settings.resolved_retry(),
        }
    }

    pub(crate) async fn configured_resource_paths(&self) -> ConfiguredResourcePaths {
        let global = self.settings.global().await.resources;
        let project = self.settings.project().await.resources;
        ConfiguredResourcePaths {
            extensions: select_resource_paths(global.extensions, project.extensions),
            skills: select_resource_paths(global.skills, project.skills),
            prompts: select_resource_paths(global.prompts, project.prompts),
            themes: select_resource_paths(global.themes, project.themes),
        }
    }

    pub(crate) async fn set_auto_compaction_enabled(&self, enabled: bool) -> Result<()> {
        self.settings
            .update_global(move |settings| {
                settings
                    .compaction
                    .get_or_insert_with(ri_ext::CompactionSettings::default)
                    .enabled = Some(enabled);
            })
            .await
            .map_err(|error| CliError::runtime("queue compaction setting", error))?;
        self.flush_settings().await
    }

    pub(crate) async fn set_auto_retry_enabled(&self, enabled: bool) -> Result<()> {
        self.settings
            .update_global(move |settings| {
                settings
                    .retry
                    .get_or_insert_with(ri_ext::RetrySettings::default)
                    .enabled = Some(enabled);
            })
            .await
            .map_err(|error| CliError::runtime("queue retry setting", error))?;
        self.flush_settings().await
    }

    pub(crate) async fn set_default_model(&self, provider: &str, model: &str) -> Result<()> {
        self.settings
            .set_default_model(provider.to_owned(), model.to_owned())
            .await
            .map_err(|error| CliError::runtime("queue default model setting", error))?;
        self.flush_settings().await
    }

    pub(crate) async fn disabled_resources(&self) -> Result<BTreeMap<String, BTreeSet<String>>> {
        let global = decode_resource_overrides(&self.settings.global().await)?;
        let project = decode_resource_overrides(&self.settings.project().await)?;
        let mut disabled = global.disabled;
        for (kind, names) in project.enabled {
            if let Some(disabled_names) = disabled.get_mut(&kind) {
                disabled_names.retain(|name| !names.contains(name));
            }
        }
        for (kind, names) in project.disabled {
            disabled.entry(kind).or_default().extend(names);
        }
        disabled.retain(|_, names| !names.is_empty());
        Ok(disabled)
    }

    pub(crate) async fn resource_overrides(
        &self,
        local: bool,
    ) -> Result<Vec<(String, String, bool)>> {
        let settings = if local {
            self.settings.project().await
        } else {
            self.settings.global().await
        };
        let overrides = decode_resource_overrides(&settings)?;
        let mut records = BTreeMap::new();
        for (kind, names) in overrides.enabled {
            for name in names {
                records.insert((kind.clone(), name), true);
            }
        }
        for (kind, names) in overrides.disabled {
            for name in names {
                records.insert((kind.clone(), name), false);
            }
        }
        Ok(records
            .into_iter()
            .map(|((kind, name), enabled)| (kind, name, enabled))
            .collect())
    }

    pub(crate) async fn set_resource_enabled(
        &self,
        kind: ResourceKind,
        name: &str,
        local: bool,
        enabled: bool,
    ) -> Result<()> {
        let mut settings = if local {
            self.settings.project().await
        } else {
            self.settings.global().await
        };
        let mut overrides = decode_resource_overrides(&settings)?;
        let key = resource_kind_key(kind).to_owned();
        let disabled = overrides.disabled.entry(key.clone()).or_default();
        if enabled {
            disabled.remove(name);
        } else {
            disabled.insert(name.to_owned());
        }
        let explicitly_enabled = overrides.enabled.entry(key).or_default();
        if enabled && local {
            explicitly_enabled.insert(name.to_owned());
        } else {
            explicitly_enabled.remove(name);
        }
        overrides.disabled.retain(|_, names| !names.is_empty());
        overrides.enabled.retain(|_, names| !names.is_empty());
        settings.extension.insert(
            RESOURCE_OVERRIDES_KEY.to_owned(),
            serde_json::to_value(overrides).map_err(|source| CliError::Json {
                operation: "encode resource settings",
                source,
            })?,
        );
        if local {
            self.settings
                .update_project(move |target| *target = settings)
                .await
        } else {
            self.settings
                .update_global(move |target| *target = settings)
                .await
        }
        .map_err(|error| CliError::runtime("queue resource settings", error))?;
        self.flush_settings().await
    }

    pub(crate) async fn resolve(&self, update: bool) -> Result<PackageSnapshot> {
        let specs = self.specs().await?;
        if specs.is_empty() {
            return Ok(PackageSnapshot {
                generation: 1,
                ..PackageSnapshot::default()
            });
        }
        self.resolve_specs(&specs, update).await
    }

    pub(crate) async fn install(
        &self,
        source: &str,
        local: bool,
        checksum: Option<&str>,
    ) -> Result<String> {
        let scope = if local {
            PackageScope::Project
        } else {
            PackageScope::User
        };
        let parsed = parse_source(source, &self.cwd)?;
        let identity = parsed.identity();
        let mut scoped = self.load_scope(scope).await?;
        if scoped
            .iter()
            .any(|existing| existing.source.identity() == identity)
        {
            return Err(CliError::InvalidArguments(format!(
                "package `{identity}` is already configured"
            )));
        }
        scoped.push(PackageSpec {
            source: parsed,
            scope,
            checksum: checksum.map(str::to_owned),
            filter: PackageFilter::default(),
        });
        let combined = self.with_scope(scope, &scoped).await?;
        self.resolve_specs(&combined, false).await?;
        self.save_scope(scope, &scoped).await?;
        Ok(identity)
    }

    pub(crate) async fn remove(&self, source: &str, local: bool) -> Result<String> {
        let scope = if local {
            PackageScope::Project
        } else {
            PackageScope::User
        };
        let identity = parse_source(source, &self.cwd)?.identity();
        let mut scoped = self.load_scope(scope).await?;
        let before = scoped.len();
        scoped.retain(|spec| spec.source.identity() != identity);
        if scoped.len() == before {
            return Err(CliError::NotFound {
                kind: "package",
                name: identity,
            });
        }
        let combined = self.with_scope(scope, &scoped).await?;
        self.resolve_specs(&combined, false).await?;
        self.save_scope(scope, &scoped).await?;
        Ok(identity)
    }

    pub(crate) async fn update(
        &self,
        source: Option<&str>,
        all: bool,
        force: bool,
    ) -> Result<usize> {
        let specs = self.specs().await?;
        if specs.is_empty() {
            return Ok(0);
        }
        if !all && let Some(source) = source {
            let identity = parse_source(source, &self.cwd)?.identity();
            if !specs.iter().any(|spec| spec.source.identity() == identity) {
                return Err(CliError::NotFound {
                    kind: "package",
                    name: identity,
                });
            }
        }
        // PackageManager updates one atomic lock snapshot. Its update switch is
        // intentionally transaction-wide, so a targeted request validates the
        // target but refreshes the complete lock to avoid a mixed-generation set.
        self.resolve_specs(&specs, force || all || source.is_some())
            .await?;
        Ok(specs.len())
    }

    pub(crate) async fn records(&self) -> Result<Vec<Value>> {
        let snapshot = self.resolve(false).await?;
        Ok(snapshot
            .packages
            .into_iter()
            .map(|package| {
                json!({
                    "name": package.manifest.package.name,
                    "version": package.manifest.package.version,
                    "description": package.manifest.package.description,
                    "root": package.root,
                    "checksum": package.checksum,
                    "source": package.metadata,
                    "resources": package.resources.iter().map(|resource| {
                        json!({
                            "kind": resource.kind,
                            "path": resource.path,
                            "enabled": resource.enabled,
                        })
                    }).collect::<Vec<_>>(),
                })
            })
            .collect())
    }

    async fn resolve_specs(&self, specs: &[PackageSpec], update: bool) -> Result<PackageSnapshot> {
        let project_root = self.cwd.join(".ri");
        tokio::fs::create_dir_all(&project_root)
            .await
            .map_err(|source| CliError::Io {
                operation: "create project package directory",
                source,
            })?;
        tokio::fs::create_dir_all(&self.agent_dir)
            .await
            .map_err(|source| CliError::Io {
                operation: "create user package directory",
                source,
            })?;
        let mut options = PackageManagerOptions::new(
            &self.cwd,
            &self.agent_dir,
            self.agent_dir.join("package-cache"),
            project_root.join("package-lock.toml"),
        );
        options.project_trusted = self.project_trusted;
        options.offline = self.offline;
        options.update = update;
        let mut manager = PackageManager::new(options, GenerationClock::default());
        manager
            .reload(specs)
            .await
            .map_err(|error| CliError::runtime("resolve packages", error))?;
        Ok(manager.snapshot().clone())
    }

    async fn specs(&self) -> Result<Vec<PackageSpec>> {
        let mut specs = self.load_scope(PackageScope::User).await?;
        specs.extend(self.load_scope(PackageScope::Project).await?);
        Ok(specs)
    }

    async fn with_scope(
        &self,
        scope: PackageScope,
        replacement: &[PackageSpec],
    ) -> Result<Vec<PackageSpec>> {
        let other = match scope {
            PackageScope::User => PackageScope::Project,
            PackageScope::Project => PackageScope::User,
            PackageScope::Temporary => {
                return Err(CliError::InvalidArguments(
                    "temporary packages cannot be persisted".to_owned(),
                ));
            }
        };
        let mut combined = replacement.to_vec();
        combined.extend(self.load_scope(other).await?);
        Ok(combined)
    }

    async fn load_scope(&self, scope: PackageScope) -> Result<Vec<PackageSpec>> {
        let settings = match scope {
            PackageScope::User => self.settings.global().await,
            PackageScope::Project => self.settings.project().await,
            PackageScope::Temporary => {
                return Err(CliError::InvalidArguments(
                    "temporary packages cannot be persisted".to_owned(),
                ));
            }
        };
        Ok(settings.packages.unwrap_or_default())
    }

    async fn save_scope(&self, scope: PackageScope, specs: &[PackageSpec]) -> Result<()> {
        match scope {
            PackageScope::User => self.settings.set_packages(specs.to_vec()).await,
            PackageScope::Project => self.settings.set_project_packages(specs.to_vec()).await,
            PackageScope::Temporary => {
                return Err(CliError::InvalidArguments(
                    "temporary packages cannot be persisted".to_owned(),
                ));
            }
        }
        .map_err(|error| CliError::runtime("queue package settings", error))?;
        self.flush_settings().await
    }

    async fn flush_settings(&self) -> Result<()> {
        self.settings
            .flush()
            .await
            .map_err(|error| CliError::runtime("flush settings", error))?;
        let errors = self.settings.drain_errors();
        if !errors.is_empty() {
            return Err(CliError::InvalidConfig {
                message: errors
                    .into_iter()
                    .map(|error| format!("{:?}: {}", error.scope, error.message))
                    .collect::<Vec<_>>()
                    .join("; "),
            });
        }
        Ok(())
    }
}

fn select_resource_paths(
    global: Option<Vec<PathBuf>>,
    project: Option<Vec<PathBuf>>,
) -> Vec<ResourcePath> {
    let (paths, scope) = project.map_or((global, SourceScope::User), |paths| {
        (Some(paths), SourceScope::Project)
    });
    paths
        .unwrap_or_default()
        .into_iter()
        .map(|path| ResourcePath::configured(path, scope))
        .collect()
}

fn decode_resource_overrides(settings: &ri_ext::Settings) -> Result<ResourceOverrides> {
    settings
        .extension
        .get(RESOURCE_OVERRIDES_KEY)
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map(Option::unwrap_or_default)
        .map_err(|source| CliError::Json {
            operation: "decode resource settings",
            source,
        })
}

const fn resource_kind_key(kind: ResourceKind) -> &'static str {
    match kind {
        ResourceKind::Extension => "extension",
        ResourceKind::Skill => "skill",
        ResourceKind::Prompt => "prompt",
        ResourceKind::Theme => "theme",
        ResourceKind::Context => "context",
        ResourceKind::Tool => "tool",
    }
}

fn parse_source(source: &str, cwd: &Path) -> Result<PackageSource> {
    if let Some(value) = source.strip_prefix("git:") {
        let (repository, revision) = value
            .rsplit_once('#')
            .map_or((value, None), |(repository, revision)| {
                (repository, Some(revision.to_owned()))
            });
        if repository.trim().is_empty() {
            return Err(CliError::InvalidArguments(
                "git package source cannot be empty".to_owned(),
            ));
        }
        return Ok(PackageSource::Git {
            repository: repository.to_owned(),
            revision,
        });
    }
    let https = if source.starts_with("https://") {
        source
    } else {
        source.strip_prefix("https:").unwrap_or(source)
    };
    if https.starts_with("https://") {
        return Ok(PackageSource::Https {
            manifest_url: https.parse().map_err(|error| {
                CliError::InvalidArguments(format!(
                    "invalid HTTPS package source `{source}`: {error}"
                ))
            })?,
        });
    }
    let path = source.strip_prefix("local:").unwrap_or(source);
    if path.trim().is_empty() {
        return Err(CliError::InvalidArguments(
            "local package path cannot be empty".to_owned(),
        ));
    }
    let path = PathBuf::from(path);
    let path = if path.is_absolute() {
        path
    } else {
        cwd.join(path)
    };
    Ok(PackageSource::Local {
        path: path.components().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_package_sources() {
        let cwd = Path::new("/workspace");
        assert_eq!(
            parse_source("git:https://example.test/repo#main", cwd)
                .unwrap()
                .identity(),
            "git:https://example.test/repo"
        );
        assert_eq!(
            parse_source("https://example.test/ri-package.toml", cwd)
                .unwrap()
                .identity(),
            "https:https://example.test/ri-package.toml"
        );
        assert_eq!(
            parse_source("./local", cwd).unwrap().identity(),
            format!(
                "local:{}",
                cwd.join("local")
                    .components()
                    .collect::<PathBuf>()
                    .display()
            )
        );
    }

    #[tokio::test]
    async fn global_project_trust_default_and_override_are_applied() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = temporary.path().join("workspace");
        let agent_dir = temporary.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(
            agent_dir.join("settings.json"),
            r#"{"defaultProjectTrust":"always"}"#,
        )
        .unwrap();

        let inherited = PackageRuntime::new(cwd.clone(), agent_dir.clone(), true, None)
            .await
            .unwrap();
        assert!(inherited.project_trusted());

        let denied = PackageRuntime::new(cwd, agent_dir, true, Some(false))
            .await
            .unwrap();
        assert!(!denied.project_trusted());
    }

    #[tokio::test]
    async fn pi_runtime_settings_are_resolved_and_live_toggles_are_durable() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = temporary.path().join("workspace");
        let agent_dir = temporary.path().join("agent");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(&agent_dir).unwrap();
        let path = agent_dir.join("settings.json");
        std::fs::write(
            &path,
            r#"{
                "defaultThinkingLevel":"high",
                "steeringMode":"all",
                "followUpMode":"one-at-a-time",
                "compaction":{"enabled":false,"reserveTokens":99,"keepRecentTokens":77},
                "retry":{"enabled":false,"maxRetries":5,"baseDelayMs":12,"provider":{"timeoutMs":34,"maxRetries":2,"maxRetryDelayMs":56}},
                "theme":"dark"
            }"#,
        )
        .unwrap();

        let runtime = PackageRuntime::new(cwd, agent_dir, true, Some(false))
            .await
            .unwrap();
        assert_eq!(
            runtime.configured_thinking_level().await,
            Some(ri_ai::ThinkingLevel::High)
        );
        let settings = runtime.configured_runtime().await;
        assert_eq!(settings.steering_mode, QueueMode::All);
        assert_eq!(settings.follow_up_mode, QueueMode::OneAtATime);
        assert_eq!(settings.compaction.reserve_tokens, 99);
        assert_eq!(settings.compaction.keep_recent_tokens, 77);
        assert!(!settings.compaction.enabled);
        assert_eq!(settings.retry.max_retries, 5);
        assert_eq!(settings.retry.base_delay_ms, 12);
        assert_eq!(settings.retry.provider_timeout_ms, Some(34));
        assert_eq!(settings.retry.provider_max_retries, Some(2));
        assert_eq!(settings.retry.provider_max_retry_delay_ms, 56);

        runtime.set_auto_compaction_enabled(true).await.unwrap();
        runtime.set_auto_retry_enabled(true).await.unwrap();
        let stored: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(stored["compaction"]["enabled"], true);
        assert_eq!(stored["retry"]["enabled"], true);
        assert_eq!(stored["theme"], "dark");
    }
}
