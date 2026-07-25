//! Pure import/export for Pi `settings.json`.

use std::collections::BTreeMap;

use ri_rpc::{QueueMode, ThinkingLevel};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Pi provider transport preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PiTransport {
    /// Server-sent events.
    Sse,
    /// WebSocket.
    Websocket,
    /// Cached WebSocket transport.
    WebsocketCached,
    /// Provider-selected transport.
    Auto,
}

/// Default policy for loading project-owned resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PiProjectTrust {
    /// Ask interactively.
    Ask,
    /// Trust by default.
    Always,
    /// Never trust by default.
    Never,
}

/// Double-escape action in the interactive editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PiDoubleEscapeAction {
    /// Open fork selection.
    Fork,
    /// Open tree navigation.
    Tree,
    /// Do nothing.
    None,
}

/// Default tree selector filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiTreeFilterMode {
    /// Standard filter.
    #[serde(rename = "default")]
    Default,
    /// Hide tool messages.
    #[serde(rename = "no-tools")]
    NoTools,
    /// Show user messages only.
    #[serde(rename = "user-only")]
    UserOnly,
    /// Show labeled entries only.
    #[serde(rename = "labeled-only")]
    LabeledOnly,
    /// Show every entry.
    #[serde(rename = "all")]
    All,
}

/// Filtered package resource declaration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PiPackageConfig {
    /// NPM, git, or filesystem package source.
    pub source: String,
    /// Whether resources are auto-discovered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub autoload: Option<bool>,
    /// Extension path filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<String>>,
    /// Skill path filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    /// Prompt path filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<String>>,
    /// Theme path filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub themes: Option<Vec<String>>,
    /// Future package metadata preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// String shorthand or filtered package source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PiPackageSource {
    /// Load every resource from the source.
    Source(String),
    /// Filter resources from the source.
    Config(PiPackageConfig),
}

/// Automatic context-compaction settings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiCompactionSettings {
    /// Enable threshold compaction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Tokens reserved for prompt and response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<u64>,
    /// Recent tokens retained.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_recent_tokens: Option<u64>,
}

/// Branch-summary settings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiBranchSummarySettings {
    /// Tokens reserved for summary generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reserve_tokens: Option<u64>,
    /// Skip the interactive summary prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_prompt: Option<bool>,
}

/// Provider-SDK retry settings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiProviderRetrySettings {
    /// Request timeout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Provider SDK retry attempts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// Maximum accepted server-requested retry delay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retry_delay_ms: Option<u64>,
}

/// Session-level retry settings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiRetrySettings {
    /// Enable session-level retry.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Maximum retry attempts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_retries: Option<u32>,
    /// Exponential-backoff base delay.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_delay_ms: Option<u64>,
    /// Provider-specific request retry controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<PiProviderRetrySettings>,
    /// Future retry fields.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Interactive terminal settings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiTerminalSettings {
    /// Render inline images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_images: Option<bool>,
    /// Preferred inline image width.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_width_cells: Option<u32>,
    /// Clear rows when rendered output shrinks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clear_on_shrink: Option<bool>,
    /// Emit terminal progress sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_terminal_progress: Option<bool>,
}

/// Image preprocessing settings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiImageSettings {
    /// Resize oversized images.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auto_resize: Option<bool>,
    /// Prevent image submission.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_images: Option<bool>,
}

/// Token budgets for budget-based reasoning providers.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PiThinkingBudgets {
    /// Minimal level budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimal: Option<u64>,
    /// Low level budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<u64>,
    /// Medium level budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub medium: Option<u64>,
    /// High level budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<u64>,
}

/// Markdown renderer settings.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiMarkdownSettings {
    /// Code-block continuation indentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_block_indent: Option<String>,
}

/// Optional warning controls.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiWarningSettings {
    /// Show Anthropic extra-usage warnings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anthropic_extra_usage: Option<bool>,
    /// Future warning toggles.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Typed, credential-free Pi settings document.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiSettings {
    /// Last displayed changelog version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_changelog_version: Option<String>,
    /// Default provider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_provider: Option<String>,
    /// Default model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    /// Default reasoning level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_thinking_level: Option<ThinkingLevel>,
    /// Provider transport preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transport: Option<PiTransport>,
    /// Steering queue mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steering_mode: Option<QueueMode>,
    /// Follow-up queue mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub follow_up_mode: Option<QueueMode>,
    /// Theme name or light/dark pair.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    /// Compaction controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction: Option<PiCompactionSettings>,
    /// Branch-summary controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_summary: Option<PiBranchSummarySettings>,
    /// Retry controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry: Option<PiRetrySettings>,
    /// Hide reasoning blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hide_thinking_block: Option<bool>,
    /// Display prompt-cache miss notices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_cache_miss_notices: Option<bool>,
    /// External editor command.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_editor: Option<String>,
    /// Custom shell executable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shell_path: Option<String>,
    /// Suppress startup output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quiet_startup: Option<bool>,
    /// Default project trust.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_project_trust: Option<PiProjectTrust>,
    /// Prefix prepended to shell commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shell_command_prefix: Option<String>,
    /// NPM command argv.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub npm_command: Option<Vec<String>>,
    /// Collapse changelog on startup.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub collapse_changelog: Option<bool>,
    /// Enable anonymous update checks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_install_telemetry: Option<bool>,
    /// Enable analytics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_analytics: Option<bool>,
    /// Analytics tracking identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tracking_id: Option<String>,
    /// Package resources.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packages: Option<Vec<PiPackageSource>>,
    /// Explicit extension paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<Vec<String>>,
    /// Explicit skill paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
    /// Explicit prompt paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<Vec<String>>,
    /// Explicit theme paths.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub themes: Option<Vec<String>>,
    /// Register skill slash commands.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_skill_commands: Option<bool>,
    /// Terminal controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminal: Option<PiTerminalSettings>,
    /// Image controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<PiImageSettings>,
    /// Model-cycle patterns.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled_models: Option<Vec<String>>,
    /// Double-escape behavior.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub double_escape_action: Option<PiDoubleEscapeAction>,
    /// Tree selector filter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tree_filter_mode: Option<PiTreeFilterMode>,
    /// Custom reasoning budgets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budgets: Option<PiThinkingBudgets>,
    /// Editor horizontal padding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub editor_padding_x: Option<u8>,
    /// Chat output padding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_pad: Option<u8>,
    /// Autocomplete visible-row count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub autocomplete_max_visible: Option<u32>,
    /// Show the hardware cursor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_hardware_cursor: Option<bool>,
    /// Markdown controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub markdown: Option<PiMarkdownSettings>,
    /// Warning controls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warnings: Option<PiWarningSettings>,
    /// Custom session directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_dir: Option<String>,
    /// HTTP proxy URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_proxy: Option<String>,
    /// HTTP stream idle timeout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub http_idle_timeout_ms: Option<u64>,
    /// WebSocket handshake timeout.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub websocket_connect_timeout_ms: Option<u64>,
    /// Future non-credential settings preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Settings import/export failure.
#[derive(Debug, thiserror::Error)]
pub enum PiSettingsError {
    /// Invalid JSON or typed field.
    #[error("invalid Pi settings: {0}")]
    Json(#[from] serde_json::Error),
    /// Top-level settings value was not an object.
    #[error("Pi settings must be a JSON object")]
    Root,
}

/// Import caller-provided Pi settings and apply Pi's legacy migrations.
///
/// This function does not locate `.pi`, read auth files, or resolve environment
/// variables and commands.
///
/// # Errors
///
/// Returns an error if the input is not a JSON object or cannot be decoded as
/// typed Pi settings after migration.
pub fn import_settings(input: &[u8]) -> Result<PiSettings, PiSettingsError> {
    let mut value: Value = serde_json::from_slice(input)?;
    migrate_settings(&mut value)?;
    serde_json::from_value(value).map_err(PiSettingsError::from)
}

/// Export typed settings as deterministic pretty JSON.
///
/// # Errors
///
/// Returns an error if the typed settings cannot be serialized as JSON.
pub fn export_settings(settings: &PiSettings) -> Result<String, PiSettingsError> {
    serde_json::to_string_pretty(settings).map_err(PiSettingsError::from)
}

fn migrate_settings(value: &mut Value) -> Result<(), PiSettingsError> {
    let settings = value.as_object_mut().ok_or(PiSettingsError::Root)?;

    if !settings.contains_key("steeringMode") {
        if let Some(queue_mode) = settings.remove("queueMode") {
            settings.insert("steeringMode".to_owned(), queue_mode);
        }
    }

    if !settings.contains_key("transport") {
        if let Some(websockets) = settings.get("websockets").and_then(Value::as_bool) {
            settings.insert(
                "transport".to_owned(),
                Value::String(if websockets { "websocket" } else { "sse" }.to_owned()),
            );
            settings.remove("websockets");
        }
    }

    let legacy_skills = settings
        .get("skills")
        .and_then(Value::as_object)
        .map(|object| {
            (
                object.get("enableSkillCommands").cloned(),
                object.get("customDirectories").cloned(),
            )
        });
    if let Some((legacy_enable, legacy_directories)) = legacy_skills {
        if !settings.contains_key("enableSkillCommands") {
            if let Some(enabled) = legacy_enable {
                settings.insert("enableSkillCommands".to_owned(), enabled);
            }
        }
        match legacy_directories {
            Some(Value::Array(directories)) if !directories.is_empty() => {
                settings.insert("skills".to_owned(), Value::Array(directories));
            }
            _ => {
                settings.remove("skills");
            }
        }
    }

    if let Some(retry) = settings.get_mut("retry").and_then(Value::as_object_mut) {
        let max_delay = retry.get("maxDelayMs").and_then(Value::as_u64);
        if let Some(max_delay) = max_delay {
            let provider_is_object = retry.get("provider").is_some_and(Value::is_object);
            if !provider_is_object {
                retry.insert("provider".to_owned(), Value::Object(serde_json::Map::new()));
            }
            if let Some(provider) = retry.get_mut("provider").and_then(Value::as_object_mut) {
                if provider.get("maxRetryDelayMs").is_none_or(Value::is_null) {
                    provider.insert(
                        "maxRetryDelayMs".to_owned(),
                        Value::Number(max_delay.into()),
                    );
                }
            }
        }
        retry.remove("maxDelayMs");
    }

    Ok(())
}
