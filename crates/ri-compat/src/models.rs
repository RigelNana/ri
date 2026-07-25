//! Pure, non-resolving import/export for Pi `models.json`.

use std::collections::BTreeMap;
use std::fmt;

use ri_rpc::{ModelInput, ThinkingLevel};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An unresolved `models.json` string.
///
/// Pi may interpret leading `!`, `$NAME`, and `${NAME}` at request time. This
/// compatibility type stores text only and never executes or interpolates it.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PiUnresolvedString(pub String);

impl fmt::Debug for PiUnresolvedString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PiUnresolvedString")
            .field(&"<redacted>")
            .finish()
    }
}

/// Supported built-in OAuth selector in `models.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PiModelOauth {
    /// Radius dynamic OAuth.
    Radius,
}

/// `OpenAI` max-token field spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PiMaxTokensField {
    /// `max_completion_tokens`.
    MaxCompletionTokens,
    /// `max_tokens`.
    MaxTokens,
}

/// Provider reasoning payload convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiThinkingFormat {
    /// `OpenAI` `reasoning_effort`.
    #[serde(rename = "openai")]
    OpenAi,
    /// `OpenRouter` reasoning object.
    #[serde(rename = "openrouter")]
    OpenRouter,
    /// Together reasoning object.
    #[serde(rename = "together")]
    Together,
    /// `DeepSeek` thinking object.
    #[serde(rename = "deepseek")]
    DeepSeek,
    /// z.ai thinking object.
    #[serde(rename = "zai")]
    Zai,
    /// Qwen top-level switch.
    #[serde(rename = "qwen")]
    Qwen,
    /// Configurable chat-template kwargs.
    #[serde(rename = "chat-template")]
    ChatTemplate,
    /// Qwen chat-template kwargs.
    #[serde(rename = "qwen-chat-template")]
    QwenChatTemplate,
    /// String-valued thinking field.
    #[serde(rename = "string-thinking")]
    StringThinking,
    /// Ant Ling reasoning object.
    #[serde(rename = "ant-ling")]
    AntLing,
}

/// Session-affinity header convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiSessionAffinityFormat {
    /// `OpenAI` headers including `session_id`.
    #[serde(rename = "openai")]
    OpenAi,
    /// `OpenAI` headers without `session_id`.
    #[serde(rename = "openai-nosession")]
    OpenAiNoSession,
    /// `OpenRouter` `x-session-id`.
    #[serde(rename = "openrouter")]
    OpenRouter,
}

/// Prompt-cache marker convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PiCacheControlFormat {
    /// Anthropic `cache_control`.
    Anthropic,
}

/// Deferred-tool serialization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PiDeferredToolsMode {
    /// Kimi deferred tools.
    Kimi,
}

/// `OpenRouter` data-collection preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PiDataCollection {
    /// Disallow provider data collection.
    Deny,
    /// Allow provider data collection.
    Allow,
}

/// Number or string accepted by `OpenRouter` price fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PiNumberOrString {
    /// Numeric form.
    Number(f64),
    /// String form.
    String(String),
}

/// `OpenRouter` percentile cutoffs.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PiPercentileCutoffs {
    /// 50th percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p50: Option<f64>,
    /// 75th percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p75: Option<f64>,
    /// 90th percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p90: Option<f64>,
    /// 99th percentile.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p99: Option<f64>,
}

/// Scalar or percentile routing threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PiRoutingThreshold {
    /// Scalar threshold.
    Number(f64),
    /// Per-percentile thresholds.
    Percentiles(PiPercentileCutoffs),
}

/// Structured `OpenRouter` sort setting.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PiOpenRouterSortConfig {
    /// Sort metric.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub by: Option<String>,
    /// Provider partition; explicit null is represented by `None`.
    #[serde(default)]
    pub partition: Option<String>,
}

/// `OpenRouter` sort shorthand or object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PiOpenRouterSort {
    /// String shorthand.
    Name(String),
    /// Structured sort.
    Config(PiOpenRouterSortConfig),
}

/// `OpenRouter` maximum prices.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PiOpenRouterMaxPrice {
    /// Prompt price.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PiNumberOrString>,
    /// Completion price.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion: Option<PiNumberOrString>,
    /// Image price.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<PiNumberOrString>,
    /// Audio price.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<PiNumberOrString>,
    /// Request price.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request: Option<PiNumberOrString>,
}

/// `OpenRouter` provider routing settings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PiOpenRouterRouting {
    /// Permit fallback providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    /// Require support for all request parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
    /// Data collection preference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<PiDataCollection>,
    /// Require zero-data-retention providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zdr: Option<bool>,
    /// Require distillable text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enforce_distillable_text: Option<bool>,
    /// Preferred provider order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
    /// Allowed providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Excluded providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,
    /// Allowed quantizations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantizations: Option<Vec<String>>,
    /// Sort setting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<PiOpenRouterSort>,
    /// Maximum prices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_price: Option<PiOpenRouterMaxPrice>,
    /// Preferred minimum throughput.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_min_throughput: Option<PiRoutingThreshold>,
    /// Preferred maximum latency.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_max_latency: Option<PiRoutingThreshold>,
}

/// Vercel AI Gateway provider routing.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PiVercelGatewayRouting {
    /// Allowed providers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub only: Option<Vec<String>>,
    /// Preferred provider order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,
}

/// Dynamic value in `chatTemplateKwargs`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PiChatTemplateVariable {
    /// Runtime variable name.
    #[serde(rename = "$var")]
    pub variable: PiChatTemplateVariableName,
    /// Omit this kwarg when reasoning is off.
    #[serde(rename = "omitWhenOff", skip_serializing_if = "Option::is_none")]
    pub omit_when_off: Option<bool>,
}

/// Runtime variables accepted in `chatTemplateKwargs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiChatTemplateVariableName {
    /// Boolean reasoning state.
    #[serde(rename = "thinking.enabled")]
    ThinkingEnabled,
    /// Mapped reasoning effort.
    #[serde(rename = "thinking.effort")]
    ThinkingEffort,
}

/// Scalar or runtime-variable chat-template kwarg.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PiChatTemplateKwarg {
    /// String scalar.
    String(String),
    /// Numeric scalar.
    Number(f64),
    /// Boolean scalar.
    Boolean(bool),
    /// Null scalar.
    Null,
    /// Runtime variable.
    Variable(PiChatTemplateVariable),
}

/// API-specific compatibility switches accepted by Pi.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiProviderCompat {
    /// `OpenAI` store-field support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    /// Developer-role support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Reasoning-effort support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    /// Streaming usage support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    /// Max-token field spelling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<PiMaxTokensField>,
    /// Tool result requires a name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    /// Assistant separator after a tool result.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    /// Convert reasoning blocks to text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    /// Replay empty reasoning content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    /// Provider reasoning convention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<PiThinkingFormat>,
    /// Chat-template values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<BTreeMap<String, PiChatTemplateKwarg>>,
    /// Prompt-cache marker convention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<PiCacheControlFormat>,
    /// `OpenRouter` routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub open_router_routing: Option<PiOpenRouterRouting>,
    /// Vercel AI Gateway routing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vercel_gateway_routing: Option<PiVercelGatewayRouting>,
    /// `OpenAI` grammar-tool support.
    #[serde(
        rename = "supportsOpenAIGrammarTools",
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_open_ai_grammar_tools: Option<bool>,
    /// Strict schema/tool mode support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// Emit session-affinity headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// Deferred-tool convention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deferred_tools_mode: Option<PiDeferredToolsMode>,
    /// Session-affinity header convention.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<PiSessionAffinityFormat>,
    /// Long cache-retention support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    /// Responses API tool-search support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    /// Anthropic eager tool-input streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    /// Cache-control markers on tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    /// Temperature support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    /// Force Anthropic adaptive thinking.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    /// Preserve empty Anthropic signatures.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    /// Anthropic strict-tool support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    /// Anthropic tool-reference support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports_tool_references: Option<bool>,
    /// Future compatibility switches.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Complete alternate token rates above a threshold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiModelCostTier {
    /// Total input-token threshold.
    pub input_tokens_above: u64,
    /// Input rate.
    pub input: f64,
    /// Output rate.
    pub output: f64,
    /// Cache-read rate.
    pub cache_read: f64,
    /// Cache-write rate.
    pub cache_write: f64,
}

/// Complete custom-model token rates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiModelCost {
    /// Input rate.
    pub input: f64,
    /// Output rate.
    pub output: f64,
    /// Cache-read rate.
    pub cache_read: f64,
    /// Cache-write rate.
    pub cache_write: f64,
    /// Optional long-context tiers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<PiModelCostTier>>,
}

/// Partial token-rate override.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiModelCostOverride {
    /// Input rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<f64>,
    /// Output rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<f64>,
    /// Cache-read rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<f64>,
    /// Cache-write rate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
    /// Replacement long-context tiers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tiers: Option<Vec<PiModelCostTier>>,
}

/// A custom model definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiModelDefinition {
    /// Provider model identifier.
    pub id: String,
    /// Display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Wire API identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Per-model base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Reasoning support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// Explicit level map; missing keys and explicit null values remain distinct.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<BTreeMap<ThinkingLevel, Option<String>>>,
    /// Input modalities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<ModelInput>>,
    /// Token rates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<PiModelCost>,
    /// Context-window size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Literal model headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// API compatibility settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<PiProviderCompat>,
    /// Future model fields.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Per-model override for built-in or extension models.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiModelOverride {
    /// Display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Reasoning support.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<bool>,
    /// Explicit level map.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<BTreeMap<ThinkingLevel, Option<String>>>,
    /// Input modalities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<ModelInput>>,
    /// Partial token-rate override.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cost: Option<PiModelCostOverride>,
    /// Context-window size.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    /// Maximum output tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Literal model headers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// API compatibility settings.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<PiProviderCompat>,
    /// Future override fields.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// One provider entry in Pi `models.json`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PiProviderConfig {
    /// Display name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Provider base URL.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Unresolved key expression. Import never resolves it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key: Option<PiUnresolvedString>,
    /// Default wire API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// Optional dynamic OAuth selector.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oauth: Option<PiModelOauth>,
    /// Unresolved header expressions. Import never resolves them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, PiUnresolvedString>>,
    /// Provider-level compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compat: Option<PiProviderCompat>,
    /// Add a bearer authorization header.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<bool>,
    /// Custom models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<PiModelDefinition>>,
    /// Overrides keyed by model identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_overrides: Option<BTreeMap<String, PiModelOverride>>,
    /// Future provider fields.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Typed Pi `models.json` document.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PiModelsConfig {
    /// Providers keyed by identifier.
    pub providers: BTreeMap<String, PiProviderConfig>,
    /// Future root metadata.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Models import/export failure.
#[derive(Debug, thiserror::Error)]
pub enum PiModelsError {
    /// Input contains invalid UTF-8 or JSON.
    #[error("invalid Pi models configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// A block comment did not terminate.
    #[error("unterminated block comment in Pi models configuration")]
    UnterminatedBlockComment,
}

/// Import caller-provided `models.json`, accepting Pi's JSON comments.
///
/// No key expression, environment reference, command, OAuth flow, credential
/// store, or filesystem path is resolved.
///
/// # Errors
///
/// Returns an error if a block comment is unterminated or the comment-stripped
/// document is not valid typed JSON.
pub fn import_models(input: &[u8]) -> Result<PiModelsConfig, PiModelsError> {
    let stripped = strip_json_comments(input)?;
    serde_json::from_slice(&stripped).map_err(PiModelsError::from)
}

/// Export models as deterministic pretty JSON without resolving key material.
///
/// # Errors
///
/// Returns an error if the typed configuration cannot be serialized as JSON.
pub fn export_models(models: &PiModelsConfig) -> Result<String, PiModelsError> {
    serde_json::to_string_pretty(models).map_err(PiModelsError::from)
}

fn strip_json_comments(input: &[u8]) -> Result<Vec<u8>, PiModelsError> {
    #[derive(Clone, Copy)]
    enum State {
        Normal,
        String,
        LineComment,
        BlockComment,
    }

    let mut state = State::Normal;
    let mut escaped = false;
    let mut output = Vec::with_capacity(input.len());
    let mut index = 0;

    while index < input.len() {
        let byte = input[index];
        match state {
            State::Normal => {
                if byte == b'"' {
                    state = State::String;
                    output.push(byte);
                } else if byte == b'/' && input.get(index + 1) == Some(&b'/') {
                    state = State::LineComment;
                    output.extend_from_slice(b"  ");
                    index += 1;
                } else if byte == b'/' && input.get(index + 1) == Some(&b'*') {
                    state = State::BlockComment;
                    output.extend_from_slice(b"  ");
                    index += 1;
                } else {
                    output.push(byte);
                }
            }
            State::String => {
                output.push(byte);
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    state = State::Normal;
                }
            }
            State::LineComment => {
                if byte == b'\n' || byte == b'\r' {
                    output.push(byte);
                    state = State::Normal;
                } else {
                    output.push(b' ');
                }
            }
            State::BlockComment => {
                if byte == b'*' && input.get(index + 1) == Some(&b'/') {
                    output.extend_from_slice(b"  ");
                    index += 1;
                    state = State::Normal;
                } else if byte == b'\n' || byte == b'\r' {
                    output.push(byte);
                } else {
                    output.push(b' ');
                }
            }
        }
        index += 1;
    }

    if matches!(state, State::BlockComment) {
        return Err(PiModelsError::UnterminatedBlockComment);
    }
    Ok(output)
}
