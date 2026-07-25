//! Model metadata, pricing, reasoning capability, and cache semantics.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use ri_protocol_core::ThinkingLevel;

use crate::message::{Usage, UsageCost};

const TOKENS_PER_MILLION: f64 = 1_000_000.0;

fn token_count_as_f64(tokens: u64) -> f64 {
    let bytes = tokens.to_be_bytes();
    let high = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let low = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    f64::from(high) * (f64::from(u32::MAX) + 1.0) + f64::from(low)
}

/// Calculates one token class from its per-million rate.
pub(crate) fn cost_for_tokens(rate: f64, tokens: u64) -> f64 {
    rate * token_count_as_f64(tokens) / TOKENS_PER_MILLION
}

/// Prompt-cache retention preference.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheRetention {
    /// Disable provider prompt-cache writes and affinity keys where supported.
    None,
    /// Provider-default short-lived caching.
    #[default]
    Short,
    /// Long-lived caching (typically Anthropic 1h or `OpenAI` 24h).
    Long,
}

/// Input modalities accepted by a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelInput {
    /// UTF-8 text.
    Text,
    /// Base64 image data.
    Image,
}

/// Prices in US dollars per million tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostRates {
    /// Uncached input.
    pub input: f64,
    /// Output.
    pub output: f64,
    /// Prompt-cache reads.
    pub cache_read: f64,
    /// Prompt-cache writes.
    pub cache_write: f64,
}

/// Request-wide price tier.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCostTier {
    /// The tier applies when total prompt usage is strictly greater than this value.
    pub input_tokens_above: u64,
    /// Tier prices.
    #[serde(flatten)]
    pub rates: ModelCostRates,
}

/// Base prices and optional request-wide tiers.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelCost {
    /// Base prices.
    #[serde(flatten)]
    pub rates: ModelCostRates,
    /// Highest matching input threshold applies to the whole request.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tiers: Vec<ModelCostTier>,
}

impl From<ModelCostRates> for ModelCost {
    fn from(rates: ModelCostRates) -> Self {
        Self {
            rates,
            tiers: Vec::new(),
        }
    }
}

/// Header layout used for request/session affinity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SessionAffinityFormat {
    /// `session_id`, `x-client-request-id`, and API-specific affinity.
    Openai,
    /// OpenAI-compatible headers without `session_id`.
    OpenaiNosession,
    /// `OpenRouter`'s `x-session-id`.
    Openrouter,
}

/// Maximum-output field accepted by an OpenAI-compatible endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxTokensField {
    /// Modern reasoning-model field.
    #[serde(rename = "max_completion_tokens")]
    MaxCompletionTokens,
    /// Legacy Chat Completions field.
    #[serde(rename = "max_tokens")]
    MaxTokens,
}

impl MaxTokensField {
    /// Returns the exact JSON field name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MaxCompletionTokens => "max_completion_tokens",
            Self::MaxTokens => "max_tokens",
        }
    }
}

/// Reasoning parameter convention used by OpenAI-compatible endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ThinkingFormat {
    /// `OpenAI` `reasoning_effort`.
    Openai,
    /// `OpenRouter` `reasoning.effort`.
    Openrouter,
    /// `DeepSeek` `thinking.type`.
    Deepseek,
    /// Together `reasoning.enabled`.
    Together,
    /// Z.AI `thinking.type`.
    Zai,
    /// Qwen `enable_thinking`.
    Qwen,
    /// Generic chat-template kwargs.
    ChatTemplate,
    /// Qwen chat-template kwargs.
    QwenChatTemplate,
    /// Top-level string-valued thinking mode.
    StringThinking,
    /// Ant Ling reasoning object.
    AntLing,
}

/// Prompt-cache marker convention for OpenAI-compatible APIs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlFormat {
    /// Anthropic-style `cache_control` markers.
    Anthropic,
}

/// Provider-specific deferred-tool serialization convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeferredToolsMode {
    /// Kimi deferred-tool payloads.
    Kimi,
}

/// Compatibility switches for custom or proxy-hosted models.
///
/// Fields are optional so API adapters can select protocol defaults when an
/// override is absent.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelCompatibility {
    /// Supports developer-role instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_developer_role: Option<bool>,
    /// Supports strict JSON-schema tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_mode: Option<bool>,
    /// Supports `OpenAI` custom grammar tools.
    #[serde(
        rename = "supportsOpenAIGrammarTools",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub supports_openai_grammar_tools: Option<bool>,
    /// Supports `reasoning_effort` or the provider's mapped equivalent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_reasoning_effort: Option<bool>,
    /// Supports provider long-cache retention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_long_cache_retention: Option<bool>,
    /// Supports deferred tool search/reference loading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_tool_search: Option<bool>,
    /// Supports explicit prompt-cache mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_explicit_prompt_cache_mode: Option<bool>,
    /// Supports per-tool Anthropic cache control.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_cache_control_on_tools: Option<bool>,
    /// Supports Anthropic eager input streaming.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_eager_tool_input_streaming: Option<bool>,
    /// Supports Anthropic strict tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_strict_tools: Option<bool>,
    /// Supports Anthropic tool references.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_tool_references: Option<bool>,
    /// Forces Anthropic adaptive thinking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force_adaptive_thinking: Option<bool>,
    /// Allows replaying empty Anthropic thinking signatures.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_empty_signature: Option<bool>,
    /// Supports the temperature request parameter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_temperature: Option<bool>,
    /// Send session affinity headers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub send_session_affinity_headers: Option<bool>,
    /// Session header convention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_affinity_format: Option<SessionAffinityFormat>,
    /// `OpenAI` completions accepts `store`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_store: Option<bool>,
    /// `OpenAI` completions streams usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supports_usage_in_streaming: Option<bool>,
    /// Tool results require a name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_tool_result_name: Option<bool>,
    /// Insert an assistant bridge after tool results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_assistant_after_tool_result: Option<bool>,
    /// Replay thinking as plain assistant text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_thinking_as_text: Option<bool>,
    /// Replayed assistant messages require a reasoning-content field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires_reasoning_content_on_assistant_messages: Option<bool>,
    /// Provider reasoning parameter convention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_format: Option<ThinkingFormat>,
    /// Z.AI supports top-level streaming tool-call deltas.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zai_tool_stream: Option<bool>,
    /// Maximum token field accepted by OpenAI-compatible APIs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_field: Option<MaxTokensField>,
    /// Prompt-cache marker convention.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control_format: Option<CacheControlFormat>,
    /// Provider-specific deferred-tool serialization mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deferred_tools_mode: Option<DeferredToolsMode>,
}

/// Provider-neutral model metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    /// Provider model identifier.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Wire API identifier.
    pub api: String,
    /// Provider identifier.
    pub provider: String,
    /// API base URL.
    pub base_url: String,
    /// Whether the model can emit reasoning.
    pub reasoning: bool,
    /// Optional model/provider mapping for reasoning levels. `None` values mark
    /// levels as unsupported; missing entries use protocol defaults.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub thinking_level_map: BTreeMap<ThinkingLevel, Option<String>>,
    /// Accepted modalities.
    pub input: Vec<ModelInput>,
    /// Pricing.
    pub cost: ModelCost,
    /// Context window in tokens.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_tokens: u64,
    /// Static provider/model headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// API compatibility overrides.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compat: Option<ModelCompatibility>,
}

impl Model {
    /// Creates model metadata with text input, zero pricing, and no reasoning.
    pub fn new(
        provider: impl Into<String>,
        id: impl Into<String>,
        api: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let id = id.into();
        Self {
            name: id.clone(),
            id,
            api: api.into(),
            provider: provider.into(),
            base_url: base_url.into(),
            reasoning: false,
            thinking_level_map: BTreeMap::new(),
            input: vec![ModelInput::Text],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 4_096,
            headers: BTreeMap::new(),
            compat: None,
        }
    }

    /// Whether this model accepts images.
    pub fn supports_images(&self) -> bool {
        self.input.contains(&ModelInput::Image)
    }

    /// Whether this model uses a specific wire API.
    pub fn has_api(&self, api: &str) -> bool {
        self.api == api
    }
}

/// Image-generation model metadata.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageModel {
    /// Model id.
    pub id: String,
    /// Display name.
    pub name: String,
    /// Image wire API id.
    pub api: String,
    /// Provider id.
    pub provider: String,
    /// API base URL.
    pub base_url: String,
    /// Accepted prompt modalities.
    pub input: Vec<ModelInput>,
    /// Emitted modalities.
    pub output: Vec<ModelInput>,
    /// Token pricing.
    pub cost: ModelCost,
    /// Static headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
}

/// Applies model pricing to usage and returns the calculated cost.
///
/// Input tier selection uses uncached input plus both cache classes. Anthropic
/// one-hour writes are charged at twice the selected base input rate.
pub fn calculate_cost(model: &Model, usage: &mut Usage) -> UsageCost {
    let prompt_tokens = usage.prompt_tokens();
    let mut rates = model.cost.rates;
    let mut matched_threshold = None;
    for tier in &model.cost.tiers {
        if prompt_tokens > tier.input_tokens_above
            && matched_threshold.is_none_or(|threshold| tier.input_tokens_above > threshold)
        {
            rates = tier.rates;
            matched_threshold = Some(tier.input_tokens_above);
        }
    }

    let long_write = usage.cache_write_1h.unwrap_or(0).min(usage.cache_write);
    let short_write = usage.cache_write.saturating_sub(long_write);
    let input = cost_for_tokens(rates.input, usage.input);
    let output = cost_for_tokens(rates.output, usage.output);
    let cache_read = cost_for_tokens(rates.cache_read, usage.cache_read);
    let cache_write = cost_for_tokens(rates.cache_write, short_write)
        + cost_for_tokens(rates.input * 2.0, long_write);
    let cost = UsageCost {
        input,
        output,
        cache_read,
        cache_write,
        total: input + output + cache_read + cache_write,
    };
    usage.cost = cost;
    cost
}

/// Returns reasoning levels supported by a model.
pub fn supported_thinking_levels(model: &Model) -> Vec<ThinkingLevel> {
    if !model.reasoning {
        return vec![ThinkingLevel::Off];
    }
    ThinkingLevel::ALL
        .into_iter()
        .filter(|level| {
            let mapped = model.thinking_level_map.get(level);
            if matches!(mapped, Some(None)) {
                return false;
            }
            if matches!(level, ThinkingLevel::Xhigh | ThinkingLevel::Max) {
                return mapped.is_some();
            }
            true
        })
        .collect()
}

/// Clamps a requested reasoning level using Pi's upward-first semantics.
///
/// If a level is unavailable, the nearest higher level wins. Only when there
/// is no higher level does the search move downward.
pub fn clamp_thinking_level(model: &Model, requested: ThinkingLevel) -> ThinkingLevel {
    let supported = supported_thinking_levels(model);
    let requested_index = ThinkingLevel::ALL
        .iter()
        .position(|candidate| *candidate == requested)
        .unwrap_or_default();
    ThinkingLevel::ALL[requested_index..]
        .iter()
        .chain(ThinkingLevel::ALL[..requested_index].iter().rev())
        .copied()
        .find(|candidate| supported.contains(candidate))
        .or_else(|| supported.first().copied())
        .unwrap_or(ThinkingLevel::Off)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        let tolerance = f64::EPSILON * expected.abs().max(1.0) * 8.0;
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn priced_model() -> Model {
        let mut model = Model::new(
            "openai",
            "gpt-tiered",
            "openai-responses",
            "https://example.test",
        );
        model.cost = ModelCost {
            rates: ModelCostRates {
                input: 5.0,
                output: 30.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
            tiers: vec![ModelCostTier {
                input_tokens_above: 272_000,
                rates: ModelCostRates {
                    input: 10.0,
                    output: 45.0,
                    cache_read: 1.0,
                    cache_write: 12.5,
                },
            }],
        };
        model
    }

    #[test]
    fn applies_highest_request_wide_tier() {
        let mut usage = Usage::from_parts(200_000, 100_000, 72_000, 1);
        let cost = calculate_cost(&priced_model(), &mut usage);
        assert_close(cost.input, 2.0);
        assert_close(cost.output, 4.5);
        assert_close(cost.cache_read, 0.072);
        assert_close(cost.cache_write, 0.000_012_5);
    }

    #[test]
    fn one_hour_cache_writes_cost_twice_input_rate() {
        let mut usage = Usage::from_parts(0, 0, 0, 10);
        usage.cache_write_1h = Some(4);
        let cost = calculate_cost(&priced_model(), &mut usage);
        let expected = (6.25 * 6.0 + 5.0 * 2.0 * 4.0) / 1_000_000.0;
        assert_close(cost.cache_write, expected);
    }

    #[test]
    fn extended_reasoning_requires_explicit_mapping() {
        let mut model = Model::new("p", "m", "a", "https://example.test");
        model.reasoning = true;
        model
            .thinking_level_map
            .insert(ThinkingLevel::Xhigh, Some("xhigh".into()));
        model.thinking_level_map.insert(ThinkingLevel::High, None);
        assert_eq!(
            supported_thinking_levels(&model),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::Xhigh,
            ]
        );
        assert_eq!(
            clamp_thinking_level(&model, ThinkingLevel::High),
            ThinkingLevel::Xhigh
        );
        assert_eq!(
            clamp_thinking_level(&model, ThinkingLevel::Max),
            ThinkingLevel::Xhigh
        );
    }

    #[test]
    fn non_reasoning_model_only_supports_off() {
        let model = Model::new("p", "m", "a", "https://example.test");
        assert_eq!(supported_thinking_levels(&model), vec![ThinkingLevel::Off]);
        assert_eq!(
            clamp_thinking_level(&model, ThinkingLevel::High),
            ThinkingLevel::Off
        );
    }
}
