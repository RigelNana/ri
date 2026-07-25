//! Typed failures and context-window overflow classification.

use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::message::{AssistantMessage, StopReason};

/// Stable error categories exposed by the crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// Dynamic model source failed.
    ModelSource,
    /// Model metadata was invalid.
    ModelValidation,
    /// Provider is missing or invalid.
    Provider,
    /// Stream or wire protocol failed.
    Stream,
    /// API-key or credential-store failure.
    Auth,
    /// OAuth refresh or conversion failure.
    Oauth,
    /// HTTP request failed.
    Http,
    /// Provider rejected a request.
    ProviderResponse,
    /// JSON/schema/tool conversion failed.
    Validation,
    /// Caller cancelled work.
    Aborted,
}

/// Error type used by provider, auth, transport, and wire layers.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AiError {
    /// Dynamic model source failure.
    #[error("model source error: {0}")]
    ModelSource(String),
    /// Invalid model metadata.
    #[error("model validation error: {0}")]
    ModelValidation(String),
    /// Unknown or invalid provider.
    #[error("provider error: {0}")]
    Provider(String),
    /// Stream ended incorrectly or an event was invalid.
    #[error("stream error: {0}")]
    Stream(String),
    /// Credential store or API-key resolution failure.
    #[error("authentication error: {0}")]
    Auth(String),
    /// OAuth refresh/derivation failure.
    #[error("OAuth error: {0}")]
    Oauth(String),
    /// Network request could not be completed.
    #[error("HTTP transport error: {0}")]
    Http(String),
    /// Non-success provider response.
    #[error("provider response {status}: {message}")]
    ProviderResponse {
        /// HTTP status.
        status: u16,
        /// Human-readable response error.
        message: String,
        /// Bounded raw or JSON response body.
        body: Option<String>,
    },
    /// Tool/schema/protocol validation failure.
    #[error("validation error: {0}")]
    Validation(String),
    /// Explicit cancellation.
    #[error("request aborted")]
    Aborted,
}

impl AiError {
    /// Stable category for programmatic handling.
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::ModelSource(_) => ErrorCode::ModelSource,
            Self::ModelValidation(_) => ErrorCode::ModelValidation,
            Self::Provider(_) => ErrorCode::Provider,
            Self::Stream(_) => ErrorCode::Stream,
            Self::Auth(_) => ErrorCode::Auth,
            Self::Oauth(_) => ErrorCode::Oauth,
            Self::Http(_) => ErrorCode::Http,
            Self::ProviderResponse { .. } => ErrorCode::ProviderResponse,
            Self::Validation(_) => ErrorCode::Validation,
            Self::Aborted => ErrorCode::Aborted,
        }
    }

    /// Whether retrying the same request may succeed.
    pub const fn is_retryable(&self) -> bool {
        match self {
            Self::Http(_) => true,
            Self::ProviderResponse { status, .. } => {
                *status == 408 || *status == 409 || *status == 429 || *status >= 500
            }
            Self::Aborted
            | Self::Auth(_)
            | Self::Oauth(_)
            | Self::Validation(_)
            | Self::ModelValidation(_)
            | Self::Provider(_)
            | Self::Stream(_)
            | Self::ModelSource(_) => false,
        }
    }
}

/// How a context overflow was identified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OverflowClassification {
    /// Error text matched a known provider context-limit signature.
    ExplicitError,
    /// A successful response reported prompt usage beyond the model window.
    UsageExceeded,
    /// A zero-output length stop filled at least 99% of the context window.
    TruncatedAtWindow,
}

fn overflow_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        [
            r"(?i)prompt is too long",
            r"(?i)request_too_large",
            r"(?i)input is too long for requested model",
            r"(?i)exceeds the context window",
            r"(?i)exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))",
            r"(?i)input token count.*exceeds the maximum",
            r"(?i)maximum prompt length is \d+",
            r"(?i)reduce the length of the messages",
            r"(?i)maximum context length is \d+ tokens",
            r"(?i)exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?",
            r"(?i)input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
            r"(?i)exceeds the limit of \d+",
            r"(?i)exceeds the available context size",
            r"(?i)greater than the context length",
            r"(?i)context window exceeds limit",
            r"(?i)exceeded model token limit",
            r"(?i)too large for model with \d+ maximum context length",
            r"(?i)prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?",
            r"(?i)model_context_window_exceeded",
            r"(?i)prompt too long; exceeded (?:max )?context length",
            r"(?i)range of input length should be",
            r"(?i)context[_ ]length[_ ]exceeded",
            r"(?i)too many tokens",
            r"(?i)token limit exceeded",
            r"(?i)^4(?:00|13)\s*(?:status code)?\s*\(no body\)",
        ]
        .into_iter()
        .filter_map(|pattern| Regex::new(pattern).ok())
        .collect()
    })
}

fn non_overflow_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        [
            r"(?i)^(Throttling error|Service unavailable):",
            r"(?i)rate limit",
            r"(?i)too many requests",
        ]
        .into_iter()
        .filter_map(|pattern| Regex::new(pattern).ok())
        .collect()
    })
}

/// Classifies a provider response as a context-window overflow.
///
/// Explicit error signatures take precedence. The usage-based fallbacks match
/// Pi's handling of providers that silently accept or truncate oversized
/// prompts.
pub fn classify_context_overflow(
    message: &AssistantMessage,
    context_window: Option<u64>,
) -> Option<OverflowClassification> {
    if message.stop_reason == StopReason::Error
        && let Some(error) = message.error_message.as_deref()
        && !non_overflow_patterns()
            .iter()
            .any(|pattern| pattern.is_match(error))
        && overflow_patterns()
            .iter()
            .any(|pattern| pattern.is_match(error))
    {
        return Some(OverflowClassification::ExplicitError);
    }

    let window = context_window?;
    let prompt_tokens = message.usage.input.saturating_add(message.usage.cache_read);
    if message.stop_reason == StopReason::Stop && prompt_tokens > window {
        return Some(OverflowClassification::UsageExceeded);
    }
    if message.stop_reason == StopReason::Length
        && message.usage.output == 0
        && u128::from(prompt_tokens) * 100 >= u128::from(window) * 99
    {
        return Some(OverflowClassification::TruncatedAtWindow);
    }
    None
}

/// Convenience boolean for overflow classification.
pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    classify_context_overflow(message, context_window).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{AssistantMessage, Usage};

    fn message(reason: StopReason, error: Option<&str>, usage: Usage) -> AssistantMessage {
        let mut result = AssistantMessage::empty("openai-completions", "test", "m");
        result.stop_reason = reason;
        result.error_message = error.map(str::to_owned);
        result.usage = usage;
        result
    }

    #[test]
    fn catches_provider_signatures() {
        for text in [
            "prompt is too long: 213462 tokens > 200000 maximum",
            "Input length (265330) exceeds model's maximum context length (262144).",
            "Prompt has 5,958,968 tokens, but the configured context size is 256,000 tokens",
            "400 `prompt too long; exceeded max context length by 100918 tokens`",
        ] {
            assert_eq!(
                classify_context_overflow(
                    &message(StopReason::Error, Some(text), Usage::default()),
                    Some(200_000)
                ),
                Some(OverflowClassification::ExplicitError)
            );
        }
    }

    #[test]
    fn excludes_throttling_false_positives() {
        for text in [
            "Throttling error: Too many tokens, please wait before trying again.",
            "Rate limit exceeded; too many tokens per minute",
            "Too many requests. Please slow down.",
        ] {
            assert_eq!(
                classify_context_overflow(
                    &message(StopReason::Error, Some(text), Usage::default()),
                    Some(200_000)
                ),
                None
            );
        }
    }

    #[test]
    fn catches_silent_and_truncated_overflow() {
        assert_eq!(
            classify_context_overflow(
                &message(StopReason::Stop, None, Usage::from_parts(100_001, 1, 0, 0)),
                Some(100_000)
            ),
            Some(OverflowClassification::UsageExceeded)
        );
        assert_eq!(
            classify_context_overflow(
                &message(
                    StopReason::Length,
                    None,
                    Usage::from_parts(58, 0, 1_048_512, 0)
                ),
                Some(1_048_576)
            ),
            Some(OverflowClassification::TruncatedAtWindow)
        );
    }
}
