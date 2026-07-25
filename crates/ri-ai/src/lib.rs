//! Provider-neutral AI messages, model metadata, authentication, streaming,
//! and wire-protocol adapters.
//!
//! The crate deliberately separates protocol conversion from I/O. Production
//! providers use [`transport::ReqwestTransport`], while tests can inject an
//! in-memory [`transport::HttpTransport`] without changing provider behavior.

pub mod auth;
pub mod catalog;
pub mod error;
pub mod handoff;
pub mod message;
pub mod model;
pub mod provider;
pub mod stream;
pub mod tool;
pub mod transport;
pub mod wire;

pub use auth::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResolutionOverrides, AuthResult, Credential,
    CredentialInfo, CredentialKind, CredentialStore, EnvApiKeyAuth, InMemoryCredentialStore,
    ModelAuth, OAuthAuth, OAuthCredential, ProviderAuth, SystemAuthContext, resolve_provider_auth,
};
pub use catalog::{builtin_adapters, builtin_image_models, builtin_models, builtin_providers};
pub use error::{AiError, ErrorCode, OverflowClassification, classify_context_overflow};
pub use handoff::{
    NON_VISION_TOOL_IMAGE_PLACEHOLDER, NON_VISION_USER_IMAGE_PLACEHOLDER, transform_messages,
};
pub use message::{
    AssistantImages, AssistantMessage, AssistantMessageEvent, ContentBlock, Context, ImageContent,
    ImagesContext, ImagesStopReason, InputContent, Message, StopReason, TextContent,
    ThinkingContent, Timestamp, ToolCall, ToolResultMessage, Usage, UsageCost, UserContent,
    UserMessage, now_millis,
};
pub use model::{
    CacheControlFormat, CacheRetention, DeferredToolsMode, ImageModel, MaxTokensField, Model,
    ModelCompatibility, ModelCost, ModelCostRates, ModelCostTier, ModelInput, ThinkingFormat,
    ThinkingLevel, calculate_cost, clamp_thinking_level, supported_thinking_levels,
};
pub use provider::{
    AuthCheck, Models, Provider, ProviderDescriptor, RefreshOptions, RefreshResult, StreamOptions,
};
pub use stream::{
    AssistantEventSender, AssistantEventStream, create_assistant_message_event_stream,
};
pub use tool::{
    ConstrainedSampling, DeferredTools, GrammarFormat, GrammarToolInputBuffer, GrammarVariants,
    JsonSchemaStrictness, Tool, ToolDescriptor, ToolValidationError, describe_tool,
    resolve_grammar_constraint, resolve_json_schema_strict, split_deferred_tools,
    validate_tool_arguments, validate_tool_call,
};
