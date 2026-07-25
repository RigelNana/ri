//! Typed native extensions, resources, settings, project trust, and packages.
//!
//! `ri-ext` deliberately presents narrow host-facing traits instead of
//! depending on high-churn harness/session APIs. JSON values are confined to
//! extension, provider, renderer, event-bus, and custom-settings boundaries.

pub mod atomic;
pub mod extension;
pub mod package;
pub mod resources;
pub mod settings;
pub mod source;
pub mod trust;

pub use extension::{
    ActionError, BeforeAgentStartEvent, BeforeAgentStartReduction, BeforeAgentStartResult,
    BuiltinShortcut, BusDeliveryError, BusEvent, BusHandler, BusHandlerError, BusSubscription,
    CommandHandler, CommandRegistration, CompactionReason, ContentPart, ContextActions,
    ContextBinding, ContextError, ContextEvent, ContextFactory, ContextMessage, CustomMessage,
    EventBus, EventHook, Extension, ExtensionContext, ExtensionDescriptor, ExtensionHost,
    ExtensionInitError, ExtensionLoadError, ExtensionMode, ExtensionRegistrar, ExtensionRunner,
    FlagKind, FlagRegistration, FlagValue, ForkPosition, GenerationClock, HandlerFailure,
    HookError, HookRegistration, InputEvent, InputResult, InputSource, MessageEndEvent,
    MessageRole, NativeTool, NoopContextActions, NotificationEvent, ProjectTrustDecision,
    ProjectTrustResult, ProviderHeaders, ProviderHeadersEvent, ProviderModel, ProviderRegistration,
    ProviderRequestEvent, Registries, RegistryError, RenderedContent, Renderer, ResolvedCommand,
    ResolvedShortcut, SessionBeforeEvent, SessionBeforeResult, SessionOverride, ShortcutHandler,
    ShortcutRegistration, StaleContextError, StreamingBehavior, ToolCallEvent, ToolCallReduction,
    ToolCallResult, ToolDescriptor, ToolOutput, ToolResultEvent, ToolResultPatch, ToolUsage,
};
pub use package::{
    ManifestResources, PackageError, PackageFilter, PackageIdentity, PackageLock, PackageLockEntry,
    PackageManager, PackageManagerOptions, PackageManifest, PackageResource, PackageResourceKind,
    PackageScope, PackageSnapshot, PackageSource, PackageSpec, ResolvedPackage,
    ResolvedSourceMetadata,
};
pub use resources::{
    ContextResource, FrontmatterError, PromptInput, PromptLoadResult, PromptTemplate,
    ResourceLoader, ResourceLoaderOptions, ResourcePath, ResourceSnapshot, Skill, SkillLoadResult,
    expand_prompt_template, expand_skill_command, format_skills_for_prompt, load_prompt_templates,
    load_resource_snapshot, load_skills, parse_command_args, parse_frontmatter, substitute_args,
};
pub use settings::{
    CompactionSettings, DefaultProjectTrust, FileSettingsStorage, MemorySettingsStorage,
    ProviderRetrySettings, QueueMode, ResolvedCompactionSettings, ResolvedRetrySettings,
    ResourceSettings, RetrySettings, Settings, SettingsError, SettingsManager,
    SettingsManagerError, SettingsScope, SettingsStorage, SettingsStorageError,
};
pub use source::{
    Collision, Diagnostic, DiagnosticLevel, PackageTransport, ResourceKind, SourceInfo, SourceKind,
    SourceOrigin, SourceScope,
};
pub use trust::{
    FileTrustStore, MemoryTrustStore, TrustEntry, TrustPrompt, TrustPromptChoice,
    TrustResolveError, TrustResolveOptions, TrustResolver, TrustStore, TrustStoreError,
    TrustUpdate, has_trust_requiring_project_resources, trust_prompt_choices,
};
