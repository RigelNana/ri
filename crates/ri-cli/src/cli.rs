//! Command-line grammar.

use std::ffi::{OsStr, OsString};
use std::path::PathBuf;

use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::error::{CliError, Result};

/// Native `ri` command-line arguments.
///
/// Boolean fields intentionally mirror independent command-line switches.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Parser)]
#[command(
    name = "ri",
    version,
    about = "Native coding agent",
    long_about = "A native coding agent with interactive, print, JSON event, and RPC modes.",
    subcommand_precedence_over_arg = true
)]
pub struct Cli {
    /// Explicit frontend mode.
    #[arg(long, value_enum)]
    pub mode: Option<ModeOption>,

    /// Process the prompt and print the final assistant text.
    #[arg(short = 'p', long)]
    pub print: bool,

    /// Provider identifier.
    #[arg(long)]
    pub provider: Option<String>,

    /// Model id or provider/model selector, optionally suffixed with `:<thinking>`.
    #[arg(long)]
    pub model: Option<String>,

    /// Ordered, comma-delimited model selectors; the first is selected at startup.
    #[arg(long, value_delimiter = ',')]
    pub models: Vec<String>,

    /// Persist the selected model as the user default.
    #[arg(long, requires = "model")]
    pub set_default: bool,

    /// One-run API key override.
    #[arg(long, hide_env_values = true)]
    pub api_key: Option<String>,

    /// Reasoning level.
    #[arg(long, value_enum)]
    pub thinking: Option<ThinkingOption>,

    /// Replace the default system prompt.
    #[arg(long)]
    pub system_prompt: Option<String>,

    /// Append text to the system prompt; may be repeated.
    #[arg(long, action = clap::ArgAction::Append)]
    pub append_system_prompt: Vec<String>,

    /// Continue the newest session for the effective working directory.
    #[arg(
        short = 'c',
        long = "continue",
        conflicts_with_all = ["resume", "session", "session_id", "fork", "no_session"]
    )]
    pub continue_session: bool,

    /// Select a previous session interactively.
    #[arg(
        short = 'r',
        long,
        conflicts_with_all = ["continue_session", "session", "session_id", "fork", "no_session"]
    )]
    pub resume: bool,

    /// Open a session by id, id prefix, or JSONL path.
    #[arg(
        long,
        value_name = "ID_OR_PATH",
        conflicts_with_all = ["continue_session", "resume", "session_id", "fork", "no_session"]
    )]
    pub session: Option<String>,

    /// Open or create an exact project-local session id.
    #[arg(
        long,
        conflicts_with_all = ["continue_session", "resume", "session", "fork", "no_session"]
    )]
    pub session_id: Option<String>,

    /// Fork a session by id, id prefix, or JSONL path before starting.
    #[arg(
        long,
        value_name = "ID_OR_PATH",
        conflicts_with_all = ["continue_session", "resume", "session", "session_id", "no_session"]
    )]
    pub fork: Option<String>,

    /// Override the session repository directory.
    #[arg(long, value_name = "DIR", env = "RI_SESSION_DIR")]
    pub session_dir: Option<PathBuf>,

    /// Keep the session in memory.
    #[arg(
        long,
        conflicts_with_all = ["continue_session", "resume", "session", "session_id", "fork"]
    )]
    pub no_session: bool,

    /// Set the session display name.
    #[arg(short = 'n', long)]
    pub name: Option<String>,

    /// Comma-delimited tool allowlist.
    #[arg(short = 't', long, value_delimiter = ',', conflicts_with = "no_tools")]
    pub tools: Vec<String>,

    /// Comma-delimited tool denylist.
    #[arg(short = 'x', long, value_delimiter = ',')]
    pub exclude_tools: Vec<String>,

    /// Disable all tools.
    #[arg(long, conflicts_with = "tools")]
    pub no_tools: bool,

    /// Disable built-in tools while retaining extension tools.
    #[arg(long)]
    pub no_builtin_tools: bool,

    /// Load an explicit extension; may be repeated.
    #[arg(short = 'e', long = "extension", action = clap::ArgAction::Append)]
    pub extensions: Vec<PathBuf>,

    /// Load an explicit skill; may be repeated.
    #[arg(long = "skill", action = clap::ArgAction::Append)]
    pub skills: Vec<PathBuf>,

    /// Load an explicit prompt template; may be repeated.
    #[arg(long = "prompt-template", action = clap::ArgAction::Append)]
    pub prompt_templates: Vec<PathBuf>,

    /// Load an explicit terminal theme; may be repeated.
    #[arg(long = "theme", action = clap::ArgAction::Append)]
    pub themes: Vec<PathBuf>,

    /// Disable discovered extensions.
    #[arg(long)]
    pub no_extensions: bool,

    /// Disable discovered skills.
    #[arg(long)]
    pub no_skills: bool,

    /// Disable discovered prompt templates.
    #[arg(long)]
    pub no_prompt_templates: bool,

    /// Disable discovered terminal themes.
    #[arg(long)]
    pub no_themes: bool,

    /// Disable project context-file discovery.
    #[arg(long)]
    pub no_context_files: bool,

    /// Trust project-local resources for this invocation.
    #[arg(short = 'a', long, global = true, conflicts_with = "no_approve")]
    pub approve: bool,

    /// Ignore project-local resources for this invocation.
    #[arg(long, global = true, conflicts_with = "approve")]
    pub no_approve: bool,

    /// List available models and exit, optionally filtering them.
    #[arg(
        long,
        value_name = "SEARCH",
        num_args = 0..=1,
        default_missing_value = ""
    )]
    pub list_models: Option<String>,

    /// List registered providers and exit.
    #[arg(long)]
    pub list_providers: bool,

    /// Disable startup network refreshes.
    #[arg(long, env = "RI_OFFLINE")]
    pub offline: bool,

    /// Emit startup diagnostics on stderr.
    #[arg(long)]
    pub verbose: bool,

    /// Administrative command.
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Initial and follow-up messages. Prefix a path with `@` to attach it.
    #[arg(value_name = "MESSAGE")]
    pub messages: Vec<String>,
}

impl Cli {
    /// Parse arguments after expanding Pi-compatible multi-character aliases.
    ///
    /// # Errors
    ///
    /// Returns Clap's structured parse error for invalid arguments.
    pub fn try_parse_compatible_from<I, T>(arguments: I) -> std::result::Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        let normalized = arguments
            .into_iter()
            .map(Into::into)
            .map(normalize_alias)
            .collect::<Vec<_>>();
        Self::try_parse_from(normalized)
    }

    /// Validate relationships that span Clap argument groups and run modes.
    ///
    /// # Errors
    ///
    /// Returns an argument error when options that cannot be honored together
    /// are combined.
    pub fn validate(&self) -> Result<()> {
        if let Some(provider) = self.provider.as_deref()
            && let Some((selector_provider, _)) = self
                .model
                .as_deref()
                .or_else(|| self.models.first().map(String::as_str))
                .and_then(|selector| selector.split_once('/'))
            && provider != selector_provider
        {
            return Err(CliError::InvalidArguments(format!(
                "--provider `{provider}` conflicts with model selector provider `{selector_provider}`"
            )));
        }
        if self.command.is_some()
            && (self.print
                || self.mode.is_some()
                || !self.messages.is_empty()
                || self.continue_session
                || self.resume
                || self.session.is_some()
                || self.session_id.is_some()
                || self.fork.is_some()
                || self.set_default)
        {
            return Err(CliError::InvalidArguments(
                "administrative commands cannot be combined with run-mode, prompt, or session-selection arguments"
                    .to_owned(),
            ));
        }
        if self.list_models.is_some() && self.list_providers {
            return Err(CliError::InvalidArguments(
                "--list-models and --list-providers are mutually exclusive".to_owned(),
            ));
        }
        if (self.list_models.is_some() || self.list_providers)
            && (self.command.is_some()
                || self.print
                || self.mode.is_some()
                || !self.messages.is_empty()
                || self.continue_session
                || self.resume
                || self.session.is_some()
                || self.session_id.is_some()
                || self.fork.is_some()
                || self.set_default)
        {
            return Err(CliError::InvalidArguments(
                "model/provider listing cannot be combined with commands, run-mode, prompt, or session-selection arguments"
                    .to_owned(),
            ));
        }
        if self.no_tools && self.no_builtin_tools {
            return Err(CliError::InvalidArguments(
                "--no-tools already disables all built-in tools".to_owned(),
            ));
        }
        if self
            .name
            .as_deref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(CliError::InvalidArguments(
                "session name cannot be empty".to_owned(),
            ));
        }
        Ok(())
    }

    /// Split `@path` inputs from textual messages without reading the files.
    pub fn inputs(&self) -> InputArguments {
        let mut files = Vec::new();
        let mut messages = Vec::new();
        for argument in &self.messages {
            match argument.strip_prefix('@') {
                Some(path) if !path.is_empty() => files.push(PathBuf::from(path)),
                _ => messages.push(argument.clone()),
            }
        }
        InputArguments { files, messages }
    }

    /// True when this invocation only prints runtime metadata.
    pub const fn is_metadata_request(&self) -> bool {
        self.command.is_some() || self.list_models.is_some() || self.list_providers
    }

    /// One-run project trust override.
    pub const fn project_trust_override(&self) -> Option<bool> {
        if self.approve {
            Some(true)
        } else if self.no_approve {
            Some(false)
        } else {
            None
        }
    }
}

fn normalize_alias(argument: OsString) -> OsString {
    let replacement = match argument.as_os_str() {
        value if value == OsStr::new("-xt") => Some("--exclude-tools"),
        value if value == OsStr::new("-nt") => Some("--no-tools"),
        value if value == OsStr::new("-nbt") => Some("--no-builtin-tools"),
        value if value == OsStr::new("-ne") => Some("--no-extensions"),
        value if value == OsStr::new("-ns") => Some("--no-skills"),
        value if value == OsStr::new("-np") => Some("--no-prompt-templates"),
        value if value == OsStr::new("-nc") => Some("--no-context-files"),
        value if value == OsStr::new("-na") => Some("--no-approve"),
        _ => None,
    };
    replacement.map_or(argument, OsString::from)
}

/// Prompt text and file attachments after lexical CLI splitting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputArguments {
    /// Explicit `@path` values.
    pub files: Vec<PathBuf>,
    /// Plain message arguments.
    pub messages: Vec<String>,
}

/// Explicit frontend mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModeOption {
    /// Full-screen terminal UI.
    Interactive,
    /// Final assistant text only.
    Text,
    /// Runtime events as JSON lines.
    Json,
    /// Strict request/response JSONL protocol.
    Rpc,
}

/// Provider-neutral reasoning level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingOption {
    /// Disable reasoning.
    Off,
    /// Minimum reasoning.
    Minimal,
    /// Low reasoning.
    Low,
    /// Medium reasoning.
    Medium,
    /// High reasoning.
    High,
    /// Extended high reasoning.
    Xhigh,
    /// Provider maximum.
    Max,
}

impl ThinkingOption {
    /// Stable runtime-facing spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

/// Administrative command.
#[derive(Debug, Clone, Subcommand)]
pub enum Command {
    /// Inspect providers or provider authentication.
    #[command(alias = "providers")]
    Provider {
        /// Provider operation.
        #[command(subcommand)]
        command: ProviderCommand,
    },

    /// Inspect available models.
    #[command(alias = "models")]
    Model {
        /// Model operation.
        #[command(subcommand)]
        command: ModelCommand,
    },

    /// Manage durable sessions.
    #[command(alias = "sessions")]
    Session {
        /// Session operation.
        #[command(subcommand)]
        command: SessionCommand,
    },

    /// Inspect and configure loaded resources.
    #[command(alias = "resources")]
    Resource {
        /// Resource operation.
        #[command(subcommand)]
        command: ResourceCommand,
    },

    /// Manage extension packages.
    #[command(alias = "packages")]
    Package {
        /// Package operation.
        #[command(subcommand)]
        command: PackageCommand,
    },

    /// Store provider credentials.
    Login(LoginArgs),

    /// Remove stored provider credentials.
    Logout(LogoutArgs),

    /// Pi-compatible package install shortcut.
    Install(PackageInstallArgs),

    /// Pi-compatible package removal shortcut.
    Remove(PackageRemoveArgs),

    /// Alias for `remove`.
    Uninstall(PackageRemoveArgs),

    /// Pi-compatible package update shortcut.
    Update(PackageUpdateArgs),

    /// Pi-compatible package listing shortcut.
    List(PackageListArgs),

    /// Show effective resource configuration.
    Config(ResourceListArgs),
}

/// Provider operations.
#[derive(Debug, Clone, Subcommand)]
pub enum ProviderCommand {
    /// List registered providers.
    List,
    /// Store credentials for a provider.
    Login(LoginArgs),
    /// Remove credentials for a provider.
    Logout(LogoutArgs),
}

/// Model operations.
#[derive(Debug, Clone, Subcommand)]
pub enum ModelCommand {
    /// List models, optionally filtered by provider or text.
    List(ModelListArgs),
}

/// Model-list filters.
#[derive(Debug, Clone, ClapArgs)]
pub struct ModelListArgs {
    /// Restrict models to one provider.
    #[arg(long)]
    pub provider: Option<String>,
    /// Case-insensitive text filter.
    pub search: Option<String>,
    /// Include models whose credentials are not currently available.
    #[arg(long)]
    pub all: bool,
    /// Emit a JSON array instead of aligned text.
    #[arg(long)]
    pub json: bool,
}

/// Credential login.
#[derive(Debug, Clone, ClapArgs)]
pub struct LoginArgs {
    /// Provider identifier.
    pub provider: String,
    /// API key to store. Omit to use an SDK-supported interactive OAuth flow.
    #[arg(long, hide_env_values = true, conflicts_with = "api_key_stdin")]
    pub api_key: Option<String>,
    /// Read an API key from stdin without echoing it in process arguments.
    #[arg(long, conflicts_with = "api_key")]
    pub api_key_stdin: bool,
}

/// Credential removal.
#[derive(Debug, Clone, ClapArgs)]
pub struct LogoutArgs {
    /// Provider identifier.
    pub provider: String,
}

/// Durable session operations.
#[derive(Debug, Clone, Subcommand)]
pub enum SessionCommand {
    /// List sessions newest first.
    List(SessionListArgs),
    /// Resolve and inspect a session.
    Open(SessionOpenArgs),
    /// Copy a session or one branch into a new session.
    Fork(SessionForkArgs),
    /// Export a session.
    Export(SessionExportArgs),
    /// Import a session file.
    Import(SessionImportArgs),
}

/// Session-list filters.
#[derive(Debug, Clone, ClapArgs)]
pub struct SessionListArgs {
    /// Include sessions from every working directory.
    #[arg(long)]
    pub all: bool,
    /// Override the working-directory filter.
    #[arg(long, value_name = "DIR")]
    pub cwd: Option<PathBuf>,
    /// Maximum number of sessions to print.
    #[arg(long)]
    pub limit: Option<usize>,
    /// Emit JSON instead of aligned text.
    #[arg(long)]
    pub json: bool,
}

/// Session lookup.
#[derive(Debug, Clone, ClapArgs)]
pub struct SessionOpenArgs {
    /// Session id, unambiguous id prefix, or path.
    pub target: String,
    /// Emit the complete append-only tree.
    #[arg(long)]
    pub tree: bool,
    /// Emit JSON instead of a human summary.
    #[arg(long)]
    pub json: bool,
}

/// Session fork.
#[derive(Debug, Clone, ClapArgs)]
pub struct SessionForkArgs {
    /// Source session id, unambiguous id prefix, or path.
    pub source: String,
    /// Fork at this entry. Without it, copy the active branch.
    #[arg(long)]
    pub entry: Option<String>,
    /// Include `--entry` itself instead of forking before a user message.
    #[arg(long, requires = "entry")]
    pub at: bool,
    /// Caller-selected destination id.
    #[arg(long)]
    pub id: Option<String>,
    /// Destination working directory.
    #[arg(long, value_name = "DIR")]
    pub cwd: Option<PathBuf>,
}

/// Session export.
#[derive(Debug, Clone, ClapArgs)]
pub struct SessionExportArgs {
    /// Session id, id prefix, or path.
    pub source: String,
    /// Output file. Defaults to stdout.
    pub output: Option<PathBuf>,
    /// Export wire format.
    #[arg(long, value_enum, default_value_t = SessionFormat::Native)]
    pub format: SessionFormat,
}

/// Session import.
#[derive(Debug, Clone, ClapArgs)]
pub struct SessionImportArgs {
    /// Native or Pi-compatible JSONL file.
    pub input: PathBuf,
    /// Input wire format.
    #[arg(long, value_enum, default_value_t = SessionFormat::Native)]
    pub format: SessionFormat,
    /// Working-directory override.
    #[arg(long, value_name = "DIR")]
    pub cwd: Option<PathBuf>,
    /// Caller-selected destination id.
    #[arg(long)]
    pub id: Option<String>,
}

/// Supported session interchange formats.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum SessionFormat {
    /// Ri's native typed JSONL.
    #[default]
    Native,
    /// Pi coding-agent JSONL v1-v3.
    Pi,
}

/// Resource operations.
#[derive(Debug, Clone, Subcommand)]
pub enum ResourceCommand {
    /// List discovered resources.
    List(ResourceListArgs),
    /// Enable a resource in settings.
    Enable(ResourceMutationArgs),
    /// Disable a resource in settings.
    Disable(ResourceMutationArgs),
    /// Reload resources from disk.
    Reload,
}

/// Resource-list filters.
#[derive(Debug, Clone, ClapArgs)]
pub struct ResourceListArgs {
    /// Resource category.
    #[arg(long, value_enum)]
    pub kind: Option<ResourceKind>,
    /// Settings scope.
    #[arg(long, value_enum, default_value_t = ResourceScope::All)]
    pub scope: ResourceScope,
    /// Emit JSON instead of aligned text.
    #[arg(long)]
    pub json: bool,
}

/// Resource settings mutation.
#[derive(Debug, Clone, ClapArgs)]
pub struct ResourceMutationArgs {
    /// Resource category.
    #[arg(value_enum)]
    pub kind: ResourceKind,
    /// Stable resource name or source.
    pub name: String,
    /// Mutate project-local rather than user settings.
    #[arg(long)]
    pub local: bool,
}

/// Resource category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// Native extension.
    Extension,
    /// Agent skill.
    Skill,
    /// Prompt template.
    Prompt,
    /// Terminal theme.
    Theme,
    /// Context file.
    Context,
    /// Registered tool.
    Tool,
}

/// Resource settings scope.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, ValueEnum)]
pub enum ResourceScope {
    /// User settings.
    Global,
    /// Project settings.
    Project,
    /// Effective union of both scopes.
    #[default]
    All,
}

/// Package operations.
#[derive(Debug, Clone, Subcommand)]
pub enum PackageCommand {
    /// Install a package and record its source.
    Install(PackageInstallArgs),
    /// Remove a package source.
    Remove(PackageRemoveArgs),
    /// Alias for `remove`.
    Uninstall(PackageRemoveArgs),
    /// Update one or all packages.
    Update(PackageUpdateArgs),
    /// List configured packages.
    List(PackageListArgs),
}

/// Package install.
#[derive(Debug, Clone, ClapArgs)]
pub struct PackageInstallArgs {
    /// Path, URL, git source, or package source understood by the SDK.
    pub source: String,
    /// Install into project-local settings.
    #[arg(short = 'l', long)]
    pub local: bool,
    /// Expected package SHA-256 (`sha256:<hex>` or bare hex).
    #[arg(long)]
    pub checksum: Option<String>,
}

/// Package removal.
#[derive(Debug, Clone, ClapArgs)]
pub struct PackageRemoveArgs {
    /// Configured package source.
    pub source: String,
    /// Remove from project-local settings.
    #[arg(short = 'l', long)]
    pub local: bool,
}

/// Package update.
#[derive(Debug, Clone, ClapArgs)]
pub struct PackageUpdateArgs {
    /// Update only this source.
    pub source: Option<String>,
    /// Update every configured package.
    #[arg(long, conflicts_with = "source")]
    pub all: bool,
    /// Refresh even when the recorded revision appears current.
    #[arg(long)]
    pub force: bool,
}

/// Package listing.
#[derive(Debug, Clone, Default, ClapArgs)]
pub struct PackageListArgs {
    /// Emit JSON instead of aligned text.
    #[arg(long)]
    pub json: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(arguments: &[&str]) -> Cli {
        Cli::try_parse_from(arguments).expect("arguments should parse")
    }

    #[test]
    fn parses_print_prompt_and_model_options() {
        let cli = parse(&[
            "ri",
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet",
            "--thinking",
            "high",
            "-p",
            "review this",
        ]);
        assert_eq!(cli.provider.as_deref(), Some("anthropic"));
        assert_eq!(cli.model.as_deref(), Some("claude-sonnet"));
        assert_eq!(cli.thinking, Some(ThinkingOption::High));
        assert!(cli.print);
        assert_eq!(cli.messages, ["review this"]);
    }

    #[test]
    fn rejects_conflicting_provider_and_qualified_model() {
        let cli = parse(&[
            "ri",
            "--provider",
            "openai",
            "--model",
            "anthropic/claude-sonnet",
            "-p",
            "review this",
        ]);
        assert!(matches!(
            cli.validate(),
            Err(CliError::InvalidArguments(message)) if message.contains("conflicts")
        ));
    }

    #[test]
    fn separates_file_arguments_without_io() {
        let cli = parse(&["ri", "--mode", "json", "@prompt.md", "summarize"]);
        assert_eq!(cli.mode, Some(ModeOption::Json));
        assert_eq!(
            cli.inputs(),
            InputArguments {
                files: vec![PathBuf::from("prompt.md")],
                messages: vec!["summarize".to_owned()],
            }
        );
    }

    #[test]
    fn parses_session_fork_command() {
        let cli = parse(&[
            "ri", "session", "fork", "abc", "--entry", "turn-2", "--at", "--id", "copy",
        ]);
        let Some(Command::Session {
            command: SessionCommand::Fork(arguments),
        }) = cli.command
        else {
            panic!("expected session fork");
        };
        assert_eq!(arguments.source, "abc");
        assert_eq!(arguments.entry.as_deref(), Some("turn-2"));
        assert!(arguments.at);
        assert_eq!(arguments.id.as_deref(), Some("copy"));
    }

    #[test]
    fn supports_pi_package_shortcuts() {
        let install = parse(&[
            "ri",
            "install",
            "git:https://example.test/repo",
            "-l",
            "--checksum",
            "0123456789abcdef",
        ]);
        let Some(Command::Install(arguments)) = install.command else {
            panic!("expected package install");
        };
        assert!(arguments.local);
        assert_eq!(arguments.checksum.as_deref(), Some("0123456789abcdef"));

        let uninstall = parse(&["ri", "uninstall", "git:https://example.test/repo"]);
        assert!(matches!(
            uninstall.command,
            Some(Command::Uninstall(PackageRemoveArgs { local: false, .. }))
        ));
    }

    #[test]
    fn expands_pi_multi_character_short_options() {
        let cli = Cli::try_parse_compatible_from([
            "ri", "-nt", "-ne", "-ns", "-np", "-nc", "-na", "-p", "inspect",
        ])
        .unwrap();
        assert!(cli.no_tools);
        assert!(cli.no_extensions);
        assert!(cli.no_skills);
        assert!(cli.no_prompt_templates);
        assert!(cli.no_context_files);
        assert!(cli.no_approve);
    }

    #[test]
    fn rejects_conflicting_session_selection() {
        let error =
            Cli::try_parse_from(["ri", "--continue", "--session", "abc"]).expect_err("conflict");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn validates_administrative_command_isolation() {
        let mut cli = parse(&["ri", "provider", "list"]);
        cli.print = true;
        assert!(cli.validate().is_err());
    }

    #[test]
    fn rejects_unknown_mode() {
        let error =
            Cli::try_parse_from(["ri", "--mode", "yaml"]).expect_err("unknown mode should fail");
        assert_eq!(error.kind(), clap::error::ErrorKind::InvalidValue);
    }

    #[test]
    fn list_models_accepts_optional_filter() {
        let all = parse(&["ri", "--list-models"]);
        assert_eq!(all.list_models.as_deref(), Some(""));
        let filtered = parse(&["ri", "--list-models=sonnet"]);
        assert_eq!(filtered.list_models.as_deref(), Some("sonnet"));
    }

    #[test]
    fn set_default_requires_an_explicit_model() {
        let cli = parse(&["ri", "--model", "anthropic/sonnet", "--set-default"]);
        assert!(cli.set_default);
        assert!(Cli::try_parse_from(["ri", "--set-default"]).is_err());
    }

    #[test]
    fn validates_metadata_request_isolation() {
        let cli = parse(&["ri", "--list-providers", "ignored prompt"]);
        assert!(cli.validate().is_err());
    }
}
