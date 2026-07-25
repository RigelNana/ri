//! Typed resource provenance and diagnostics.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Scope in which a resource was configured.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SourceScope {
    /// User-wide configuration.
    User,
    /// Project-local configuration. Project trust is required.
    Project,
    /// A command-line or in-memory resource that is not persisted.
    #[default]
    Temporary,
}

/// Whether a resource is a top-level entry or came from a package manifest.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum SourceOrigin {
    /// Explicitly configured or conventionally discovered resource.
    #[default]
    TopLevel,
    /// Resource exported by a package.
    Package,
}

/// Transport used to obtain a package.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageTransport {
    /// A directory already present on the local filesystem.
    Local,
    /// A Git repository.
    Git,
    /// A manifest and its files fetched over HTTPS.
    Https,
}

/// Typed description of how a resource entered the process.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SourceKind {
    /// A path explicitly present in settings.
    Configured,
    /// A path found by convention.
    Auto,
    /// A command-line path.
    Cli,
    /// A native extension compiled into the process.
    Inline {
        /// Stable identifier assigned by the native host.
        id: String,
    },
    /// A path contributed by another extension.
    Extension {
        /// Identifier of the contributing extension.
        id: String,
    },
    /// A package resource.
    Package {
        /// Package name declared by its manifest.
        name: String,
        /// Transport used to resolve the package.
        transport: PackageTransport,
    },
}

impl SourceKind {
    /// Whether this source was found automatically rather than configured.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Complete provenance attached to every loaded resource and registration.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SourceInfo {
    /// Path of the concrete resource. Synthetic native resources may use a
    /// display path such as `<inline:example>`.
    pub path: PathBuf,
    /// How the resource entered the process.
    pub source: SourceKind,
    /// User, project, or temporary scope.
    pub scope: SourceScope,
    /// Whether this resource came from a package.
    pub origin: SourceOrigin,
    /// Root against which manifest patterns and relative references resolve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_dir: Option<PathBuf>,
}

impl SourceInfo {
    /// Construct provenance for an explicitly configured local path.
    pub fn configured(path: impl Into<PathBuf>, scope: SourceScope) -> Self {
        let path = path.into();
        let base_dir = path.parent().map(Path::to_path_buf);
        Self {
            path,
            source: SourceKind::Configured,
            scope,
            origin: SourceOrigin::TopLevel,
            base_dir,
        }
    }

    /// Construct provenance for an automatically discovered path.
    pub fn auto(
        path: impl Into<PathBuf>,
        scope: SourceScope,
        base_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            path: path.into(),
            source: SourceKind::Auto,
            scope,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(base_dir.into()),
        }
    }

    /// Construct provenance for a native extension.
    pub fn inline(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            path: PathBuf::from(format!("<inline:{id}>")),
            source: SourceKind::Inline { id },
            scope: SourceScope::Temporary,
            origin: SourceOrigin::TopLevel,
            base_dir: None,
        }
    }

    /// Rank used for deterministic first-wins resource collision resolution.
    ///
    /// The stable ordering is project settings, project auto-discovery, user
    /// settings, user auto-discovery, and finally package resources.
    pub fn precedence_rank(&self) -> u8 {
        if self.origin == SourceOrigin::Package {
            return 4;
        }
        match (&self.scope, &self.source) {
            (SourceScope::Project, SourceKind::Auto) => 1,
            (SourceScope::User, SourceKind::Auto) => 3,
            (SourceScope::User, _) => 2,
            (SourceScope::Project | SourceScope::Temporary, _) => 0,
        }
    }

    /// Return a copy whose concrete path has been replaced while retaining
    /// package/configuration provenance.
    #[must_use]
    pub fn with_path(&self, path: impl Into<PathBuf>) -> Self {
        let mut copy = self.clone();
        copy.path = path.into();
        copy
    }
}

/// Kind of resource participating in discovery or a registry.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// Native or packaged extension.
    Extension,
    /// Tool registration.
    Tool,
    /// Slash-command registration.
    Command,
    /// Model-provider registration.
    Provider,
    /// Command-line flag registration.
    Flag,
    /// Keyboard-shortcut registration.
    Shortcut,
    /// Custom-message renderer.
    MessageRenderer,
    /// Custom session-entry renderer.
    EntryRenderer,
    /// Agent skill.
    Skill,
    /// Prompt template.
    Prompt,
    /// Hierarchical context file.
    Context,
    /// System-prompt replacement.
    SystemPrompt,
    /// Resolved package.
    Package,
}

/// Diagnostic severity and category.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticLevel {
    /// Non-fatal condition worth surfacing to the user.
    Warning,
    /// Resource could not be loaded or validated.
    Error,
    /// Two resources claimed the same registry key.
    Collision,
}

/// Details retained for a first-wins or last-wins collision.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Collision {
    /// Kind of resource that collided.
    pub resource_kind: ResourceKind,
    /// Conflicting registration name.
    pub name: String,
    /// Source retained by the collision policy.
    pub winner: SourceInfo,
    /// Source rejected or replaced by the collision policy.
    pub loser: SourceInfo,
}

/// A non-fatal resource or registry diagnostic.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Diagnostic severity.
    pub level: DiagnosticLevel,
    /// Human-readable explanation.
    pub message: String,
    /// Source directly associated with the diagnostic, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
    /// Structured collision details, for collision diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collision: Option<Collision>,
}

impl Diagnostic {
    /// Create a warning associated with a source.
    pub fn warning(message: impl Into<String>, source: SourceInfo) -> Self {
        Self {
            level: DiagnosticLevel::Warning,
            message: message.into(),
            source: Some(source),
            collision: None,
        }
    }

    /// Create an error associated with a source.
    pub fn error(message: impl Into<String>, source: SourceInfo) -> Self {
        Self {
            level: DiagnosticLevel::Error,
            message: message.into(),
            source: Some(source),
            collision: None,
        }
    }

    /// Create a collision diagnostic. `winner` is the registration that
    /// remains active after applying the registry's policy.
    pub fn collision(
        kind: ResourceKind,
        name: impl Into<String>,
        winner: SourceInfo,
        loser: SourceInfo,
    ) -> Self {
        let name = name.into();
        Self {
            level: DiagnosticLevel::Collision,
            message: format!("{kind:?} name {name:?} collision"),
            source: Some(loser.clone()),
            collision: Some(Collision {
                resource_kind: kind,
                name,
                winner,
                loser,
            }),
        }
    }
}

/// Canonical comparison key for an existing path. If canonicalization fails,
/// the absolute lexical path is used so a diagnostic can still be produced.
pub(crate) fn canonical_key(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_matches_documented_order() {
        let path = PathBuf::from("x");
        let project_setting = SourceInfo::configured(&path, SourceScope::Project);
        let project_auto = SourceInfo::auto(&path, SourceScope::Project, ".");
        let user_setting = SourceInfo::configured(&path, SourceScope::User);
        let user_auto = SourceInfo::auto(&path, SourceScope::User, ".");
        let mut package = user_setting.clone();
        package.origin = SourceOrigin::Package;

        assert_eq!(project_setting.precedence_rank(), 0);
        assert_eq!(project_auto.precedence_rank(), 1);
        assert_eq!(user_setting.precedence_rank(), 2);
        assert_eq!(user_auto.precedence_rank(), 3);
        assert_eq!(package.precedence_rank(), 4);
    }

    #[test]
    fn collision_retains_both_sources() {
        let first = SourceInfo::inline("first");
        let second = SourceInfo::inline("second");
        let diagnostic =
            Diagnostic::collision(ResourceKind::Tool, "read", first.clone(), second.clone());
        let collision = diagnostic.collision.expect("collision");
        assert_eq!(collision.winner, first);
        assert_eq!(collision.loser, second);
    }
}
