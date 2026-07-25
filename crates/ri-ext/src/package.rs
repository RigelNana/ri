//! Native `ri-package.toml` packages with lockfiles and integrity checks.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;

use globset::Glob;
use reqwest::header::{ETAG, LAST_MODIFIED};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::process::Command;
use url::Url;
use uuid::Uuid;
use walkdir::WalkDir;

use crate::atomic::{AtomicWriteOptions, atomic_write};
use crate::extension::GenerationClock;
use crate::source::{PackageTransport, SourceInfo, SourceKind, SourceOrigin, SourceScope};

const LOCK_VERSION: u32 = 1;
const MANIFEST_NAME: &str = "ri-package.toml";

/// Scope controlling path roots and project trust.
#[derive(
    Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PackageScope {
    /// User-wide package resolved relative to the user package root.
    User,
    /// Trust-gated package resolved relative to the project package root.
    Project,
    /// Session-only package resolved relative to the current directory.
    #[default]
    Temporary,
}

impl From<PackageScope> for SourceScope {
    fn from(value: PackageScope) -> Self {
        match value {
            PackageScope::User => Self::User,
            PackageScope::Project => Self::Project,
            PackageScope::Temporary => Self::Temporary,
        }
    }
}

/// Local, Git, or HTTPS package source.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PackageSource {
    /// Package already present on the local filesystem.
    Local {
        /// Absolute path or scope-relative directory.
        path: PathBuf,
    },
    /// Package cloned from a Git repository.
    Git {
        /// Repository URL or local repository path.
        repository: String,
        /// Optional branch, tag, or commit to check out.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        revision: Option<String>,
    },
    /// Package downloaded from an HTTPS manifest.
    Https {
        /// URL of the package's `ri-package.toml`.
        manifest_url: Url,
    },
}

impl PackageSource {
    /// Stable identity independent of a Git revision.
    pub fn identity(&self) -> String {
        match self {
            Self::Local { path } => format!("local:{}", path.display()),
            Self::Git { repository, .. } => format!("git:{repository}"),
            Self::Https { manifest_url } => format!("https:{manifest_url}"),
        }
    }

    fn transport(&self) -> PackageTransport {
        match self {
            Self::Local { .. } => PackageTransport::Local,
            Self::Git { .. } => PackageTransport::Git,
            Self::Https { .. } => PackageTransport::Https,
        }
    }
}

/// Resource filter attached to a package setting.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PackageFilter {
    /// If false, resources are disabled unless matched by an include list.
    pub autoload: Option<bool>,
    /// Patterns applied to every resource kind.
    pub include: Vec<String>,
    /// Patterns that disable matching resources after inclusion.
    pub exclude: Vec<String>,
    /// Kind-specific pattern lists. `Some([])` disables that kind.
    pub extensions: Option<Vec<String>>,
    /// Optional include patterns applied only to skills.
    pub skills: Option<Vec<String>>,
    /// Optional include patterns applied only to prompt templates.
    pub prompts: Option<Vec<String>>,
    /// Optional include patterns applied only to context files.
    pub contexts: Option<Vec<String>>,
}

/// One package configured in settings or on the command line.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PackageSpec {
    /// Package transport and location.
    pub source: PackageSource,
    /// Scope controlling relative paths and trust.
    pub scope: PackageScope,
    /// `sha256:<hex>` or a bare 64-character SHA-256 digest.
    pub checksum: Option<String>,
    /// Resource activation filter.
    pub filter: PackageFilter,
}

impl Default for PackageSpec {
    fn default() -> Self {
        Self {
            source: PackageSource::Local {
                path: PathBuf::from("."),
            },
            scope: PackageScope::Temporary,
            checksum: None,
            filter: PackageFilter::default(),
        }
    }
}

/// `[package]` identity in `ri-package.toml`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageIdentity {
    /// Stable package name.
    pub name: String,
    /// Package version retained in the lockfile.
    pub version: String,
    /// Optional human-readable summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// Resource exports in `ri-package.toml`.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ManifestResources {
    /// Exported `.wasm` extension paths or globs.
    pub extensions: Vec<String>,
    /// Exported skill markdown paths or globs.
    pub skills: Vec<String>,
    /// Exported prompt-template paths or globs.
    pub prompts: Vec<String>,
    /// Exported context markdown paths or globs.
    pub contexts: Vec<String>,
}

/// Native package manifest.
///
/// Resources normally live under `[resources]`. Top-level arrays are accepted
/// as a convenience and merged after the nested arrays.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageManifest {
    /// Required package identity table.
    pub package: PackageIdentity,
    /// Preferred nested resource declarations.
    #[serde(default)]
    pub resources: ManifestResources,
    /// Legacy/convenience top-level extension declarations.
    #[serde(default)]
    pub extensions: Vec<String>,
    /// Legacy/convenience top-level skill declarations.
    #[serde(default)]
    pub skills: Vec<String>,
    /// Legacy/convenience top-level prompt declarations.
    #[serde(default)]
    pub prompts: Vec<String>,
    /// Legacy/convenience top-level context declarations.
    #[serde(default)]
    pub contexts: Vec<String>,
}

impl PackageManifest {
    /// Parse and validate a native manifest.
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::TomlDeserialize`] for malformed TOML or
    /// [`PackageError::InvalidManifest`] when required identity fields are
    /// empty.
    pub fn parse(input: &str) -> Result<Self, PackageError> {
        let manifest: Self = toml::from_str(input)?;
        if manifest.package.name.trim().is_empty() {
            return Err(PackageError::InvalidManifest(
                "package.name must not be empty".to_owned(),
            ));
        }
        if manifest.package.version.trim().is_empty() {
            return Err(PackageError::InvalidManifest(
                "package.version must not be empty".to_owned(),
            ));
        }
        Ok(manifest)
    }

    fn merged_resources(&self) -> ManifestResources {
        let mut resources = self.resources.clone();
        resources.extensions.extend(self.extensions.clone());
        resources.skills.extend(self.skills.clone());
        resources.prompts.extend(self.prompts.clone());
        resources.contexts.extend(self.contexts.clone());
        resources
    }
}

/// Resource kind exported by a package.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageResourceKind {
    /// WebAssembly extension component.
    Extension,
    /// Agent skill markdown.
    Skill,
    /// Prompt-template markdown.
    Prompt,
    /// Context markdown.
    Context,
}

/// Concrete package resource after manifest expansion and filtering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackageResource {
    /// Resource category.
    pub kind: PackageResourceKind,
    /// Absolute resolved file path.
    pub path: PathBuf,
    /// Package-root-relative path.
    pub relative_path: PathBuf,
    /// Whether filtering selected the resource for automatic loading.
    pub enabled: bool,
    /// Package and transport provenance.
    pub source: SourceInfo,
}

/// Transport-specific source metadata retained in the lockfile and snapshot.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedSourceMetadata {
    /// Canonical local package location.
    Local {
        /// Canonical package root.
        canonical_path: PathBuf,
    },
    /// Resolved Git checkout metadata.
    Git {
        /// Repository requested by the package specification.
        repository: String,
        /// Optional branch, tag, or commit requested by the user.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        requested_revision: Option<String>,
        /// Exact checked-out commit hash.
        commit: String,
    },
    /// HTTP cache validators for an HTTPS package.
    Https {
        /// Source manifest URL.
        manifest_url: Url,
        /// HTTP entity tag returned with the manifest.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        etag: Option<String>,
        /// HTTP last-modified value returned with the manifest.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        last_modified: Option<String>,
    },
}

/// Resolved package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedPackage {
    /// Parsed package manifest.
    pub manifest: PackageManifest,
    /// Local root containing the resolved package.
    pub root: PathBuf,
    /// Transport-specific pinned metadata.
    pub metadata: ResolvedSourceMetadata,
    /// SHA-256 checksum covering the manifest and exported files.
    pub checksum: String,
    /// Expanded and filtered package resources.
    pub resources: Vec<PackageResource>,
}

/// Lockfile entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageLockEntry {
    /// Configured package source.
    pub source: PackageSource,
    /// Configured package scope.
    pub scope: PackageScope,
    /// Name read from the resolved manifest.
    pub package_name: String,
    /// Version read from the resolved manifest.
    pub package_version: String,
    /// Pinned package checksum.
    pub checksum: String,
    /// Pinned transport metadata.
    pub metadata: ResolvedSourceMetadata,
}

/// Deterministically serialized package lock.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageLock {
    /// Lockfile schema version.
    pub version: u32,
    /// Entries keyed by stable scope/source identity.
    #[serde(default)]
    pub packages: BTreeMap<String, PackageLockEntry>,
}

impl Default for PackageLock {
    fn default() -> Self {
        Self {
            version: LOCK_VERSION,
            packages: BTreeMap::new(),
        }
    }
}

impl PackageLock {
    /// Parse and validate a package lockfile.
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::TomlDeserialize`] for malformed TOML or
    /// [`PackageError::UnsupportedLockVersion`] for an unknown schema.
    pub fn parse(input: &str) -> Result<Self, PackageError> {
        let lock: Self = toml::from_str(input)?;
        if lock.version != LOCK_VERSION {
            return Err(PackageError::UnsupportedLockVersion(lock.version));
        }
        Ok(lock)
    }

    /// Serialize this lockfile deterministically as TOML.
    ///
    /// # Errors
    ///
    /// Returns [`PackageError::TomlSerialize`] if an entry cannot be encoded.
    pub fn to_toml(&self) -> Result<String, PackageError> {
        Ok(toml::to_string_pretty(self)?)
    }
}

/// Complete package-manager snapshot.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PackageSnapshot {
    /// Runtime generation that produced the snapshot.
    pub generation: u64,
    /// Resolved packages in specification order.
    pub packages: Vec<ResolvedPackage>,
    /// Flattened resources from all resolved packages.
    pub resources: Vec<PackageResource>,
    /// Lockfile written for this snapshot.
    pub lock: PackageLock,
}

/// Package manager configuration.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct PackageManagerOptions {
    /// Current working directory for temporary relative sources.
    pub cwd: PathBuf,
    /// Root used for user-scoped relative sources.
    pub user_root: PathBuf,
    /// Root used for project-scoped relative sources.
    pub project_root: PathBuf,
    /// Persistent Git and HTTPS cache directory.
    pub cache_dir: PathBuf,
    /// Package lockfile path.
    pub lock_path: PathBuf,
    /// Whether project-scoped packages may be resolved.
    pub project_trusted: bool,
    /// Whether network access is forbidden and caches must be used.
    pub offline: bool,
    /// Refresh unpinned Git checkouts instead of keeping their lock commit.
    pub update: bool,
    /// HTTPS is fail-closed unless a setting or prior lock supplies a checksum.
    pub allow_unverified_https: bool,
}

impl PackageManagerOptions {
    /// Create fail-closed package options with conventional project roots.
    pub fn new(
        cwd: impl Into<PathBuf>,
        user_root: impl Into<PathBuf>,
        cache_dir: impl Into<PathBuf>,
        lock_path: impl Into<PathBuf>,
    ) -> Self {
        let cwd = cwd.into();
        Self {
            project_root: cwd.join(".ri"),
            cwd,
            user_root: user_root.into(),
            cache_dir: cache_dir.into(),
            lock_path: lock_path.into(),
            project_trusted: false,
            offline: false,
            update: false,
            allow_unverified_https: false,
        }
    }
}

/// Package resolution failure.
#[derive(Debug, Error)]
pub enum PackageError {
    /// Package manifest or lockfile TOML could not be decoded.
    #[error("package manifest: {0}")]
    TomlDeserialize(#[from] toml::de::Error),
    /// Package lockfile TOML could not be encoded.
    #[error("package serialization: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    /// Manifest values or declarations are invalid.
    #[error("invalid package manifest: {0}")]
    InvalidManifest(String),
    /// A resource path escapes the package root.
    #[error("unsafe package path {0:?}")]
    UnsafePath(PathBuf),
    /// A declared package resource is absent.
    #[error("package resource does not exist: {0}")]
    MissingResource(PathBuf),
    /// A project-scoped package was requested without trust.
    #[error("project is not trusted; refusing project package {0}")]
    UntrustedProject(String),
    /// An HTTPS source used a non-HTTPS URL.
    #[error("HTTPS package URL must use https: {0}")]
    InsecureUrl(Url),
    /// No explicit or locked checksum protects an HTTPS package.
    #[error("HTTPS package requires a checksum or existing lock: {0}")]
    MissingChecksum(Url),
    /// Resolved content did not match the expected digest.
    #[error("package checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Configured or locked checksum.
        expected: String,
        /// Checksum computed from resolved content.
        actual: String,
    },
    /// Offline mode could not find a required cached package.
    #[error("offline package cache is missing: {0}")]
    OfflineCacheMiss(PathBuf),
    /// Filesystem operation failed.
    #[error("package I/O at {path}: {source}")]
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// HTTP request or response processing failed.
    #[error("package HTTP request to {url}: {message}")]
    Http {
        /// URL being requested.
        url: Url,
        /// Transport or HTTP status failure.
        message: String,
    },
    /// Git subprocess failed.
    #[error("git command failed ({command}): {message}")]
    Git {
        /// Rendered Git command.
        command: String,
        /// Standard error or process failure.
        message: String,
    },
    /// Lockfile schema version is unsupported.
    #[error("unsupported package lock version {0}")]
    UnsupportedLockVersion(u32),
}

/// Reloadable package manager.
#[derive(Debug)]
pub struct PackageManager {
    options: PackageManagerOptions,
    clock: GenerationClock,
    client: reqwest::Client,
    snapshot: PackageSnapshot,
}

impl PackageManager {
    /// Create a package manager at the clock's current generation.
    pub fn new(options: PackageManagerOptions, clock: GenerationClock) -> Self {
        let generation = clock.current();
        Self {
            options,
            clock,
            client: reqwest::Client::new(),
            snapshot: PackageSnapshot {
                generation,
                ..PackageSnapshot::default()
            },
        }
    }

    /// Mutably access options used by the next reload.
    pub fn options_mut(&mut self) -> &mut PackageManagerOptions {
        &mut self.options
    }

    /// Return the active immutable package snapshot.
    pub fn snapshot(&self) -> &PackageSnapshot {
        &self.snapshot
    }

    /// Resolve all specs, atomically write the lockfile, then advance the
    /// shared generation. Failed reloads leave the previous snapshot active.
    ///
    /// # Errors
    ///
    /// Returns [`PackageError`] when trust is missing, a source cannot be
    /// resolved, integrity validation fails, or the lockfile cannot be read or
    /// written.
    pub async fn reload(&mut self, specs: &[PackageSpec]) -> Result<(), PackageError> {
        let mut lock = load_lock(&self.options.lock_path).await?;
        let specs = dedupe_specs(specs);
        let mut packages = Vec::new();
        let mut resources = Vec::new();
        let mut next_entries = BTreeMap::new();

        for spec in specs {
            self.assert_trusted(&spec)?;
            let key = lock_key(&spec);
            let previous = lock.packages.get(&key);
            let package = self.resolve_one(&spec, previous).await?;
            next_entries.insert(
                key,
                PackageLockEntry {
                    source: spec.source.clone(),
                    scope: spec.scope,
                    package_name: package.manifest.package.name.clone(),
                    package_version: package.manifest.package.version.clone(),
                    checksum: package.checksum.clone(),
                    metadata: package.metadata.clone(),
                },
            );
            resources.extend(package.resources.clone());
            packages.push(package);
        }

        lock.version = LOCK_VERSION;
        lock.packages = next_entries;
        write_lock(&self.options.lock_path, &lock).await?;
        let generation = self.clock.advance();
        self.snapshot = PackageSnapshot {
            generation,
            packages,
            resources,
            lock,
        };
        Ok(())
    }

    fn assert_trusted(&self, spec: &PackageSpec) -> Result<(), PackageError> {
        if spec.scope == PackageScope::Project && !self.options.project_trusted {
            return Err(PackageError::UntrustedProject(spec.source.identity()));
        }
        Ok(())
    }

    async fn resolve_one(
        &self,
        spec: &PackageSpec,
        previous: Option<&PackageLockEntry>,
    ) -> Result<ResolvedPackage, PackageError> {
        match &spec.source {
            PackageSource::Local { path } => self.resolve_local(spec, path).await,
            PackageSource::Git {
                repository,
                revision,
            } => {
                self.resolve_git(spec, repository, revision.as_deref(), previous)
                    .await
            }
            PackageSource::Https { manifest_url } => {
                self.resolve_https(spec, manifest_url, previous).await
            }
        }
    }

    async fn resolve_local(
        &self,
        spec: &PackageSpec,
        input: &Path,
    ) -> Result<ResolvedPackage, PackageError> {
        let root = if input.is_absolute() {
            input.to_path_buf()
        } else {
            self.scope_root(spec.scope).join(input)
        };
        let root = std::fs::canonicalize(&root).map_err(|source| PackageError::Io {
            path: root.clone(),
            source,
        })?;
        let manifest_path = root.join(MANIFEST_NAME);
        let manifest_bytes =
            tokio::fs::read(&manifest_path)
                .await
                .map_err(|source| PackageError::Io {
                    path: manifest_path.clone(),
                    source,
                })?;
        let manifest_text = String::from_utf8(manifest_bytes.clone()).map_err(|error| {
            PackageError::InvalidManifest(format!("manifest must be UTF-8: {error}"))
        })?;
        let manifest = PackageManifest::parse(&manifest_text)?;
        let all = expand_local_resources(&root, &manifest)?;
        let checksum = hash_package(&manifest_bytes, &root, &all).await?;
        verify_checksum(spec.checksum.as_deref(), &checksum)?;
        let source = package_source_info(spec, &manifest, &root);
        let resources = apply_filter(all, &spec.filter, &source)?;
        Ok(ResolvedPackage {
            manifest,
            root: root.clone(),
            metadata: ResolvedSourceMetadata::Local {
                canonical_path: root,
            },
            checksum,
            resources,
        })
    }

    async fn resolve_git(
        &self,
        spec: &PackageSpec,
        repository: &str,
        revision: Option<&str>,
        previous: Option<&PackageLockEntry>,
    ) -> Result<ResolvedPackage, PackageError> {
        let cache_key = sha256_text(&format!("git:{repository}"));
        let root = self.options.cache_dir.join("git").join(&cache_key[..24]);
        if !root.exists() {
            if self.options.offline {
                return Err(PackageError::OfflineCacheMiss(root));
            }
            clone_git(repository, &root).await?;
        } else if self.options.update && !self.options.offline {
            run_git(&root, &["fetch", "--prune", "--tags", "origin"]).await?;
        }

        let locked_commit = previous.and_then(|entry| match &entry.metadata {
            ResolvedSourceMetadata::Git { commit, .. } => Some(commit.as_str()),
            _ => None,
        });
        let target = revision.or_else(|| (!self.options.update).then_some(locked_commit).flatten());
        if let Some(target) = target {
            if !self.options.offline {
                run_git(&root, &["fetch", "origin", target]).await?;
            }
            run_git(&root, &["checkout", "--detach", target]).await?;
        } else if self.options.update {
            let remote = run_git(&root, &["rev-parse", "origin/HEAD^{commit}"]).await?;
            run_git(&root, &["checkout", "--detach", remote.trim()]).await?;
        }
        let commit = run_git(&root, &["rev-parse", "HEAD^{commit}"])
            .await?
            .trim()
            .to_owned();

        let manifest_path = root.join(MANIFEST_NAME);
        let manifest_bytes =
            tokio::fs::read(&manifest_path)
                .await
                .map_err(|source| PackageError::Io {
                    path: manifest_path.clone(),
                    source,
                })?;
        let manifest_text = String::from_utf8(manifest_bytes.clone()).map_err(|error| {
            PackageError::InvalidManifest(format!("manifest must be UTF-8: {error}"))
        })?;
        let manifest = PackageManifest::parse(&manifest_text)?;
        let all = expand_local_resources(&root, &manifest)?;
        let checksum = hash_package(&manifest_bytes, &root, &all).await?;
        let request_unchanged = previous.is_some_and(|entry| {
            matches!(
                &entry.metadata,
                ResolvedSourceMetadata::Git {
                    requested_revision,
                    ..
                } if requested_revision.as_deref() == revision
            )
        });
        let expected = spec.checksum.as_deref().or_else(|| {
            (!self.options.update && request_unchanged)
                .then(|| previous.map(|entry| entry.checksum.as_str()))
                .flatten()
        });
        verify_checksum(expected, &checksum)?;
        let source = package_source_info(spec, &manifest, &root);
        let resources = apply_filter(all, &spec.filter, &source)?;
        Ok(ResolvedPackage {
            manifest,
            root,
            metadata: ResolvedSourceMetadata::Git {
                repository: repository.to_owned(),
                requested_revision: revision.map(str::to_owned),
                commit,
            },
            checksum,
            resources,
        })
    }

    async fn resolve_https(
        &self,
        spec: &PackageSpec,
        manifest_url: &Url,
        previous: Option<&PackageLockEntry>,
    ) -> Result<ResolvedPackage, PackageError> {
        if manifest_url.scheme() != "https" {
            return Err(PackageError::InsecureUrl(manifest_url.clone()));
        }
        let cache_key = sha256_text(manifest_url.as_str());
        let root = self.options.cache_dir.join("https").join(&cache_key[..24]);
        let expected = spec
            .checksum
            .as_deref()
            .or_else(|| previous.map(|entry| entry.checksum.as_str()));
        if expected.is_none() && !self.options.allow_unverified_https {
            return Err(PackageError::MissingChecksum(manifest_url.clone()));
        }
        if self.options.offline {
            if !root.join(MANIFEST_NAME).exists() {
                return Err(PackageError::OfflineCacheMiss(root));
            }
            return resolve_cached_https(spec, manifest_url, &root, expected).await;
        }

        let response = self
            .client
            .get(manifest_url.clone())
            .send()
            .await
            .map_err(|error| PackageError::Http {
                url: manifest_url.clone(),
                message: error.to_string(),
            })?;
        let response = response
            .error_for_status()
            .map_err(|error| PackageError::Http {
                url: manifest_url.clone(),
                message: error.to_string(),
            })?;
        let etag = response
            .headers()
            .get(ETAG)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let last_modified = response
            .headers()
            .get(LAST_MODIFIED)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let manifest_bytes = response.bytes().await.map_err(|error| PackageError::Http {
            url: manifest_url.clone(),
            message: error.to_string(),
        })?;
        let manifest_text = String::from_utf8(manifest_bytes.to_vec()).map_err(|error| {
            PackageError::InvalidManifest(format!("manifest must be UTF-8: {error}"))
        })?;
        let manifest = PackageManifest::parse(&manifest_text)?;
        let declarations = literal_manifest_resources(&manifest)?;
        let temporary = self
            .options
            .cache_dir
            .join("https")
            .join(format!(".{}.tmp", Uuid::new_v4()));
        tokio::fs::create_dir_all(&temporary)
            .await
            .map_err(|source| PackageError::Io {
                path: temporary.clone(),
                source,
            })?;
        tokio::fs::write(temporary.join(MANIFEST_NAME), &manifest_bytes)
            .await
            .map_err(|source| PackageError::Io {
                path: temporary.join(MANIFEST_NAME),
                source,
            })?;
        for declaration in &declarations {
            let relative = &declaration.relative;
            let resource_url = manifest_url.join(&path_to_url(relative)).map_err(|error| {
                PackageError::InvalidManifest(format!(
                    "invalid HTTPS resource path {}: {error}",
                    relative.display()
                ))
            })?;
            let response = self
                .client
                .get(resource_url.clone())
                .send()
                .await
                .map_err(|error| PackageError::Http {
                    url: resource_url.clone(),
                    message: error.to_string(),
                })?
                .error_for_status()
                .map_err(|error| PackageError::Http {
                    url: resource_url.clone(),
                    message: error.to_string(),
                })?;
            let bytes = response.bytes().await.map_err(|error| PackageError::Http {
                url: resource_url,
                message: error.to_string(),
            })?;
            let target = safe_join(&temporary, relative)?;
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .map_err(|source| PackageError::Io {
                        path: parent.to_path_buf(),
                        source,
                    })?;
            }
            tokio::fs::write(&target, bytes)
                .await
                .map_err(|source| PackageError::Io {
                    path: target,
                    source,
                })?;
        }
        let all = declarations
            .iter()
            .map(|declaration| LocalDeclaration {
                kind: declaration.kind,
                relative: declaration.relative.clone(),
                absolute: temporary.join(&declaration.relative),
            })
            .collect::<Vec<_>>();
        let checksum = hash_package(&manifest_bytes, &temporary, &all).await?;
        verify_checksum(expected, &checksum)?;
        if root.exists() {
            tokio::fs::remove_dir_all(&root)
                .await
                .map_err(|source| PackageError::Io {
                    path: root.clone(),
                    source,
                })?;
        }
        if let Some(parent) = root.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| PackageError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
        }
        tokio::fs::rename(&temporary, &root)
            .await
            .map_err(|source| PackageError::Io {
                path: root.clone(),
                source,
            })?;

        let all = declarations
            .into_iter()
            .map(|declaration| LocalDeclaration {
                kind: declaration.kind,
                absolute: root.join(&declaration.relative),
                relative: declaration.relative,
            })
            .collect();
        let source = package_source_info(spec, &manifest, &root);
        let resources = apply_filter(all, &spec.filter, &source)?;
        Ok(ResolvedPackage {
            manifest,
            root,
            metadata: ResolvedSourceMetadata::Https {
                manifest_url: manifest_url.clone(),
                etag,
                last_modified,
            },
            checksum,
            resources,
        })
    }

    fn scope_root(&self, scope: PackageScope) -> &Path {
        match scope {
            PackageScope::User => &self.options.user_root,
            PackageScope::Project => &self.options.project_root,
            PackageScope::Temporary => &self.options.cwd,
        }
    }
}

async fn resolve_cached_https(
    spec: &PackageSpec,
    manifest_url: &Url,
    root: &Path,
    expected: Option<&str>,
) -> Result<ResolvedPackage, PackageError> {
    let manifest_path = root.join(MANIFEST_NAME);
    let manifest_bytes =
        tokio::fs::read(&manifest_path)
            .await
            .map_err(|source| PackageError::Io {
                path: manifest_path,
                source,
            })?;
    let manifest_text = String::from_utf8(manifest_bytes.clone()).map_err(|error| {
        PackageError::InvalidManifest(format!("manifest must be UTF-8: {error}"))
    })?;
    let manifest = PackageManifest::parse(&manifest_text)?;
    let declarations = literal_manifest_resources(&manifest)?
        .into_iter()
        .map(|declaration| LocalDeclaration {
            kind: declaration.kind,
            absolute: root.join(&declaration.relative),
            relative: declaration.relative,
        })
        .collect::<Vec<_>>();
    for declaration in &declarations {
        if !declaration.absolute.is_file() {
            return Err(PackageError::OfflineCacheMiss(declaration.absolute.clone()));
        }
    }
    let checksum = hash_package(&manifest_bytes, root, &declarations).await?;
    verify_checksum(expected, &checksum)?;
    let source = package_source_info(spec, &manifest, root);
    let resources = apply_filter(declarations, &spec.filter, &source)?;
    Ok(ResolvedPackage {
        manifest,
        root: root.to_path_buf(),
        metadata: ResolvedSourceMetadata::Https {
            manifest_url: manifest_url.clone(),
            etag: None,
            last_modified: None,
        },
        checksum,
        resources,
    })
}

#[derive(Clone, Debug)]
struct LocalDeclaration {
    kind: PackageResourceKind,
    relative: PathBuf,
    absolute: PathBuf,
}

fn expand_local_resources(
    root: &Path,
    manifest: &PackageManifest,
) -> Result<Vec<LocalDeclaration>, PackageError> {
    let resources = manifest.merged_resources();
    let mut declarations = Vec::new();
    expand_entries(
        root,
        &resources.extensions,
        PackageResourceKind::Extension,
        &mut declarations,
    )?;
    expand_entries(
        root,
        &resources.skills,
        PackageResourceKind::Skill,
        &mut declarations,
    )?;
    expand_entries(
        root,
        &resources.prompts,
        PackageResourceKind::Prompt,
        &mut declarations,
    )?;
    expand_entries(
        root,
        &resources.contexts,
        PackageResourceKind::Context,
        &mut declarations,
    )?;
    declarations.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.relative.cmp(&right.relative))
    });
    declarations.dedup_by(|left, right| left.kind == right.kind && left.relative == right.relative);
    Ok(declarations)
}

fn expand_entries(
    root: &Path,
    entries: &[String],
    kind: PackageResourceKind,
    output: &mut Vec<LocalDeclaration>,
) -> Result<(), PackageError> {
    for entry in entries {
        if contains_glob(entry) {
            let matcher = Glob::new(entry)
                .map_err(|error| {
                    PackageError::InvalidManifest(format!(
                        "invalid resource glob {entry:?}: {error}"
                    ))
                })?
                .compile_matcher();
            let mut matched_paths = Vec::new();
            for item in WalkDir::new(root).follow_links(false) {
                let item = item.map_err(|error| package_walk_error(error, root))?;
                if !item.file_type().is_file() {
                    continue;
                }
                let relative = item
                    .path()
                    .strip_prefix(root)
                    .expect("WalkDir entry remains below its root")
                    .to_path_buf();
                if matcher.is_match(&relative) {
                    matched_paths.push((relative, item.into_path()));
                }
            }
            matched_paths.sort_by(|left, right| left.0.cmp(&right.0));
            for (relative, absolute) in matched_paths {
                validate_resource_type(kind, &absolute)?;
                output.push(LocalDeclaration {
                    kind,
                    relative,
                    absolute,
                });
            }
            continue;
        }
        let relative = validate_relative(Path::new(entry))?;
        let absolute = safe_join(root, &relative)?;
        if absolute.is_dir() {
            let mut files = Vec::new();
            for item in WalkDir::new(&absolute).follow_links(false) {
                let item = item.map_err(|error| package_walk_error(error, &absolute))?;
                if item.file_type().is_file() && validate_resource_type(kind, item.path()).is_ok() {
                    files.push(item.into_path());
                }
            }
            files.sort();
            for absolute in files {
                output.push(LocalDeclaration {
                    kind,
                    relative: absolute
                        .strip_prefix(root)
                        .expect("walked below root")
                        .to_path_buf(),
                    absolute,
                });
            }
        } else if absolute.is_file() {
            validate_resource_type(kind, &absolute)?;
            output.push(LocalDeclaration {
                kind,
                relative,
                absolute,
            });
        } else {
            return Err(PackageError::MissingResource(absolute));
        }
    }
    Ok(())
}

fn package_walk_error(error: walkdir::Error, root: &Path) -> PackageError {
    let path = error.path().unwrap_or(root).to_path_buf();
    let message = error.to_string();
    PackageError::Io {
        path,
        source: error
            .into_io_error()
            .unwrap_or_else(|| std::io::Error::other(message)),
    }
}

#[derive(Clone, Debug)]
struct LiteralDeclaration {
    kind: PackageResourceKind,
    relative: PathBuf,
}

fn literal_manifest_resources(
    manifest: &PackageManifest,
) -> Result<Vec<LiteralDeclaration>, PackageError> {
    let resources = manifest.merged_resources();
    let mut output = Vec::new();
    for (kind, entries) in [
        (PackageResourceKind::Extension, resources.extensions),
        (PackageResourceKind::Skill, resources.skills),
        (PackageResourceKind::Prompt, resources.prompts),
        (PackageResourceKind::Context, resources.contexts),
    ] {
        for entry in entries {
            if contains_glob(&entry) {
                return Err(PackageError::InvalidManifest(format!(
                    "HTTPS manifests cannot use resource globs: {entry:?}"
                )));
            }
            let relative = validate_relative(Path::new(&entry))?;
            validate_resource_type(kind, &relative)?;
            output.push(LiteralDeclaration { kind, relative });
        }
    }
    output.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.relative.cmp(&right.relative))
    });
    output.dedup_by(|left, right| left.kind == right.kind && left.relative == right.relative);
    Ok(output)
}

fn validate_resource_type(kind: PackageResourceKind, path: &Path) -> Result<(), PackageError> {
    let valid = match kind {
        PackageResourceKind::Extension => path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("wasm")),
        PackageResourceKind::Skill | PackageResourceKind::Prompt | PackageResourceKind::Context => {
            path.extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
        }
    };
    if valid {
        Ok(())
    } else {
        Err(PackageError::InvalidManifest(format!(
            "{kind:?} resource has an unsupported extension: {}",
            path.display()
        )))
    }
}

fn contains_glob(value: &str) -> bool {
    value.contains('*') || value.contains('?') || value.contains('[')
}

fn apply_filter(
    declarations: Vec<LocalDeclaration>,
    filter: &PackageFilter,
    source: &SourceInfo,
) -> Result<Vec<PackageResource>, PackageError> {
    let general_includes = compile_patterns(&filter.include)?;
    let general_excludes = compile_patterns(&filter.exclude)?;
    let mut output = Vec::with_capacity(declarations.len());
    for declaration in declarations {
        let relative = path_to_url(&declaration.relative);
        let kind_patterns = match declaration.kind {
            PackageResourceKind::Extension => filter.extensions.as_ref(),
            PackageResourceKind::Skill => filter.skills.as_ref(),
            PackageResourceKind::Prompt => filter.prompts.as_ref(),
            PackageResourceKind::Context => filter.contexts.as_ref(),
        };
        let kind_patterns = kind_patterns
            .map(|patterns| compile_patterns(patterns))
            .transpose()?;
        let autoload = filter.autoload.unwrap_or(true);
        let mut enabled = autoload;
        if !general_includes.is_empty() {
            enabled = matches_patterns(&general_includes, &relative);
        }
        if let Some(patterns) = &kind_patterns {
            enabled = !patterns.is_empty() && matches_patterns(patterns, &relative);
        }
        if matches_patterns(&general_excludes, &relative) {
            enabled = false;
        }
        output.push(PackageResource {
            kind: declaration.kind,
            path: declaration.absolute.clone(),
            relative_path: declaration.relative,
            enabled,
            source: source.with_path(declaration.absolute),
        });
    }
    Ok(output)
}

fn compile_patterns(patterns: &[String]) -> Result<Vec<globset::GlobMatcher>, PackageError> {
    patterns
        .iter()
        .map(|pattern| {
            Glob::new(pattern)
                .map(|glob| glob.compile_matcher())
                .map_err(|error| {
                    PackageError::InvalidManifest(format!(
                        "invalid package filter {pattern:?}: {error}"
                    ))
                })
        })
        .collect()
}

fn matches_patterns(patterns: &[globset::GlobMatcher], relative: &str) -> bool {
    patterns.iter().any(|matcher| matcher.is_match(relative))
}

fn package_source_info(spec: &PackageSpec, manifest: &PackageManifest, root: &Path) -> SourceInfo {
    SourceInfo {
        path: root.to_path_buf(),
        source: SourceKind::Package {
            name: manifest.package.name.clone(),
            transport: spec.source.transport(),
        },
        scope: spec.scope.into(),
        origin: SourceOrigin::Package,
        base_dir: Some(root.to_path_buf()),
    }
}

async fn hash_package(
    manifest: &[u8],
    root: &Path,
    declarations: &[LocalDeclaration],
) -> Result<String, PackageError> {
    let mut digest = Sha256::new();
    digest.update(b"ri-package-v1\0");
    digest.update((manifest.len() as u64).to_le_bytes());
    digest.update(manifest);
    let mut ordered = declarations.to_vec();
    ordered.sort_by(|left, right| {
        left.kind
            .cmp(&right.kind)
            .then_with(|| left.relative.cmp(&right.relative))
    });
    let canonical_root = std::fs::canonicalize(root).map_err(|source| PackageError::Io {
        path: root.to_path_buf(),
        source,
    })?;
    for declaration in ordered {
        let relative = path_to_url(&declaration.relative);
        let canonical_file =
            std::fs::canonicalize(&declaration.absolute).map_err(|source| PackageError::Io {
                path: declaration.absolute.clone(),
                source,
            })?;
        if canonical_file != canonical_root && !canonical_file.starts_with(&canonical_root) {
            return Err(PackageError::UnsafePath(declaration.absolute));
        }
        let bytes = tokio::fs::read(&declaration.absolute)
            .await
            .map_err(|source| PackageError::Io {
                path: declaration.absolute.clone(),
                source,
            })?;
        digest.update([declaration.kind as u8]);
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative.as_bytes());
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(format!(
        "sha256:{}",
        hex_digest(digest.finalize().as_slice())
    ))
}

fn verify_checksum(expected: Option<&str>, actual: &str) -> Result<(), PackageError> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let expected = normalize_checksum(expected);
    if expected.eq_ignore_ascii_case(actual) {
        Ok(())
    } else {
        Err(PackageError::ChecksumMismatch {
            expected,
            actual: actual.to_owned(),
        })
    }
}

fn normalize_checksum(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("sha256:"))
    {
        format!("sha256:{}", &trimmed[7..])
    } else {
        format!("sha256:{trimmed}")
    }
}

fn sha256_text(value: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(value.as_bytes());
    hex_digest(digest.finalize().as_slice())
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn validate_relative(path: &Path) -> Result<PathBuf, PackageError> {
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(PackageError::UnsafePath(path.to_path_buf()));
    }
    for component in path.components() {
        if matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        ) {
            return Err(PackageError::UnsafePath(path.to_path_buf()));
        }
    }
    Ok(path.to_path_buf())
}

fn safe_join(root: &Path, relative: &Path) -> Result<PathBuf, PackageError> {
    let relative = validate_relative(relative)?;
    Ok(root.join(relative))
}

fn path_to_url(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

async fn clone_git(repository: &str, root: &Path) -> Result<(), PackageError> {
    let parent = root.parent().unwrap_or_else(|| Path::new("."));
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|source| PackageError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    let temporary = parent.join(format!(".git-package.{}.tmp", Uuid::new_v4()));
    let output = Command::new("git")
        .args(["clone", "--no-tags", repository])
        .arg(&temporary)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|source| PackageError::Io {
            path: PathBuf::from("git"),
            source,
        })?;
    if !output.status.success() {
        let _ = tokio::fs::remove_dir_all(&temporary).await;
        return Err(PackageError::Git {
            command: format!("git clone --no-tags {repository} {}", temporary.display()),
            message: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    tokio::fs::rename(&temporary, root)
        .await
        .map_err(|source| PackageError::Io {
            path: root.to_path_buf(),
            source,
        })
}

async fn run_git(root: &Path, arguments: &[&str]) -> Result<String, PackageError> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(root)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|source| PackageError::Io {
            path: PathBuf::from("git"),
            source,
        })?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(PackageError::Git {
            command: format!("git {}", arguments.join(" ")),
            message: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        })
    }
}

fn dedupe_specs(specs: &[PackageSpec]) -> Vec<PackageSpec> {
    let mut ordered = specs.to_vec();
    ordered.sort_by_key(|spec| match spec.scope {
        PackageScope::Project => 0,
        PackageScope::Temporary => 1,
        PackageScope::User => 2,
    });
    let mut seen = BTreeSet::new();
    ordered
        .into_iter()
        .filter(|spec| seen.insert(spec.source.identity()))
        .collect()
}

fn lock_key(spec: &PackageSpec) -> String {
    format!("{:?}:{}", spec.scope, spec.source.identity())
}

async fn load_lock(path: &Path) -> Result<PackageLock, PackageError> {
    match tokio::fs::read_to_string(path).await {
        Ok(content) => PackageLock::parse(&content),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(PackageLock::default()),
        Err(source) => Err(PackageError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}

async fn write_lock(path: &Path, lock: &PackageLock) -> Result<(), PackageError> {
    let content = lock.to_toml()?;
    atomic_write(path, content.as_bytes(), AtomicWriteOptions::default())
        .await
        .map_err(|source| PackageError::Io {
            path: path.to_path_buf(),
            source,
        })
}

#[cfg(test)]
mod tests {
    use std::process::Command as StdCommand;

    use tempfile::tempdir;

    use super::*;

    fn write(path: &Path, content: &[u8]) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, content).expect("write");
    }

    fn run_test_git(directory: &Path, arguments: &[&str]) -> String {
        let output = StdCommand::new("git")
            .current_dir(directory)
            .env("GIT_AUTHOR_NAME", "ri-ext tests")
            .env("GIT_AUTHOR_EMAIL", "ri-ext@example.invalid")
            .env("GIT_COMMITTER_NAME", "ri-ext tests")
            .env("GIT_COMMITTER_EMAIL", "ri-ext@example.invalid")
            .args(arguments)
            .output()
            .expect("run git");
        assert!(
            output.status.success(),
            "git {arguments:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("git output")
    }

    fn local_spec(path: PathBuf) -> PackageSpec {
        PackageSpec {
            source: PackageSource::Local { path },
            scope: PackageScope::Temporary,
            checksum: None,
            filter: PackageFilter::default(),
        }
    }

    fn options(temp: &Path) -> PackageManagerOptions {
        PackageManagerOptions::new(
            temp,
            temp.join("user"),
            temp.join("cache"),
            temp.join("ri-packages.lock"),
        )
    }

    #[test]
    fn manifest_merges_nested_and_top_level_resources() {
        let manifest = PackageManifest::parse(
            r#"
extensions = ["extra.wasm"]

[package]
name = "demo"
version = "1.0.0"

[resources]
extensions = ["main.wasm"]
skills = ["skills/demo/SKILL.md"]
"#,
        )
        .expect("manifest");
        let resources = manifest.merged_resources();
        assert_eq!(resources.extensions, vec!["main.wasm", "extra.wasm"]);
        assert_eq!(resources.skills, vec!["skills/demo/SKILL.md"]);
    }

    #[tokio::test]
    async fn local_package_resolves_filters_locks_and_invalidates_generation() {
        let temp = tempdir().expect("tempdir");
        let package = temp.path().join("package");
        write(
            &package.join(MANIFEST_NAME),
            br#"
[package]
name = "demo"
version = "1.2.3"

[resources]
extensions = ["extensions/*.wasm"]
skills = ["skills/demo/SKILL.md"]
prompts = ["prompts/*.md"]
"#,
        );
        write(&package.join("extensions").join("one.wasm"), b"wasm");
        write(
            &package.join("skills").join("demo").join("SKILL.md"),
            b"skill",
        );
        write(&package.join("prompts").join("one.md"), b"prompt");

        let clock = GenerationClock::default();
        let before = clock.current();
        let mut manager = PackageManager::new(options(temp.path()), clock.clone());
        let mut spec = local_spec(package);
        spec.filter.prompts = Some(Vec::new());
        manager.reload(&[spec]).await.expect("reload");
        assert_eq!(clock.current(), before + 1);
        assert_eq!(manager.snapshot().packages.len(), 1);
        assert!(
            manager
                .snapshot()
                .resources
                .iter()
                .any(|resource| resource.kind == PackageResourceKind::Prompt && !resource.enabled)
        );
        assert!(temp.path().join("ri-packages.lock").exists());
        let lock = load_lock(&temp.path().join("ri-packages.lock"))
            .await
            .expect("lock");
        assert_eq!(lock.packages.len(), 1);
        assert!(
            lock.packages
                .values()
                .next()
                .expect("entry")
                .checksum
                .starts_with("sha256:")
        );
    }

    #[tokio::test]
    async fn git_package_pins_commit_and_checksum_in_lock() {
        let temp = tempdir().expect("tempdir");
        let repository = temp.path().join("repository");
        std::fs::create_dir_all(&repository).expect("repository");
        run_test_git(&repository, &["init"]);
        write(
            &repository.join(MANIFEST_NAME),
            br#"
[package]
name = "git-demo"
version = "1"

[resources]
skills = ["skills/demo/SKILL.md"]
"#,
        );
        write(
            &repository.join("skills").join("demo").join("SKILL.md"),
            b"git skill",
        );
        run_test_git(&repository, &["add", "."]);
        run_test_git(&repository, &["commit", "-m", "initial"]);

        let spec = PackageSpec {
            source: PackageSource::Git {
                repository: repository.to_string_lossy().into_owned(),
                revision: None,
            },
            scope: PackageScope::Temporary,
            checksum: None,
            filter: PackageFilter::default(),
        };
        let mut manager = PackageManager::new(options(temp.path()), GenerationClock::default());
        manager.reload(&[spec]).await.expect("resolve Git package");

        let package = &manager.snapshot().packages[0];
        let initial_checksum = package.checksum.clone();
        let ResolvedSourceMetadata::Git {
            repository: resolved_repository,
            commit,
            ..
        } = &package.metadata
        else {
            panic!("expected Git metadata");
        };
        assert_eq!(
            resolved_repository.as_str(),
            repository.to_string_lossy().as_ref()
        );
        assert_eq!(commit.len(), 40);
        let lock_entry = manager
            .snapshot()
            .lock
            .packages
            .values()
            .next()
            .expect("lock entry");
        assert_eq!(lock_entry.checksum, package.checksum);
        assert_eq!(lock_entry.metadata, package.metadata);

        write(
            &repository.join("skills").join("demo").join("SKILL.md"),
            b"updated git skill",
        );
        run_test_git(&repository, &["add", "."]);
        run_test_git(&repository, &["commit", "-m", "update"]);
        let second_commit = run_test_git(&repository, &["rev-parse", "HEAD"])
            .trim()
            .to_owned();
        let pinned_spec = PackageSpec {
            source: PackageSource::Git {
                repository: repository.to_string_lossy().into_owned(),
                revision: Some(second_commit.clone()),
            },
            scope: PackageScope::Temporary,
            checksum: None,
            filter: PackageFilter::default(),
        };
        manager
            .reload(&[pinned_spec])
            .await
            .expect("resolve changed revision");
        let updated = &manager.snapshot().packages[0];
        let ResolvedSourceMetadata::Git { commit, .. } = &updated.metadata else {
            panic!("expected Git metadata");
        };
        assert_eq!(commit, &second_commit);
        assert_ne!(updated.checksum, initial_checksum);
    }

    #[tokio::test]
    async fn https_package_is_fail_closed_without_checksum_or_lock() {
        let temp = tempdir().expect("tempdir");
        let mut manager = PackageManager::new(options(temp.path()), GenerationClock::default());
        let manifest_url = Url::parse("https://example.invalid/ri-package.toml").expect("URL");
        let spec = PackageSpec {
            source: PackageSource::Https {
                manifest_url: manifest_url.clone(),
            },
            scope: PackageScope::Temporary,
            checksum: None,
            filter: PackageFilter::default(),
        };
        assert!(matches!(
            manager.reload(&[spec]).await,
            Err(PackageError::MissingChecksum(url)) if url == manifest_url
        ));
    }

    #[tokio::test]
    async fn checksum_mismatch_keeps_previous_generation() {
        let temp = tempdir().expect("tempdir");
        let package = temp.path().join("package");
        write(
            &package.join(MANIFEST_NAME),
            br#"
[package]
name = "demo"
version = "1"
"#,
        );
        let clock = GenerationClock::default();
        let before = clock.current();
        let mut manager = PackageManager::new(options(temp.path()), clock.clone());
        let mut spec = local_spec(package);
        spec.checksum = Some("00".repeat(32));
        assert!(matches!(
            manager.reload(&[spec]).await,
            Err(PackageError::ChecksumMismatch { .. })
        ));
        assert_eq!(clock.current(), before);
    }

    #[tokio::test]
    async fn project_packages_are_gated_by_trust() {
        let temp = tempdir().expect("tempdir");
        let mut options = options(temp.path());
        options.project_trusted = false;
        let mut manager = PackageManager::new(options, GenerationClock::default());
        let spec = PackageSpec {
            source: PackageSource::Local {
                path: PathBuf::from("package"),
            },
            scope: PackageScope::Project,
            checksum: None,
            filter: PackageFilter::default(),
        };
        assert!(matches!(
            manager.reload(&[spec]).await,
            Err(PackageError::UntrustedProject(_))
        ));
    }

    #[test]
    fn unsafe_manifest_paths_are_rejected() {
        let temp = tempdir().expect("tempdir");
        let manifest = PackageManifest::parse(
            r#"
[package]
name = "unsafe"
version = "1"

[resources]
prompts = ["../outside.md"]
"#,
        )
        .expect("parse");
        assert!(matches!(
            expand_local_resources(temp.path(), &manifest),
            Err(PackageError::UnsafePath(_))
        ));
    }

    #[test]
    fn lock_serialization_is_deterministic() {
        let mut lock = PackageLock::default();
        lock.packages.insert(
            "z".to_owned(),
            PackageLockEntry {
                source: PackageSource::Local {
                    path: PathBuf::from("z"),
                },
                scope: PackageScope::User,
                package_name: "z".to_owned(),
                package_version: "1".to_owned(),
                checksum: format!("sha256:{}", "00".repeat(32)),
                metadata: ResolvedSourceMetadata::Local {
                    canonical_path: PathBuf::from("z"),
                },
            },
        );
        lock.packages.insert(
            "a".to_owned(),
            PackageLockEntry {
                source: PackageSource::Local {
                    path: PathBuf::from("a"),
                },
                scope: PackageScope::User,
                package_name: "a".to_owned(),
                package_version: "1".to_owned(),
                checksum: format!("sha256:{}", "11".repeat(32)),
                metadata: ResolvedSourceMetadata::Local {
                    canonical_path: PathBuf::from("a"),
                },
            },
        );
        let serialized = lock.to_toml().expect("toml");
        assert!(
            serialized.find("[packages.a]").expect("a")
                < serialized.find("[packages.z]").expect("z")
        );
        assert_eq!(PackageLock::parse(&serialized).expect("parse"), lock);
    }

    #[test]
    fn project_specs_win_identity_deduplication() {
        let source = PackageSource::Git {
            repository: "https://example.invalid/repo.git".to_owned(),
            revision: None,
        };
        let user = PackageSpec {
            source: source.clone(),
            scope: PackageScope::User,
            checksum: None,
            filter: PackageFilter::default(),
        };
        let project = PackageSpec {
            source,
            scope: PackageScope::Project,
            checksum: None,
            filter: PackageFilter::default(),
        };
        assert_eq!(dedupe_specs(&[user, project.clone()]), vec![project]);
    }
}
