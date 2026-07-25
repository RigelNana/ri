//! Capability declarations and deny-by-default policy evaluation.

use crate::error::{HostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt;
use url::{Origin, Url};

/// The six host service categories exposed by the Component Model ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityKind {
    /// Scoped filesystem access.
    Filesystem,
    /// Scoped outbound HTTP access.
    Network,
    /// Scoped child-process execution.
    Process,
    /// User-interface interactions.
    Ui,
    /// Session-scoped key/value access.
    Session,
    /// Model/provider calls.
    Provider,
}

impl fmt::Display for CapabilityKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Filesystem => "filesystem",
            Self::Network => "network",
            Self::Process => "process",
            Self::Ui => "ui",
            Self::Session => "session",
            Self::Provider => "provider",
        })
    }
}

/// A capability and the JSON scope requested by an extension.
///
/// Scope objects intentionally remain JSON at the `ri-ext` boundary. The WIT
/// constructors use typed scope records; the host converts those records into
/// this canonical representation before evaluating policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityRequest {
    /// Capability category.
    pub kind: CapabilityKind,
    /// Category-specific scope object.
    pub scope: Value,
    /// Whether load must fail when policy does not grant this request.
    #[serde(default = "required_by_default")]
    pub required: bool,
}

const fn required_by_default() -> bool {
    true
}

impl CapabilityRequest {
    /// Constructs and validates a capability request.
    ///
    /// # Errors
    ///
    /// Returns an invalid-manifest error when `scope` does not match the
    /// category-specific schema.
    pub fn new(kind: CapabilityKind, scope: Value, required: bool) -> Result<Self> {
        validate_scope(kind, &scope)?;
        Ok(Self {
            kind,
            scope,
            required,
        })
    }

    /// Validates the category-specific scope.
    ///
    /// # Errors
    ///
    /// Returns an invalid-manifest error for malformed or unsafe scope fields.
    pub fn validate(&self) -> Result<()> {
        validate_scope(self.kind, &self.scope)
    }
}

/// A host policy rule for a single capability kind.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "scope", rename_all = "kebab-case")]
pub enum ScopeRule {
    /// Allow every well-formed scope for this capability kind.
    Any,
    /// Allow requests contained by this scope.
    Scoped(Value),
}

/// Deny-by-default policy applied before a component is instantiated.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CapabilityPolicy {
    rules: BTreeMap<CapabilityKind, Vec<ScopeRule>>,
}

impl CapabilityPolicy {
    /// Returns a policy that grants no capabilities.
    pub fn deny_all() -> Self {
        Self::default()
    }

    /// Adds an unrestricted rule for `kind`.
    #[must_use]
    pub fn allow_any(mut self, kind: CapabilityKind) -> Self {
        self.rules.entry(kind).or_default().push(ScopeRule::Any);
        self
    }

    /// Adds a validated scoped rule for `kind`.
    ///
    /// # Errors
    ///
    /// Returns an invalid-manifest error when the policy scope is malformed.
    pub fn allow_scope(mut self, kind: CapabilityKind, scope: Value) -> Result<Self> {
        validate_scope(kind, &scope)?;
        self.rules
            .entry(kind)
            .or_default()
            .push(ScopeRule::Scoped(scope));
        Ok(self)
    }

    /// Returns whether a single request is permitted.
    pub fn permits(&self, request: &CapabilityRequest) -> bool {
        self.rules.get(&request.kind).is_some_and(|rules| {
            rules.iter().any(|rule| match rule {
                ScopeRule::Any => true,
                ScopeRule::Scoped(scope) => scope_contains(request.kind, scope, &request.scope),
            })
        })
    }

    /// Resolves manifest requests into the set granted to an instance.
    ///
    /// A denied required request fails the load. A denied optional request is
    /// omitted, so its resource constructor returns a typed denial at runtime.
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid request or
    /// [`HostError::CapabilityDenied`] for a required request not covered by a
    /// policy rule.
    pub fn authorize(&self, requests: &[CapabilityRequest]) -> Result<GrantedCapabilities> {
        let mut granted = Vec::new();
        for request in requests {
            request.validate()?;
            if self.permits(request) {
                granted.push(request.clone());
            } else if request.required {
                return Err(HostError::CapabilityDenied {
                    kind: request.kind,
                    reason: "no policy rule contains the requested scope".to_owned(),
                });
            }
        }
        Ok(GrantedCapabilities { grants: granted })
    }
}

/// Capabilities actually granted to one component instance.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GrantedCapabilities {
    grants: Vec<CapabilityRequest>,
}

impl GrantedCapabilities {
    /// Returns the effective grants.
    pub fn as_slice(&self) -> &[CapabilityRequest] {
        &self.grants
    }

    /// Checks a typed resource-constructor scope against effective grants.
    pub fn permits(&self, kind: CapabilityKind, scope: &Value) -> bool {
        self.grants
            .iter()
            .filter(|grant| grant.kind == kind)
            .any(|grant| scope_contains(kind, &grant.scope, scope))
    }
}

/// Validates a canonical scope object for a capability kind.
///
/// # Errors
///
/// Returns an invalid-manifest error when required scope fields are missing,
/// malformed, unsafe, or outside the supported category schema.
pub fn validate_scope(kind: CapabilityKind, scope: &Value) -> Result<()> {
    let object = scope.as_object().ok_or_else(|| {
        HostError::InvalidManifest(format!("{kind} capability scope must be a JSON object"))
    })?;

    match kind {
        CapabilityKind::Filesystem => {
            nonempty_string_array(object, "roots", kind)?;
            enum_string(object, "access", &["read", "read-write"], kind)?;
            positive_integer(object, "max-read-bytes", kind)?;
            positive_integer(object, "max-write-bytes", kind)?;
            for root in string_array(object, "roots", kind)? {
                validate_root(root)?;
            }
        }
        CapabilityKind::Network => {
            nonempty_string_array(object, "origins", kind)?;
            nonempty_string_array(object, "methods", kind)?;
            positive_integer(object, "max-response-bytes", kind)?;
            for origin in string_array(object, "origins", kind)? {
                if origin != "*" && !is_canonical_http_origin(origin) {
                    return invalid_scope(
                        kind,
                        format!("origin `{origin}` must be a canonical HTTP(S) origin or `*`"),
                    );
                }
            }
            for method in string_array(object, "methods", kind)? {
                if method != "*" && method.to_ascii_uppercase() != method {
                    return invalid_scope(
                        kind,
                        format!("method `{method}` must be uppercase or `*`"),
                    );
                }
            }
        }
        CapabilityKind::Process => {
            nonempty_string_array(object, "programs", kind)?;
            positive_integer(object, "max-runtime-ms", kind)?;
            positive_integer(object, "max-output-bytes", kind)?;
            boolean(object, "allow-environment", kind)?;
        }
        CapabilityKind::Ui => {
            nonempty_string_array(object, "surfaces", kind)?;
            let allowed = ["notification", "prompt", "clipboard", "external-uri"];
            for surface in string_array(object, "surfaces", kind)? {
                if !allowed.contains(&surface) {
                    return invalid_scope(kind, format!("unknown UI surface `{surface}`"));
                }
            }
        }
        CapabilityKind::Session => {
            nonempty_string_array(object, "namespaces", kind)?;
            boolean(object, "writable", kind)?;
        }
        CapabilityKind::Provider => {
            nonempty_string_array(object, "providers", kind)?;
            let _ = string_array(object, "models", kind)?;
            boolean(object, "allow-streaming", kind)?;
        }
    }
    Ok(())
}

fn is_canonical_http_origin(value: &str) -> bool {
    let Ok(url) = Url::parse(value) else {
        return false;
    };
    if !matches!(url.scheme(), "http" | "https")
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return false;
    }
    let Origin::Tuple(..) = url.origin() else {
        return false;
    };
    url.origin()
        .ascii_serialization()
        .eq_ignore_ascii_case(value.trim_end_matches('/'))
}

/// Returns whether `requested` is no broader than `allowed`.
pub fn scope_contains(kind: CapabilityKind, allowed: &Value, requested: &Value) -> bool {
    let (Some(allowed), Some(requested)) = (allowed.as_object(), requested.as_object()) else {
        return false;
    };

    requested.iter().all(|(key, requested_value)| {
        let Some(allowed_value) = allowed.get(key) else {
            return false;
        };
        if kind == CapabilityKind::Filesystem && key == "roots" {
            return filesystem_roots_contain(allowed_value, requested_value);
        }
        if kind == CapabilityKind::Filesystem && key == "access" {
            return access_contains(allowed_value, requested_value);
        }
        if allowed_value.is_array() && requested_value.is_array() {
            let case_insensitive = matches!(kind, CapabilityKind::Network | CapabilityKind::Ui)
                || kind == CapabilityKind::Process && cfg!(windows);
            return string_array_contains(allowed_value, requested_value, case_insensitive);
        }
        json_contains(allowed_value, requested_value)
    })
}

fn string_array_contains(allowed: &Value, requested: &Value, case_insensitive: bool) -> bool {
    let (Some(allowed), Some(requested)) = (allowed.as_array(), requested.as_array()) else {
        return false;
    };
    requested.iter().all(|requested| {
        requested.as_str().is_some_and(|requested| {
            allowed.iter().filter_map(Value::as_str).any(|allowed| {
                allowed == "*"
                    || if case_insensitive {
                        allowed.eq_ignore_ascii_case(requested)
                    } else {
                        allowed == requested
                    }
            })
        })
    })
}

fn json_contains(allowed: &Value, requested: &Value) -> bool {
    match (allowed, requested) {
        (Value::String(allowed), Value::String(requested)) => {
            allowed == "*" || allowed == requested
        }
        (Value::Bool(allowed), Value::Bool(requested)) => *allowed || !*requested,
        (Value::Number(allowed), Value::Number(requested)) => allowed
            .as_u64()
            .zip(requested.as_u64())
            .is_some_and(|(allowed, requested)| requested <= allowed),
        (Value::Array(_), Value::Array(_)) => string_array_contains(allowed, requested, false),
        (Value::Object(allowed), Value::Object(requested)) => {
            requested.iter().all(|(key, requested)| {
                allowed
                    .get(key)
                    .is_some_and(|allowed| json_contains(allowed, requested))
            })
        }
        _ => allowed == requested,
    }
}

fn filesystem_roots_contain(allowed: &Value, requested: &Value) -> bool {
    let Some(allowed) = allowed.as_array() else {
        return false;
    };
    let Some(requested) = requested.as_array() else {
        return false;
    };
    requested.iter().all(|requested| {
        requested.as_str().is_some_and(|requested| {
            allowed.iter().any(|allowed| {
                allowed
                    .as_str()
                    .is_some_and(|allowed| path_is_within(allowed, requested))
            })
        })
    })
}

fn access_contains(allowed: &Value, requested: &Value) -> bool {
    let (Some(allowed), Some(requested)) = (allowed.as_str(), requested.as_str()) else {
        return false;
    };
    allowed == "read-write" && matches!(requested, "read-write" | "read")
        || allowed == "read" && requested == "read"
}

fn path_is_within(root: &str, candidate: &str) -> bool {
    if root == "*" {
        return true;
    }
    let root = normalize_path(root);
    let candidate = normalize_path(candidate);
    candidate == root
        || candidate
            .strip_prefix(&root)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

fn normalize_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    let normalized = normalized
        .strip_suffix('/')
        .unwrap_or(&normalized)
        .to_owned();
    if cfg!(windows) {
        normalized.to_ascii_lowercase()
    } else {
        normalized
    }
}

fn validate_root(root: &str) -> Result<()> {
    if root == "*" {
        return Ok(());
    }
    if root.contains('\0') || root.split(['/', '\\']).any(|part| part == "..") {
        return Err(HostError::InvalidManifest(format!(
            "filesystem root `{root}` is not a safe lexical root"
        )));
    }
    let bytes = root.as_bytes();
    let rooted = root.starts_with('/')
        || root.starts_with("file://")
        || (bytes.len() >= 3
            && bytes[0].is_ascii_alphabetic()
            && bytes[1] == b':'
            && matches!(bytes[2], b'/' | b'\\'));
    if !rooted {
        return Err(HostError::InvalidManifest(format!(
            "filesystem root `{root}` must be absolute"
        )));
    }
    Ok(())
}

fn string_array<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    kind: CapabilityKind,
) -> Result<Vec<&'a str>> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| {
            HostError::InvalidManifest(format!("{kind} scope field `{key}` must be an array"))
        })?
        .iter()
        .map(|value| {
            value
                .as_str()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    HostError::InvalidManifest(format!(
                        "{kind} scope field `{key}` must contain non-empty strings"
                    ))
                })
        })
        .collect()
}

fn nonempty_string_array(
    object: &Map<String, Value>,
    key: &str,
    kind: CapabilityKind,
) -> Result<()> {
    if string_array(object, key, kind)?.is_empty() {
        return invalid_scope(kind, format!("field `{key}` must not be empty"));
    }
    Ok(())
}

fn enum_string(
    object: &Map<String, Value>,
    key: &str,
    values: &[&str],
    kind: CapabilityKind,
) -> Result<()> {
    let value = object.get(key).and_then(Value::as_str).ok_or_else(|| {
        HostError::InvalidManifest(format!("{kind} scope field `{key}` must be a string"))
    })?;
    if !values.contains(&value) {
        return invalid_scope(kind, format!("field `{key}` has invalid value `{value}`"));
    }
    Ok(())
}

fn positive_integer(object: &Map<String, Value>, key: &str, kind: CapabilityKind) -> Result<()> {
    if object
        .get(key)
        .and_then(Value::as_u64)
        .is_none_or(|value| value == 0)
    {
        return invalid_scope(kind, format!("field `{key}` must be a positive integer"));
    }
    Ok(())
}

fn boolean(object: &Map<String, Value>, key: &str, kind: CapabilityKind) -> Result<()> {
    if object.get(key).and_then(Value::as_bool).is_none() {
        return invalid_scope(kind, format!("field `{key}` must be a boolean"));
    }
    Ok(())
}

fn invalid_scope<T>(kind: CapabilityKind, message: impl AsRef<str>) -> Result<T> {
    Err(HostError::InvalidManifest(format!(
        "invalid {kind} capability scope: {}",
        message.as_ref()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn filesystem_scope_cannot_escalate_access_or_root() {
        let allowed = json!({
            "roots": ["C:/workspace"],
            "access": "read",
            "max-read-bytes": 1024,
            "max-write-bytes": 1
        });
        let inside = json!({
            "roots": ["C:/workspace/subdir"],
            "access": "read",
            "max-read-bytes": 512,
            "max-write-bytes": 1
        });
        let write = json!({
            "roots": ["C:/workspace"],
            "access": "read-write",
            "max-read-bytes": 512,
            "max-write-bytes": 1
        });
        let outside = json!({
            "roots": ["C:/other"],
            "access": "read",
            "max-read-bytes": 512,
            "max-write-bytes": 1
        });

        assert!(scope_contains(
            CapabilityKind::Filesystem,
            &allowed,
            &inside
        ));
        assert!(!scope_contains(
            CapabilityKind::Filesystem,
            &allowed,
            &write
        ));
        assert!(!scope_contains(
            CapabilityKind::Filesystem,
            &allowed,
            &outside
        ));
    }

    #[test]
    fn denied_required_capability_fails_authorization() {
        let request = CapabilityRequest::new(
            CapabilityKind::Network,
            json!({
                "origins": ["https://example.com"],
                "methods": ["GET"],
                "max-response-bytes": 1024
            }),
            true,
        )
        .expect("valid request");

        let error = CapabilityPolicy::deny_all()
            .authorize(&[request])
            .expect_err("deny-all policy must reject required network access");
        assert!(matches!(
            error,
            HostError::CapabilityDenied {
                kind: CapabilityKind::Network,
                ..
            }
        ));
    }

    #[test]
    fn denied_optional_capability_is_not_granted() {
        let request = CapabilityRequest::new(
            CapabilityKind::Ui,
            json!({"surfaces": ["notification"]}),
            false,
        )
        .expect("valid request");
        let grants = CapabilityPolicy::deny_all()
            .authorize(&[request])
            .expect("optional request may be omitted");
        assert!(grants.as_slice().is_empty());
    }

    #[test]
    fn case_sensitive_scope_identifiers_cannot_escalate() {
        let allowed = json!({"namespaces": ["Private"], "writable": false});
        let requested = json!({"namespaces": ["private"], "writable": false});
        assert!(!scope_contains(
            CapabilityKind::Session,
            &allowed,
            &requested
        ));
    }
}
