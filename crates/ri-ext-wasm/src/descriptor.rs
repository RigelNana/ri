//! Serializable manifests and descriptors at the native-extension boundary.

use crate::error::{HostError, Result};
use crate::policy::{CapabilityRequest, scope_contains};
use semver::Version;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

/// Current WIT package and world version.
pub const ABI_VERSION: &str = "1.0.0";

/// Current host-side manifest schema version.
pub const MANIFEST_SCHEMA_VERSION: u32 = 1;

const MAX_ID_BYTES: usize = 128;
const MAX_NAME_BYTES: usize = 256;
const MAX_DESCRIPTION_BYTES: usize = 16 * 1024;
const MAX_REGISTRATIONS: usize = 1_024;

/// Host-provided metadata used before untrusted component code is run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtensionManifest {
    /// Manifest format version.
    #[serde(default = "manifest_schema_version")]
    pub schema_version: u32,
    /// Stable extension ID.
    pub id: String,
    /// Extension semantic version.
    pub version: String,
    /// Required Component Model ABI version.
    pub abi_version: String,
    /// Capabilities the host may grant to this component.
    #[serde(default)]
    pub capabilities: Vec<CapabilityRequest>,
}

const fn manifest_schema_version() -> u32 {
    MANIFEST_SCHEMA_VERSION
}

impl ExtensionManifest {
    /// Parses a JSON manifest and validates all fields.
    ///
    /// # Errors
    ///
    /// Returns a typed validation error when JSON is malformed or a field is
    /// unsupported, inconsistent, or unsafe.
    pub fn from_json(json: &str) -> Result<Self> {
        let manifest: Self = serde_json::from_str(json)
            .map_err(|error| HostError::InvalidManifest(error.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    /// Validates this manifest without executing component code.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported schema or ABI, malformed identity,
    /// invalid version, duplicate capability, or invalid scope.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != MANIFEST_SCHEMA_VERSION {
            return Err(HostError::InvalidManifest(format!(
                "unsupported schema version {}; expected {}",
                self.schema_version, MANIFEST_SCHEMA_VERSION
            )));
        }
        validate_identifier("manifest id", &self.id)?;
        validate_semver(
            "manifest version",
            &self.version,
            HostError::InvalidManifest,
        )?;
        if self.abi_version != ABI_VERSION {
            return Err(HostError::UnsupportedAbi {
                found: self.abi_version.clone(),
                expected: ABI_VERSION,
            });
        }
        if self.capabilities.len() > MAX_REGISTRATIONS {
            return Err(HostError::InvalidManifest(format!(
                "manifest has more than {MAX_REGISTRATIONS} capability requests"
            )));
        }

        let mut unique = BTreeSet::new();
        for capability in &self.capabilities {
            capability.validate()?;
            let canonical_scope = canonical_json(&capability.scope);
            if !unique.insert((capability.kind, canonical_scope)) {
                return Err(HostError::InvalidManifest(format!(
                    "duplicate {} capability scope",
                    capability.kind
                )));
            }
        }
        Ok(())
    }
}

/// Tool metadata published by an extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRegistration {
    /// Tool-local identifier.
    pub id: String,
    /// User-facing title.
    pub title: String,
    /// User-facing description.
    pub description: String,
    /// JSON Schema for tool input.
    pub input_schema_json: String,
    /// Optional JSON Schema for tool output.
    pub output_schema_json: Option<String>,
}

/// Where a command should be exposed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CommandPlacement {
    /// General command palette.
    Palette,
    /// Context-specific menu.
    ContextMenu,
    /// Slash-command menu.
    SlashMenu,
    /// Programmatic use only.
    Hidden,
}

/// Command metadata published by an extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommandRegistration {
    /// Command-local identifier.
    pub id: String,
    /// User-facing title.
    pub title: String,
    /// User-facing description.
    pub description: String,
    /// Requested command placement.
    pub placement: CommandPlacement,
    /// Optional JSON Schema for command arguments.
    pub argument_schema_json: Option<String>,
}

/// Supported host locations for declarative views.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ViewLocation {
    /// Sidebar region.
    Sidebar,
    /// Bottom or auxiliary panel.
    Panel,
    /// Main editor region.
    Editor,
    /// Modal overlay.
    Modal,
}

/// Declarative view metadata published by an extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewRegistration {
    /// View-local identifier.
    pub id: String,
    /// User-facing title.
    pub title: String,
    /// Requested host location.
    pub location: ViewLocation,
}

/// Validated metadata returned by the guest's `descriptor` export.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtensionDescriptor {
    /// Stable extension ID.
    pub id: String,
    /// User-facing extension name.
    pub name: String,
    /// Extension semantic version.
    pub version: String,
    /// Component Model ABI version.
    pub abi_version: String,
    /// Optional longer description.
    pub description: Option<String>,
    /// Capabilities declared by the component.
    pub capabilities: Vec<CapabilityRequest>,
    /// Tools implemented by the component.
    pub tools: Vec<ToolRegistration>,
    /// Commands implemented by the component.
    pub commands: Vec<CommandRegistration>,
    /// Declarative views implemented by the component.
    pub views: Vec<ViewRegistration>,
}

impl ExtensionDescriptor {
    /// Validates descriptor structure and JSON schema fields.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata, capability declarations,
    /// registrations, or embedded JSON schemas are invalid.
    pub fn validate(&self) -> Result<()> {
        validate_identifier("descriptor id", &self.id)
            .map_err(|error| descriptor_error(error.to_string()))?;
        validate_semver(
            "descriptor version",
            &self.version,
            HostError::InvalidDescriptor,
        )?;
        if self.abi_version != ABI_VERSION {
            return Err(HostError::UnsupportedAbi {
                found: self.abi_version.clone(),
                expected: ABI_VERSION,
            });
        }
        validate_text("descriptor name", &self.name, 1, MAX_NAME_BYTES)?;
        if let Some(description) = &self.description {
            validate_text(
                "descriptor description",
                description,
                0,
                MAX_DESCRIPTION_BYTES,
            )?;
        }
        if self.capabilities.len() > MAX_REGISTRATIONS {
            return Err(descriptor_error(format!(
                "descriptor has more than {MAX_REGISTRATIONS} capability requests"
            )));
        }
        let mut capability_scopes = BTreeSet::new();
        for capability in &self.capabilities {
            capability
                .validate()
                .map_err(|error| descriptor_error(error.to_string()))?;
            if !capability_scopes.insert((capability.kind, canonical_json(&capability.scope))) {
                return Err(descriptor_error(format!(
                    "duplicate {} capability scope",
                    capability.kind
                )));
            }
        }
        validate_registrations(self)?;
        Ok(())
    }

    /// Ensures the untrusted descriptor cannot contradict or escalate the
    /// pre-validated manifest.
    ///
    /// # Errors
    ///
    /// Returns an error when descriptor identity or ABI metadata differs from
    /// the manifest, or a capability exceeds its manifest declaration.
    pub fn validate_against(&self, manifest: &ExtensionManifest) -> Result<()> {
        self.validate()?;
        if self.id != manifest.id {
            return Err(descriptor_error(format!(
                "descriptor id `{}` does not match manifest id `{}`",
                self.id, manifest.id
            )));
        }
        if self.version != manifest.version {
            return Err(descriptor_error(format!(
                "descriptor version `{}` does not match manifest version `{}`",
                self.version, manifest.version
            )));
        }
        if self.abi_version != manifest.abi_version {
            return Err(descriptor_error(format!(
                "descriptor ABI `{}` does not match manifest ABI `{}`",
                self.abi_version, manifest.abi_version
            )));
        }

        for requested in &self.capabilities {
            let declared = manifest.capabilities.iter().any(|manifest_request| {
                manifest_request.kind == requested.kind
                    && (!requested.required || manifest_request.required)
                    && scope_contains(requested.kind, &manifest_request.scope, &requested.scope)
            });
            if !declared {
                return Err(descriptor_error(format!(
                    "descriptor escalates undeclared {} capability scope",
                    requested.kind
                )));
            }
        }
        Ok(())
    }
}

fn validate_registrations(descriptor: &ExtensionDescriptor) -> Result<()> {
    if descriptor.tools.len() + descriptor.commands.len() + descriptor.views.len()
        > MAX_REGISTRATIONS
    {
        return Err(descriptor_error(format!(
            "descriptor has more than {MAX_REGISTRATIONS} registrations"
        )));
    }

    let mut ids = BTreeSet::new();
    for tool in &descriptor.tools {
        validate_registration_text("tool", &tool.id, &tool.title, &tool.description)?;
        insert_unique(&mut ids, "tool", &tool.id)?;
        validate_json_schema("tool input", &tool.input_schema_json)?;
        if let Some(schema) = &tool.output_schema_json {
            validate_json_schema("tool output", schema)?;
        }
    }
    for command in &descriptor.commands {
        validate_registration_text("command", &command.id, &command.title, &command.description)?;
        insert_unique(&mut ids, "command", &command.id)?;
        if let Some(schema) = &command.argument_schema_json {
            validate_json_schema("command argument", schema)?;
        }
    }
    for view in &descriptor.views {
        validate_identifier("view id", &view.id)
            .map_err(|error| descriptor_error(error.to_string()))?;
        validate_text("view title", &view.title, 1, MAX_NAME_BYTES)?;
        insert_unique(&mut ids, "view", &view.id)?;
    }
    Ok(())
}

fn validate_registration_text(kind: &str, id: &str, title: &str, description: &str) -> Result<()> {
    validate_identifier(&format!("{kind} id"), id)
        .map_err(|error| descriptor_error(error.to_string()))?;
    validate_text(&format!("{kind} title"), title, 1, MAX_NAME_BYTES)?;
    validate_text(
        &format!("{kind} description"),
        description,
        0,
        MAX_DESCRIPTION_BYTES,
    )
}

fn insert_unique(ids: &mut BTreeSet<String>, kind: &str, id: &str) -> Result<()> {
    if !ids.insert(id.to_owned()) {
        return Err(descriptor_error(format!(
            "duplicate registration id `{id}` encountered at {kind}"
        )));
    }
    Ok(())
}

fn validate_json_schema(label: &str, schema: &str) -> Result<()> {
    let value: Value = serde_json::from_str(schema)
        .map_err(|error| descriptor_error(format!("{label} schema is invalid JSON: {error}")))?;
    if !value.is_object() && !value.is_boolean() {
        return Err(descriptor_error(format!(
            "{label} schema must be a JSON object or boolean"
        )));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > MAX_ID_BYTES {
        return Err(HostError::InvalidManifest(format!(
            "{label} must contain 1..={MAX_ID_BYTES} bytes"
        )));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
        || !value
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
    {
        return Err(HostError::InvalidManifest(format!(
            "{label} `{value}` contains unsupported characters"
        )));
    }
    Ok(())
}

fn validate_text(label: &str, value: &str, min: usize, max: usize) -> Result<()> {
    if value.len() < min || value.len() > max || value.contains('\0') {
        return Err(descriptor_error(format!(
            "{label} must contain {min}..={max} bytes and no NUL characters"
        )));
    }
    Ok(())
}

fn validate_semver(label: &str, value: &str, error: fn(String) -> HostError) -> Result<()> {
    if Version::parse(value).is_err() {
        return Err(error(format!(
            "{label} `{value}` is not a supported semantic version"
        )));
    }
    Ok(())
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<_> = object.iter().collect();
            entries.sort_by_key(|(key, _)| *key);
            let body = entries
                .into_iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("string serialization is infallible"),
                        canonical_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        Value::Array(values) => {
            let body = values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",");
            format!("[{body}]")
        }
        _ => value.to_string(),
    }
}

fn descriptor_error(message: String) -> HostError {
    HostError::InvalidDescriptor(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::CapabilityKind;
    use serde_json::json;

    fn valid_manifest() -> ExtensionManifest {
        ExtensionManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            id: "dev.ri.example".to_owned(),
            version: "1.2.3".to_owned(),
            abi_version: ABI_VERSION.to_owned(),
            capabilities: Vec::new(),
        }
    }

    fn valid_descriptor() -> ExtensionDescriptor {
        ExtensionDescriptor {
            id: "dev.ri.example".to_owned(),
            name: "Example".to_owned(),
            version: "1.2.3".to_owned(),
            abi_version: ABI_VERSION.to_owned(),
            description: None,
            capabilities: Vec::new(),
            tools: vec![ToolRegistration {
                id: "echo".to_owned(),
                title: "Echo".to_owned(),
                description: "Returns its input".to_owned(),
                input_schema_json: r#"{"type":"object"}"#.to_owned(),
                output_schema_json: None,
            }],
            commands: Vec::new(),
            views: Vec::new(),
        }
    }

    #[test]
    fn manifest_rejects_unknown_abi() {
        let mut manifest = valid_manifest();
        manifest.abi_version = "2.0.0".to_owned();
        assert!(matches!(
            manifest.validate(),
            Err(HostError::UnsupportedAbi { .. })
        ));
    }

    #[test]
    fn manifest_rejects_noncanonical_semver() {
        let mut manifest = valid_manifest();
        manifest.version = "1.0.0-01".to_owned();
        assert!(matches!(
            manifest.validate(),
            Err(HostError::InvalidManifest(_))
        ));
    }

    #[test]
    fn descriptor_rejects_duplicate_registration_ids() {
        let mut descriptor = valid_descriptor();
        descriptor.commands.push(CommandRegistration {
            id: "echo".to_owned(),
            title: "Echo".to_owned(),
            description: String::new(),
            placement: CommandPlacement::Palette,
            argument_schema_json: None,
        });
        assert!(matches!(
            descriptor.validate(),
            Err(HostError::InvalidDescriptor(_))
        ));
    }

    #[test]
    fn descriptor_cannot_escalate_manifest_capabilities() {
        let mut descriptor = valid_descriptor();
        descriptor.capabilities.push(
            CapabilityRequest::new(
                CapabilityKind::Ui,
                json!({"surfaces": ["notification"]}),
                true,
            )
            .expect("valid capability"),
        );
        assert!(matches!(
            descriptor.validate_against(&valid_manifest()),
            Err(HostError::InvalidDescriptor(_))
        ));
    }

    #[test]
    fn manifest_json_rejects_unknown_fields() {
        let json = r#"{
            "id":"dev.ri.example",
            "version":"1.0.0",
            "abi_version":"1.0.0",
            "capabilities":[],
            "ambient":true
        }"#;
        assert!(matches!(
            ExtensionManifest::from_json(json),
            Err(HostError::InvalidManifest(_))
        ));
    }
}
