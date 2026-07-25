//! Tool schemas, validation, constrained sampling, and deferred loading.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Number, Value};
use thiserror::Error;

use crate::message::{ContentBlock, Context, Message, ToolCall};

pub mod partial_json;

/// Strictness requested for JSON-schema constrained sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JsonSchemaStrictness {
    /// Use strict sampling when the provider supports it.
    Prefer,
    /// Reject adapters that cannot provide strict sampling.
    Require,
}

/// Provider grammar encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrammarFormat {
    /// `OpenAI` Lark grammar.
    Lark,
    /// `OpenAI` regular-expression grammar.
    Regex,
}

/// Provider-specific encodings of one grammar.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct GrammarVariants {
    /// `OpenAI` Lark definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_lark: Option<String>,
    /// `OpenAI` regex definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openai_regex: Option<String>,
}

/// Optional provider-side constrained sampling for a tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstrainedSampling {
    /// Strict function argument generation.
    JsonSchema {
        /// Required support behavior.
        strict: JsonSchemaStrictness,
    },
    /// `OpenAI` custom grammar tool.
    Grammar {
        /// Supported grammar encodings.
        variants: GrammarVariants,
    },
}

/// Provider-neutral tool definition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Tool {
    /// Function name.
    pub name: String,
    /// Human-readable purpose.
    pub description: String,
    /// JSON Schema for arguments.
    pub parameters: Value,
    /// Optional constrained sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrained_sampling: Option<ConstrainedSampling>,
}

impl Tool {
    /// Creates an unconstrained function tool.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            constrained_sampling: None,
        }
    }
}

/// Wire-ready tool descriptor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolDescriptor {
    /// JSON-schema function tool.
    Function {
        /// Tool name.
        name: String,
        /// Description.
        description: String,
        /// Argument schema.
        parameters: Value,
        /// Strict JSON-schema decoding when supported.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
        /// Defer definition loading.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        defer_loading: bool,
    },
    /// `OpenAI` custom grammar tool.
    Custom {
        /// Tool name.
        name: String,
        /// Description.
        description: String,
        /// Grammar syntax.
        format: GrammarFormat,
        /// Grammar source.
        definition: String,
        /// The sole required string property represented by the grammar.
        input_property: String,
        /// Defer definition loading.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        defer_loading: bool,
    },
}

/// Tool validation failure.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ToolValidationError {
    /// No matching tool exists.
    #[error("tool \"{0}\" not found")]
    NotFound(String),
    /// Schema itself is invalid.
    #[error("invalid schema for tool \"{tool}\": {message}")]
    InvalidSchema {
        /// Tool name.
        tool: String,
        /// Schema compiler error.
        message: String,
    },
    /// Arguments do not satisfy the schema after coercion.
    #[error("validation failed for tool \"{tool}\": {errors:?}; received {received}")]
    InvalidArguments {
        /// Tool name.
        tool: String,
        /// Human-readable validation errors.
        errors: Vec<String>,
        /// Original arguments.
        received: Value,
    },
    /// Constrained sampling cannot be represented by the provider.
    #[error("{0}")]
    UnsupportedConstraint(String),
}

/// Finds a tool by name, coerces arguments using AJV-compatible primitive
/// rules, and validates the result against JSON Schema.
///
/// # Errors
///
/// Returns an error when the named tool is absent, its schema cannot be
/// compiled, or its arguments remain invalid after coercion.
pub fn validate_tool_call(tools: &[Tool], call: &ToolCall) -> Result<Value, ToolValidationError> {
    let tool = tools
        .iter()
        .find(|tool| tool.name == call.name)
        .ok_or_else(|| ToolValidationError::NotFound(call.name.clone()))?;
    validate_tool_arguments(tool, call)
}

/// Coerces and validates one tool call.
///
/// # Errors
///
/// Returns an error when the tool schema cannot be compiled or the arguments
/// remain invalid after coercion.
pub fn validate_tool_arguments(tool: &Tool, call: &ToolCall) -> Result<Value, ToolValidationError> {
    let original = call.arguments.clone();
    let coerced = coerce_with_schema(original.clone(), &tool.parameters);
    let validator = jsonschema::validator_for(&tool.parameters).map_err(|error| {
        ToolValidationError::InvalidSchema {
            tool: tool.name.clone(),
            message: error.to_string(),
        }
    })?;
    let errors = validator
        .iter_errors(&coerced)
        .map(|error| {
            let path = error.instance_path().to_string();
            let path = path.trim_start_matches('/').replace('/', ".");
            format!("{}: {error}", if path.is_empty() { "root" } else { &path })
        })
        .collect::<Vec<_>>();
    if errors.is_empty() {
        Ok(coerced)
    } else {
        Err(ToolValidationError::InvalidArguments {
            tool: tool.name.clone(),
            errors,
            received: original,
        })
    }
}

fn schema_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind],
        Some(Value::Array(kinds)) => kinds.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_json_type(value: &Value, kind: &str) -> bool {
    match kind {
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn coerce_primitive(value: &Value, kind: &str) -> Option<Value> {
    match kind {
        "number" => match value {
            Value::Null => Some(Value::Number(Number::from(0))),
            Value::Bool(value) => Some(Value::Number(Number::from(u8::from(*value)))),
            Value::String(value) if !value.trim().is_empty() => value
                .parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .and_then(Number::from_f64)
                .map(Value::Number),
            _ => None,
        },
        "integer" => match value {
            Value::Null => Some(Value::Number(Number::from(0))),
            Value::Bool(value) => Some(Value::Number(Number::from(u8::from(*value)))),
            Value::String(value) if !value.trim().is_empty() => value
                .parse::<i64>()
                .ok()
                .map(Number::from)
                .map(Value::Number),
            _ => None,
        },
        "boolean" => match value {
            Value::Null => Some(Value::Bool(false)),
            Value::String(value) if value == "true" => Some(Value::Bool(true)),
            Value::String(value) if value == "false" => Some(Value::Bool(false)),
            Value::Number(value) if value.as_i64() == Some(1) => Some(Value::Bool(true)),
            Value::Number(value) if value.as_i64() == Some(0) => Some(Value::Bool(false)),
            _ => None,
        },
        "string" => match value {
            Value::Null => Some(Value::String(String::new())),
            Value::Bool(value) => Some(Value::String(value.to_string())),
            Value::Number(value) => Some(Value::String(value.to_string())),
            _ => None,
        },
        "null"
            if matches!(value, Value::String(text) if text.is_empty())
                || value.as_i64() == Some(0)
                || value.as_bool() == Some(false) =>
        {
            Some(Value::Null)
        }
        _ => None,
    }
}

fn schema_valid(schema: &Value, value: &Value) -> bool {
    jsonschema::validator_for(schema).is_ok_and(|validator| validator.is_valid(value))
}

fn coerce_union(value: Value, schemas: &[Value]) -> Value {
    for schema in schemas {
        let candidate = coerce_with_schema(value.clone(), schema);
        if schema_valid(schema, &candidate) {
            return candidate;
        }
    }
    value
}

fn coerce_with_schema(mut value: Value, schema: &Value) -> Value {
    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        for nested in all_of {
            value = coerce_with_schema(value, nested);
        }
    }
    if let Some(any_of) = schema.get("anyOf").and_then(Value::as_array) {
        value = coerce_union(value, any_of);
    }
    if let Some(one_of) = schema.get("oneOf").and_then(Value::as_array) {
        value = coerce_union(value, one_of);
    }

    let types = schema_types(schema);
    let union_already_matches =
        types.len() > 1 && types.iter().any(|kind| matches_json_type(&value, kind));
    if !types.is_empty() && !union_already_matches {
        for kind in &types {
            if let Some(coerced) = coerce_primitive(&value, kind) {
                value = coerced;
                break;
            }
        }
    }

    if types.contains(&"object")
        && let Value::Object(object) = &mut value
    {
        coerce_object(object, schema);
    }
    if types.contains(&"array")
        && let Value::Array(array) = &mut value
    {
        coerce_array(array, schema);
    }
    value
}

fn coerce_object(object: &mut Map<String, Value>, schema: &Value) {
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (name, property_schema) in properties {
            if let Some(value) = object.get_mut(name) {
                *value = coerce_with_schema(std::mem::take(value), property_schema);
            }
        }
    }
    if let Some(additional) = schema.get("additionalProperties")
        && additional.is_object()
    {
        for (name, value) in object {
            if properties.is_some_and(|properties| properties.contains_key(name)) {
                continue;
            }
            *value = coerce_with_schema(std::mem::take(value), additional);
        }
    }
}

fn coerce_array(array: &mut [Value], schema: &Value) {
    let Some(items) = schema.get("items") else {
        return;
    };
    if let Some(tuple) = items.as_array() {
        for (value, item_schema) in array.iter_mut().zip(tuple) {
            *value = coerce_with_schema(std::mem::take(value), item_schema);
        }
    } else if items.is_object() {
        for value in array {
            *value = coerce_with_schema(std::mem::take(value), items);
        }
    }
}

/// Resolves JSON-schema strict mode for an adapter.
///
/// # Errors
///
/// Returns [`ToolValidationError::UnsupportedConstraint`] when strict sampling
/// is required but unsupported by the target adapter.
pub fn resolve_json_schema_strict(
    tool: &Tool,
    supports_strict_mode: bool,
) -> Result<Option<bool>, ToolValidationError> {
    let Some(ConstrainedSampling::JsonSchema { strict }) = &tool.constrained_sampling else {
        return Ok(None);
    };
    if supports_strict_mode {
        return Ok(Some(true));
    }
    if *strict == JsonSchemaStrictness::Require {
        return Err(ToolValidationError::UnsupportedConstraint(format!(
            "tool \"{}\" requires JSON-schema constrained sampling, but strict tools are unsupported",
            tool.name
        )));
    }
    Ok(None)
}

/// Resolves a grammar constraint into `(format, definition, input_property)`.
///
/// # Errors
///
/// Returns an unsupported-constraint error when enabled grammar sampling lacks
/// a usable provider variant or a valid single-string input property.
pub fn resolve_grammar_constraint(
    tool: &Tool,
    supports_openai_grammar_tools: bool,
) -> Result<Option<(GrammarFormat, String, String)>, ToolValidationError> {
    let Some(ConstrainedSampling::Grammar { variants }) = &tool.constrained_sampling else {
        return Ok(None);
    };
    if !supports_openai_grammar_tools {
        return Ok(None);
    }
    let selected = variants
        .openai_lark
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|value| (GrammarFormat::Lark, value))
        .or_else(|| {
            variants
                .openai_regex
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(|value| (GrammarFormat::Regex, value))
        })
        .ok_or_else(|| {
            ToolValidationError::UnsupportedConstraint(format!(
                "tool \"{}\" cannot use grammar constrained sampling: no supported grammar variant was provided",
                tool.name
            ))
        })?;
    let property = infer_grammar_input_property(tool)?;
    Ok(Some((selected.0, selected.1.to_owned(), property)))
}

fn infer_grammar_input_property(tool: &Tool) -> Result<String, ToolValidationError> {
    let error = |message: &str| {
        ToolValidationError::UnsupportedConstraint(format!(
            "tool \"{}\" cannot use grammar constrained sampling: {message}",
            tool.name
        ))
    };
    if tool.parameters.get("type").and_then(Value::as_str) != Some("object") {
        return Err(error(
            "grammar constrained sampling requires an object parameter schema",
        ));
    }
    let required = tool
        .parameters
        .get("required")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            error("grammar constrained sampling requires exactly one required string property")
        })?;
    if required.len() != 1 {
        return Err(error(
            "grammar constrained sampling requires exactly one required string property",
        ));
    }
    let property = required[0].as_str().ok_or_else(|| {
        error("grammar constrained sampling requires exactly one required string property")
    })?;
    let property_schema = tool
        .parameters
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(property))
        .ok_or_else(|| error("grammar input property is missing from properties"))?;
    if property_schema.get("type").and_then(Value::as_str) != Some("string") {
        return Err(error("grammar input property must have type string"));
    }
    Ok(property.to_owned())
}

/// Converts a tool to a wire-ready descriptor.
///
/// # Errors
///
/// Returns an unsupported-constraint error when the tool's requested strict or
/// grammar sampling mode cannot be represented by the target adapter.
pub fn describe_tool(
    tool: &Tool,
    supports_strict_mode: bool,
    supports_openai_grammar_tools: bool,
    defer_loading: bool,
) -> Result<ToolDescriptor, ToolValidationError> {
    if let Some((format, definition, input_property)) =
        resolve_grammar_constraint(tool, supports_openai_grammar_tools)?
    {
        return Ok(ToolDescriptor::Custom {
            name: tool.name.clone(),
            description: tool.description.clone(),
            format,
            definition,
            input_property,
            defer_loading,
        });
    }
    Ok(ToolDescriptor::Function {
        name: tool.name.clone(),
        description: tool.description.clone(),
        parameters: tool.parameters.clone(),
        strict: resolve_json_schema_strict(tool, supports_strict_mode)?,
        defer_loading,
    })
}

/// Scratch state for append-only grammar-tool JSON deltas.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GrammarToolInputBuffer {
    input: String,
    started: bool,
    closed: bool,
}

impl GrammarToolInputBuffer {
    /// Current unescaped grammar input.
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Appends a monotonic input snapshot and returns the JSON delta to emit.
    ///
    /// # Errors
    ///
    /// Returns an unsupported-constraint error when input changes
    /// non-monotonically, changes after closure, or cannot be JSON encoded.
    pub fn append(
        &mut self,
        input_property: &str,
        next_input: &str,
        close: bool,
    ) -> Result<Option<String>, ToolValidationError> {
        if self.closed {
            if close && next_input == self.input {
                return Ok(None);
            }
            return Err(ToolValidationError::UnsupportedConstraint(format!(
                "grammar tool input for property \"{input_property}\" changed after it was closed"
            )));
        }
        if !next_input.starts_with(&self.input) {
            return Err(ToolValidationError::UnsupportedConstraint(format!(
                "grammar tool input for property \"{input_property}\" changed non-monotonically"
            )));
        }
        let input_delta = &next_input[self.input.len()..];
        if !close && input_delta.is_empty() {
            return Ok(None);
        }
        let mut delta = String::new();
        if !self.started {
            let property = serde_json::to_string(input_property)
                .map_err(|error| ToolValidationError::UnsupportedConstraint(error.to_string()))?;
            delta.push('{');
            delta.push_str(&property);
            delta.push_str(":\"");
            self.started = true;
        }
        let encoded = serde_json::to_string(input_delta)
            .map_err(|error| ToolValidationError::UnsupportedConstraint(error.to_string()))?;
        let encoded_body = encoded
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
            .ok_or_else(|| {
                ToolValidationError::UnsupportedConstraint(
                    "grammar input did not encode as a JSON string".into(),
                )
            })?;
        delta.push_str(encoded_body);
        next_input.clone_into(&mut self.input);
        if close {
            delta.push_str("\"}");
            self.closed = true;
        }
        Ok(Some(delta))
    }
}

/// Immediate and transcript-loaded tool sets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeferredTools {
    /// Tools sent with the initial request.
    pub immediate: Vec<Tool>,
    /// Tools introduced at transcript markers, indexed by normalized name.
    pub deferred: IndexMap<String, Tool>,
}

/// Splits current tools into immediate and deferred definitions.
pub fn split_deferred_tools<F>(context: &Context, enabled: bool, normalize_name: F) -> DeferredTools
where
    F: Fn(&str) -> String,
{
    let mut unique = IndexMap::<String, Tool>::new();
    for tool in &context.tools {
        unique.insert(normalize_name(&tool.name), tool.clone());
    }
    if !enabled {
        return DeferredTools {
            immediate: unique.into_values().collect(),
            deferred: IndexMap::new(),
        };
    }

    let mut deferred_names = IndexMap::<String, ()>::new();
    let mut used_names = IndexMap::<String, ()>::new();
    for message in &context.messages {
        match message {
            Message::Assistant(message) => {
                for block in &message.content {
                    if let ContentBlock::ToolCall(call) = block {
                        used_names.insert(normalize_name(&call.name), ());
                    }
                }
            }
            Message::ToolResult(message) => {
                for name in &message.added_tool_names {
                    let normalized = normalize_name(name);
                    if !used_names.contains_key(&normalized) {
                        deferred_names.insert(normalized, ());
                    }
                }
            }
            Message::User(_) => {}
        }
    }

    let mut result = DeferredTools::default();
    for (name, tool) in unique {
        if deferred_names.contains_key(&name) {
            result.deferred.insert(name, tool);
        } else {
            result.immediate.push(tool);
        }
    }
    result
}

/// Extracts the grammar input string from parsed arguments.
///
/// # Errors
///
/// Returns an unsupported-constraint error when the expected argument is
/// absent or is not a string.
pub fn grammar_tool_input<'a>(
    tool_name: &str,
    arguments: &'a Value,
    input_property: &str,
) -> Result<&'a str, ToolValidationError> {
    arguments
        .get(input_property)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ToolValidationError::UnsupportedConstraint(format!(
                "grammar tool call \"{tool_name}\" requires argument \"{input_property}\" to be a string"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{AssistantMessage, Message, StopReason, ToolResultMessage, now_millis};

    fn call(value: &Value) -> ToolCall {
        ToolCall {
            id: "call".into(),
            name: "echo".into(),
            arguments: serde_json::json!({"value": value}),
            thought_signature: None,
        }
    }

    #[test]
    fn validates_and_coerces_plain_json_schema() {
        let cases = [
            (
                serde_json::json!({"type": "number"}),
                serde_json::json!("42"),
                serde_json::json!(42.0),
            ),
            (
                serde_json::json!({"type": "boolean"}),
                serde_json::json!(1),
                serde_json::json!(true),
            ),
            (
                serde_json::json!({"type": "string"}),
                Value::Null,
                serde_json::json!(""),
            ),
        ];
        for (property, input, expected) in cases {
            let tool = Tool::new(
                "echo",
                "echo",
                serde_json::json!({
                    "type": "object",
                    "properties": {"value": property},
                    "required": ["value"],
                    "additionalProperties": false
                }),
            );
            assert_eq!(
                validate_tool_arguments(&tool, &call(&input)).expect("valid arguments"),
                serde_json::json!({"value": expected})
            );
        }
    }

    #[test]
    fn rejects_invalid_coercion() {
        let tool = Tool::new(
            "echo",
            "echo",
            serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "boolean"}},
                "required": ["value"]
            }),
        );
        assert!(matches!(
            validate_tool_arguments(&tool, &call(&serde_json::json!("1"))),
            Err(ToolValidationError::InvalidArguments { .. })
        ));
    }

    #[test]
    fn emits_strict_and_grammar_descriptors() {
        let mut strict = Tool::new("echo", "echo", serde_json::json!({"type": "object"}));
        strict.constrained_sampling = Some(ConstrainedSampling::JsonSchema {
            strict: JsonSchemaStrictness::Require,
        });
        assert!(matches!(
            describe_tool(&strict, true, false, false).expect("strict descriptor"),
            ToolDescriptor::Function {
                strict: Some(true),
                ..
            }
        ));
        assert!(describe_tool(&strict, false, false, false).is_err());

        let mut grammar = Tool::new(
            "parse",
            "parse",
            serde_json::json!({
                "type": "object",
                "properties": {"payload": {"type": "string"}},
                "required": ["payload"]
            }),
        );
        grammar.constrained_sampling = Some(ConstrainedSampling::Grammar {
            variants: GrammarVariants {
                openai_lark: Some("start: /[a-z]+/".into()),
                openai_regex: None,
            },
        });
        assert!(matches!(
            describe_tool(&grammar, true, true, true).expect("grammar descriptor"),
            ToolDescriptor::Custom {
                format: GrammarFormat::Lark,
                ref input_property,
                defer_loading: true,
                ..
            } if input_property == "payload"
        ));
    }

    #[test]
    fn grammar_deltas_form_valid_json() {
        let mut buffer = GrammarToolInputBuffer::default();
        let first = buffer
            .append("payload", "a\"", false)
            .expect("first delta")
            .expect("nonempty");
        let second = buffer
            .append("payload", "a\"\nb", true)
            .expect("second delta")
            .expect("nonempty");
        assert_eq!(
            serde_json::from_str::<Value>(&format!("{first}{second}")).expect("valid JSON"),
            serde_json::json!({"payload": "a\"\nb"})
        );
        assert_eq!(
            buffer
                .append("payload", "a\"\nb", true)
                .expect("idempotent"),
            None
        );
        assert!(buffer.append("payload", "changed", true).is_err());
    }

    #[test]
    fn deferred_marker_only_defers_unused_active_tools() {
        let base = Tool::new("base", "base", serde_json::json!({"type": "object"}));
        let late = Tool::new("late", "late", serde_json::json!({"type": "object"}));
        let mut assistant = AssistantMessage::empty("a", "p", "m");
        assistant.stop_reason = StopReason::ToolUse;
        assistant.content.push(ContentBlock::ToolCall(ToolCall {
            id: "c".into(),
            name: "base".into(),
            arguments: serde_json::json!({}),
            thought_signature: None,
        }));
        let context = Context {
            system_prompt: None,
            messages: vec![
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "c".into(),
                    tool_name: "base".into(),
                    content: vec![],
                    details: None,
                    usage: None,
                    added_tool_names: vec!["late".into()],
                    is_error: false,
                    timestamp: now_millis(),
                }),
            ],
            tools: vec![base, late],
        };
        let split = split_deferred_tools(&context, true, str::to_owned);
        assert_eq!(split.immediate.len(), 1);
        assert!(split.deferred.contains_key("late"));
    }
}
