//! Native request and response types for exported guest operations.

use crate::error::{HostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const MAX_VIEW_NODES: usize = 10_000;

/// Runtime lifecycle phase tracked by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum LifecyclePhase {
    /// Instantiated and descriptor-validated.
    Loaded,
    /// Successfully activated.
    Active,
    /// Successfully deactivated.
    Deactivated,
}

impl LifecyclePhase {
    /// Stable phase name used in typed errors.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Loaded => "loaded",
            Self::Active => "active",
            Self::Deactivated => "deactivated",
        }
    }
}

/// Context supplied to guest activation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationContext {
    /// Current native session, when any.
    pub session_id: Option<String>,
    /// Current workspace URI, when any.
    pub workspace_uri: Option<String>,
    /// Extension configuration object.
    pub configuration: Value,
}

/// Events to which an extension may subscribe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EventKind {
    /// Activation completed.
    Activated,
    /// Deactivation is about to begin.
    Deactivating,
    /// Extension configuration changed.
    ConfigurationChanged,
    /// A session opened.
    SessionOpened,
    /// A session closed.
    SessionClosed,
    /// Provider configuration changed.
    ProviderChanged,
    /// A tool call completed.
    ToolCompleted,
    /// A command was invoked.
    CommandInvoked,
    /// Extension-defined topic.
    Custom,
}

/// Guest activation response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivationResult {
    /// Event categories requested by the component.
    pub subscriptions: Vec<EventKind>,
    /// Optional opaque JSON state.
    pub state: Option<Value>,
}

/// A host event delivered to an active component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExtensionEvent {
    /// Event category.
    pub kind: EventKind,
    /// Stable topic name.
    pub topic: String,
    /// Event payload.
    pub payload: Value,
    /// Monotonic sequence assigned by the native host.
    pub sequence: u64,
}

/// A registered tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolInvocation {
    /// Tool ID from the validated descriptor.
    pub id: String,
    /// Tool input value.
    pub input: Value,
    /// Native invocation context.
    pub context: Value,
}

/// Result of a registered tool invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResult {
    /// Structured tool result.
    pub content: Value,
    /// Whether this is a model-visible tool failure.
    pub is_error: bool,
}

/// A registered command invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandInvocation {
    /// Command ID from the validated descriptor.
    pub id: String,
    /// Command arguments.
    pub arguments: Value,
    /// Native invocation context.
    pub context: Value,
}

/// Result of a registered command invocation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandResult {
    /// Structured command output.
    pub output: Value,
    /// View IDs the native host should refresh.
    pub refresh_views: Vec<String>,
}

/// Requested host location of an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActionKind {
    /// Primary activation.
    Activate,
    /// Input value changed.
    Change,
    /// Form or workflow submission.
    Submit,
    /// Dismissal.
    Dismiss,
    /// Extension-defined action.
    Custom,
}

/// An action attached to a declarative node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionBinding {
    /// Extension-defined action ID.
    pub action_id: String,
    /// Host interaction category.
    pub kind: ActionKind,
    /// Static payload included with the event.
    pub payload: Value,
}

/// Node kinds supported by the host renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ViewNodeKind {
    /// Layout-only node.
    Container,
    /// Plain text.
    Text,
    /// Host-rendered markdown.
    Markdown,
    /// Button control.
    Button,
    /// Text input control.
    Input,
    /// Selection control.
    Select,
    /// Image reference.
    Image,
    /// Layout spacer.
    Spacer,
}

/// A property on a declarative view node.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewProperty {
    /// Property name interpreted by the host renderer.
    pub name: String,
    /// Structured property value.
    pub value: Value,
}

/// One node in a flat, ID-addressed declarative view tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewNode {
    /// View-local node ID.
    pub id: u64,
    /// Renderer primitive.
    pub kind: ViewNodeKind,
    /// Optional visible text.
    pub text: Option<String>,
    /// Declarative properties.
    pub properties: Vec<ViewProperty>,
    /// Host-routed actions.
    pub actions: Vec<ActionBinding>,
    /// Child node IDs.
    pub children: Vec<u64>,
}

/// A declarative view represented as a flat, acyclic tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct View {
    /// Root node ID.
    pub root: u64,
    /// All nodes in the view.
    pub nodes: Vec<ViewNode>,
}

impl View {
    /// Ensures IDs are unique, references exist, and the graph is one tree.
    ///
    /// # Errors
    ///
    /// Returns a descriptor error for an empty or oversized view, duplicate or
    /// missing nodes, cycles, multiple parents, or unreachable nodes.
    pub fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return invalid_view("view must contain at least one node");
        }
        if self.nodes.len() > MAX_VIEW_NODES {
            return invalid_view(format!("view contains more than {MAX_VIEW_NODES} nodes"));
        }
        let mut nodes = BTreeMap::new();
        for node in &self.nodes {
            if nodes.insert(node.id, node).is_some() {
                return invalid_view(format!("duplicate view node id {}", node.id));
            }
            for property in &node.properties {
                if property.name.is_empty() {
                    return invalid_view("view property names must not be empty");
                }
            }
            let mut action_ids = BTreeSet::new();
            for action in &node.actions {
                if action.action_id.is_empty() || !action_ids.insert(action.action_id.as_str()) {
                    return invalid_view(format!(
                        "node {} has an empty or duplicate action id",
                        node.id
                    ));
                }
            }
        }
        if !nodes.contains_key(&self.root) {
            return invalid_view(format!("view root {} does not exist", self.root));
        }
        let mut parent_counts = BTreeMap::<u64, usize>::new();
        for node in &self.nodes {
            for child in &node.children {
                *parent_counts.entry(*child).or_default() += 1;
            }
        }
        if parent_counts.get(&self.root).copied().unwrap_or(0) != 0 {
            return invalid_view("view root must not have a parent");
        }
        for id in nodes.keys().copied().filter(|id| *id != self.root) {
            if parent_counts.get(&id).copied().unwrap_or(0) != 1 {
                return invalid_view(format!("view node {id} must have exactly one parent"));
            }
        }

        let mut visiting = BTreeSet::new();
        let mut visited = BTreeSet::new();
        visit_node(self.root, &nodes, &mut visiting, &mut visited)?;
        if visited.len() != nodes.len() {
            return invalid_view("view contains nodes unreachable from its root");
        }
        Ok(())
    }
}

/// Request to render a registered view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ViewRequest {
    /// View ID from the validated descriptor.
    pub id: String,
    /// Native rendering context.
    pub context: Value,
}

/// User action routed back to the component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionEvent {
    /// Registered view ID.
    pub view_id: String,
    /// Action ID from the rendered node.
    pub action_id: String,
    /// Host interaction category.
    pub kind: ActionKind,
    /// Dynamic event payload.
    pub payload: Value,
}

/// Result of handling a declarative action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionResult {
    /// Structured action output.
    pub output: Value,
    /// Optional complete replacement for the current view.
    pub replacement_view: Option<View>,
}

/// Why an extension is being deactivated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DeactivateReason {
    /// Disabled by policy or user choice.
    Disabled,
    /// Replaced by a newer generation.
    Reload,
    /// Host shutdown.
    Shutdown,
    /// Extension failure.
    Failure,
}

/// Operations available through [`crate::WasmExtensionHost::invoke`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "request", rename_all = "kebab-case")]
pub enum Invocation {
    /// Activate a loaded component.
    Activate(ActivationContext),
    /// Deliver an event.
    Event(ExtensionEvent),
    /// Invoke a registered tool.
    Tool(ToolInvocation),
    /// Invoke a registered command.
    Command(CommandInvocation),
    /// Render a registered declarative view.
    RenderView(ViewRequest),
    /// Route an action from a rendered view.
    Action(ActionEvent),
    /// Deactivate the component.
    Deactivate(DeactivateReason),
}

/// Results returned by [`crate::WasmExtensionHost::invoke`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", content = "result", rename_all = "kebab-case")]
pub enum InvocationResult {
    /// Activation completed.
    Activated(ActivationResult),
    /// Event delivery completed.
    EventDelivered,
    /// Tool invocation completed.
    Tool(ToolResult),
    /// Command invocation completed.
    Command(CommandResult),
    /// View rendering completed.
    View(View),
    /// Action handling completed.
    Action(ActionResult),
    /// Deactivation completed.
    Deactivated,
}

fn visit_node(
    id: u64,
    nodes: &BTreeMap<u64, &ViewNode>,
    visiting: &mut BTreeSet<u64>,
    visited: &mut BTreeSet<u64>,
) -> Result<()> {
    if visited.contains(&id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return invalid_view(format!("view contains a cycle at node {id}"));
    }
    let node = nodes.get(&id).ok_or_else(|| {
        HostError::InvalidDescriptor(format!("view references missing node {id}"))
    })?;
    let mut unique_children = BTreeSet::new();
    for child in &node.children {
        if !unique_children.insert(*child) {
            return invalid_view(format!("node {id} references child {child} twice"));
        }
        if !nodes.contains_key(child) {
            return invalid_view(format!("node {id} references missing child {child}"));
        }
        visit_node(*child, nodes, visiting, visited)?;
    }
    visiting.remove(&id);
    visited.insert(id);
    Ok(())
}

fn invalid_view<T>(message: impl Into<String>) -> Result<T> {
    Err(HostError::InvalidDescriptor(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, children: Vec<u64>) -> ViewNode {
        ViewNode {
            id,
            kind: ViewNodeKind::Container,
            text: None,
            properties: Vec::new(),
            actions: Vec::new(),
            children,
        }
    }

    #[test]
    fn declarative_view_must_be_an_acyclic_tree() {
        let view = View {
            root: 1,
            nodes: vec![node(1, vec![2]), node(2, vec![1])],
        };
        assert!(matches!(
            view.validate(),
            Err(HostError::InvalidDescriptor(_))
        ));
    }

    #[test]
    fn declarative_view_rejects_unreachable_nodes() {
        let view = View {
            root: 1,
            nodes: vec![node(1, Vec::new()), node(2, Vec::new())],
        };
        assert!(view.validate().is_err());
    }
}
