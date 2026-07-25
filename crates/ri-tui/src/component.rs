//! Component traits and the vertical component tree.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::Result;
use crate::keys::KeyEvent;
use crate::line::ConstrainedLine;

static NEXT_COMPONENT_ID: AtomicU64 = AtomicU64::new(1);

/// Stable identity assigned when a component is mounted.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ComponentId(u64);

impl ComponentId {
    /// Allocates an identifier unique within this process.
    pub fn allocate() -> Self {
        Self(NEXT_COMPONENT_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Returns the numeric identifier.
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Immutable information supplied during component rendering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderContext {
    /// Maximum width in terminal cells.
    pub width: usize,
    /// Visible terminal height.
    pub height: usize,
    /// Whether this component owns keyboard focus.
    pub focused: bool,
}

/// Input routed through the component tree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputEvent {
    /// Decoded key event.
    Key(KeyEvent),
    /// Complete bracketed paste.
    Paste(String),
    /// Terminal resize.
    Resize {
        /// New columns.
        columns: u16,
        /// New rows.
        rows: u16,
    },
}

/// A renderable, optionally interactive TUI unit.
pub trait Component: Send {
    /// Renders lines respecting `context.width`.
    ///
    /// # Errors
    ///
    /// Returns an error when a rendered line violates its width contract or
    /// when component-specific rendering fails.
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>>;

    /// Handles routed input. `true` means the event was consumed.
    fn handle_event(&mut self, _event: &InputEvent) -> bool {
        false
    }

    /// Receives focus transitions.
    fn set_focused(&mut self, _focused: bool) {}

    /// Whether this component can receive focus.
    fn focusable(&self) -> bool {
        false
    }

    /// Invalidates internal render caches.
    fn invalidate(&mut self) {}
}

struct MountedComponent {
    id: ComponentId,
    component: Box<dyn Component>,
}

/// Ordered vertical tree of mounted components.
#[derive(Default)]
pub struct ComponentTree {
    children: Vec<MountedComponent>,
}

impl std::fmt::Debug for ComponentTree {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ComponentTree")
            .field(
                "children",
                &self
                    .children
                    .iter()
                    .map(|child| child.id)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ComponentTree {
    /// Creates an empty tree.
    pub const fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    /// Mounts a child at the end of the vertical stack.
    pub fn push(&mut self, component: impl Component + 'static) -> ComponentId {
        self.push_boxed(Box::new(component))
    }

    /// Mounts an erased child.
    pub fn push_boxed(&mut self, component: Box<dyn Component>) -> ComponentId {
        let id = ComponentId::allocate();
        self.children.push(MountedComponent { id, component });
        id
    }

    /// Removes and returns a child.
    pub fn remove(&mut self, id: ComponentId) -> Option<Box<dyn Component>> {
        let index = self.children.iter().position(|child| child.id == id)?;
        Some(self.children.remove(index).component)
    }

    /// Removes all children.
    pub fn clear(&mut self) {
        self.children.clear();
    }

    /// Returns whether an identifier is mounted.
    pub fn contains(&self, id: ComponentId) -> bool {
        self.children.iter().any(|child| child.id == id)
    }

    /// Returns child identifiers in render order.
    pub fn ids(&self) -> impl Iterator<Item = ComponentId> + '_ {
        self.children.iter().map(|child| child.id)
    }

    /// Borrows a child.
    pub fn get(&self, id: ComponentId) -> Option<&dyn Component> {
        self.children
            .iter()
            .find(|child| child.id == id)
            .map(|child| child.component.as_ref())
    }

    /// Mutably borrows a child.
    pub fn get_mut(&mut self, id: ComponentId) -> Option<&mut (dyn Component + '_)> {
        for child in &mut self.children {
            if child.id == id {
                return Some(child.component.as_mut());
            }
        }
        None
    }

    /// Renders all children into one vertical frame.
    ///
    /// # Errors
    ///
    /// Returns the first error produced by a child component.
    pub fn render(
        &mut self,
        width: usize,
        height: usize,
        focused: Option<ComponentId>,
    ) -> Result<Vec<ConstrainedLine>> {
        let mut output = Vec::new();
        for child in &mut self.children {
            output.extend(child.component.render(RenderContext {
                width,
                height,
                focused: focused == Some(child.id),
            })?);
        }
        Ok(output)
    }

    /// Routes input to one mounted child.
    pub fn dispatch(&mut self, id: ComponentId, event: &InputEvent) -> bool {
        self.get_mut(id)
            .is_some_and(|component| component.handle_event(event))
    }

    /// Applies a focus transition to mounted children.
    pub fn transition_focus(&mut self, previous: Option<ComponentId>, next: Option<ComponentId>) {
        if previous == next {
            return;
        }
        if let Some(component) = previous.and_then(|id| self.get_mut(id)) {
            component.set_focused(false);
        }
        if let Some(component) = next.and_then(|id| self.get_mut(id)) {
            component.set_focused(true);
        }
    }

    /// Invalidates all children.
    pub fn invalidate(&mut self) {
        for child in &mut self.children {
            child.component.invalidate();
        }
    }
}
