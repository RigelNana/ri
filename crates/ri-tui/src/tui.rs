//! High-level component, overlay, input, and renderer coordinator.

use std::time::Duration;

use crate::component::{Component, ComponentId, ComponentTree, InputEvent};
use crate::overlay::{OverlayId, OverlayManager, OverlayOptions};
use crate::renderer::{DifferentialRenderer, Frame};
use crate::terminal::{Terminal, TerminalEvent};
use crate::{Error, Result};

/// Interactive TUI runtime.
#[derive(Debug)]
pub struct Tui<T> {
    tree: ComponentTree,
    overlays: OverlayManager,
    renderer: DifferentialRenderer<T>,
    focused: Option<ComponentId>,
}

impl<T: Terminal> Tui<T> {
    /// Creates a runtime around a terminal backend.
    pub fn new(terminal: T) -> Self {
        Self {
            tree: ComponentTree::new(),
            overlays: OverlayManager::default(),
            renderer: DifferentialRenderer::new(terminal),
            focused: None,
        }
    }

    /// Enables terminal modes.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal setup fails.
    pub fn start(&mut self) -> Result<()> {
        self.renderer.start()?;
        Ok(())
    }

    /// Restores terminal modes.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal restoration fails.
    pub fn stop(&mut self) -> Result<()> {
        self.renderer.stop()?;
        Ok(())
    }

    /// Mounts a root component.
    pub fn mount(&mut self, component: impl Component + 'static) -> ComponentId {
        let focusable = component.focusable();
        let id = self.tree.push(component);
        if self.focused.is_none() && focusable {
            self.transition_focus(Some(id));
        }
        id
    }

    /// Mounts an erased root component.
    pub fn mount_boxed(&mut self, component: Box<dyn Component>) -> ComponentId {
        let focusable = component.focusable();
        let id = self.tree.push_boxed(component);
        if self.focused.is_none() && focusable {
            self.transition_focus(Some(id));
        }
        id
    }

    /// Removes a root component.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingComponent`] when `id` is not mounted.
    pub fn remove(&mut self, id: ComponentId) -> Result<Box<dyn Component>> {
        let component = self.tree.remove(id).ok_or(Error::MissingComponent(id))?;
        if self.focused == Some(id) {
            let next = self
                .tree
                .ids()
                .find(|candidate| self.tree.get(*candidate).is_some_and(Component::focusable));
            self.transition_focus(next);
        }
        Ok(component)
    }

    /// Shows a responsive overlay.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal dimensions cannot be queried.
    pub fn show_overlay(
        &mut self,
        component: impl Component + 'static,
        options: OverlayOptions,
    ) -> Result<OverlayId> {
        self.show_overlay_boxed(Box::new(component), options)
    }

    /// Shows an erased responsive overlay.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal dimensions cannot be queried.
    pub fn show_overlay_boxed(
        &mut self,
        component: Box<dyn Component>,
        options: OverlayOptions,
    ) -> Result<OverlayId> {
        let size = self.renderer.terminal().size()?;
        let (id, next) = self.overlays.show(component, options, self.focused, size);
        self.transition_focus(next);
        Ok(id)
    }

    /// Hides or reveals an overlay without unmounting it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingComponent`] for an unknown overlay, or an I/O
    /// error when terminal dimensions cannot be queried.
    pub fn set_overlay_hidden(&mut self, id: OverlayId, hidden: bool) -> Result<()> {
        self.ensure_overlay(id)?;
        let size = self.renderer.terminal().size()?;
        let next = self.overlays.set_hidden(id, hidden, self.focused, size);
        self.transition_focus(next);
        self.renderer.force_redraw();
        Ok(())
    }

    /// Removes an overlay and restores the best previous focus target.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingComponent`] for an unknown overlay, or an I/O
    /// error when terminal dimensions cannot be queried.
    pub fn remove_overlay(&mut self, id: OverlayId) -> Result<()> {
        self.ensure_overlay(id)?;
        let size = self.renderer.terminal().size()?;
        let next = self.overlays.remove(id, self.focused, size);
        self.transition_focus(next);
        self.renderer.force_redraw();
        Ok(())
    }

    /// Gives an overlay focus and brings it to the top.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingComponent`] for an unknown overlay, or an I/O
    /// error when terminal dimensions cannot be queried.
    pub fn focus_overlay(&mut self, id: OverlayId) -> Result<()> {
        self.ensure_overlay(id)?;
        let size = self.renderer.terminal().size()?;
        let next = self.overlays.focus(id, self.focused, size);
        self.transition_focus(next);
        Ok(())
    }

    /// Returns overlay focus to the next capture target or its pre-focus owner.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingComponent`] for an unknown overlay, or an I/O
    /// error when terminal dimensions cannot be queried.
    pub fn unfocus_overlay(&mut self, id: OverlayId, explicit: Option<ComponentId>) -> Result<()> {
        self.ensure_overlay(id)?;
        let size = self.renderer.terminal().size()?;
        let next = self.overlays.unfocus(id, self.focused, explicit, size);
        self.transition_focus(next);
        Ok(())
    }

    /// Focuses a mounted root or overlay component.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingComponent`] when `id` is not mounted.
    pub fn focus(&mut self, id: ComponentId) -> Result<()> {
        let exists = self.tree.contains(id) || self.overlays.contains(id);
        if !exists {
            return Err(Error::MissingComponent(id));
        }
        self.transition_focus(Some(id));
        Ok(())
    }

    /// Clears keyboard focus.
    pub fn blur(&mut self) {
        self.transition_focus(None);
    }

    /// Current focus owner.
    pub const fn focused(&self) -> Option<ComponentId> {
        self.focused
    }

    /// Moves focus among focusable root components.
    pub fn focus_next(&mut self, reverse: bool) {
        let focusable = self
            .tree
            .ids()
            .filter(|id| self.tree.get(*id).is_some_and(Component::focusable))
            .collect::<Vec<_>>();
        let Some(last) = focusable.last().copied() else {
            return;
        };
        let index = self
            .focused
            .and_then(|focused| focusable.iter().position(|id| *id == focused));
        let next = match (index, reverse) {
            (Some(0) | None, true) => last,
            (Some(index), true) => focusable[index - 1],
            (Some(index), false) => focusable[(index + 1) % focusable.len()],
            (None, false) => focusable[0],
        };
        self.transition_focus(Some(next));
    }

    /// Renders the component tree and overlay stack.
    ///
    /// # Errors
    ///
    /// Returns an error when terminal sizing or output fails, a component
    /// cannot render, or a rendered line violates its width contract.
    pub fn render(&mut self) -> Result<()> {
        let (columns, rows) = self.renderer.terminal().size()?;
        let width = usize::from(columns.max(1));
        let height = usize::from(rows.max(1));
        let root = self.tree.render(width, height, self.focused)?;
        let root = root
            .into_iter()
            .map(crate::line::ConstrainedLine::into_string)
            .collect();
        let composited = self.overlays.composite(root, width, height, self.focused)?;
        let lines = composited
            .into_iter()
            .map(|line| crate::line::ConstrainedLine::new(line, width))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        self.renderer.render(Frame::new(lines))
    }

    /// Routes a terminal event. Returns whether application state may have changed.
    pub fn dispatch_terminal_event(&mut self, event: TerminalEvent) -> bool {
        match event {
            TerminalEvent::Key(key) => self.dispatch_input(&InputEvent::Key(key)),
            TerminalEvent::Paste(text) => self.dispatch_input(&InputEvent::Paste(text)),
            TerminalEvent::Resize { columns, rows } => {
                let event = InputEvent::Resize { columns, rows };
                self.tree.invalidate();
                self.overlays.invalidate();
                let ids = self.tree.ids().collect::<Vec<_>>();
                for id in ids {
                    self.tree.dispatch(id, &event);
                }
                let next = self.overlays.reconcile_focus(self.focused, (columns, rows));
                self.transition_focus(next);
                self.renderer.force_redraw();
                true
            }
            TerminalEvent::FocusGained => true,
            TerminalEvent::FocusLost => false,
        }
    }

    /// Waits for one event and routes it.
    ///
    /// # Errors
    ///
    /// Returns an I/O error when terminal input cannot be polled or read.
    pub fn poll_event(&mut self, timeout: Duration) -> Result<bool> {
        let event = self.renderer.terminal_mut().read_event(timeout)?;
        Ok(event.is_some_and(|event| self.dispatch_terminal_event(event)))
    }

    /// Polls input and renders when the event was consumed.
    ///
    /// # Errors
    ///
    /// Returns an error when polling input or rendering the changed state
    /// fails.
    pub fn tick(&mut self, timeout: Duration) -> Result<bool> {
        let changed = self.poll_event(timeout)?;
        if changed {
            self.render()?;
        }
        Ok(changed)
    }

    /// Borrows the terminal.
    pub const fn terminal(&self) -> &T {
        self.renderer.terminal()
    }

    /// Mutably borrows the terminal.
    pub fn terminal_mut(&mut self) -> &mut T {
        self.renderer.terminal_mut()
    }

    /// Borrows renderer statistics.
    pub const fn renderer(&self) -> &DifferentialRenderer<T> {
        &self.renderer
    }

    /// Invalidates component caches and forces a safe redraw.
    pub fn invalidate(&mut self) {
        self.tree.invalidate();
        self.overlays.invalidate();
        self.renderer.force_redraw();
    }

    fn dispatch_input(&mut self, event: &InputEvent) -> bool {
        let Some(focused) = self.focused else {
            return false;
        };
        if self.overlays.contains(focused) {
            self.overlays.dispatch(focused, event)
        } else {
            self.tree.dispatch(focused, event)
        }
    }

    fn transition_focus(&mut self, next: Option<ComponentId>) {
        if self.focused == next {
            return;
        }
        let previous = self.focused;
        self.tree.transition_focus(previous, next);
        if let Some(id) = previous {
            self.overlays.set_focused(id, false);
        }
        if let Some(id) = next {
            self.overlays.set_focused(id, true);
        }
        self.focused = next;
    }

    fn ensure_overlay(&self, id: OverlayId) -> Result<()> {
        if self.overlays.contains(id.component_id()) {
            Ok(())
        } else {
            Err(Error::MissingComponent(id.component_id()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::RenderContext;
    use crate::line::ConstrainedLine;
    use crate::overlay::ResponsiveVisibility;
    use crate::virtual_terminal::VirtualTerminal;

    #[derive(Debug)]
    struct Focusable(&'static str);

    impl Component for Focusable {
        fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
            Ok(vec![ConstrainedLine::new(self.0, context.width)?])
        }

        fn focusable(&self) -> bool {
            true
        }
    }

    #[test]
    fn nested_overlays_restore_focus_in_stack_order() {
        let mut tui = Tui::new(VirtualTerminal::new(40, 10));
        let root = tui.mount(Focusable("root"));
        let first = tui
            .show_overlay(Focusable("first"), OverlayOptions::default())
            .unwrap();
        let second = tui
            .show_overlay(Focusable("second"), OverlayOptions::default())
            .unwrap();
        assert_eq!(tui.focused(), Some(second.component_id()));
        tui.remove_overlay(second).unwrap();
        assert_eq!(tui.focused(), Some(first.component_id()));
        tui.remove_overlay(first).unwrap();
        assert_eq!(tui.focused(), Some(root));
    }

    #[test]
    fn responsive_overlay_captures_and_restores_focus() {
        let mut tui = Tui::new(VirtualTerminal::new(40, 10));
        let root = tui.mount(Focusable("root"));
        let overlay = tui
            .show_overlay(
                Focusable("wide"),
                OverlayOptions {
                    responsive: ResponsiveVisibility {
                        min_width: Some(50),
                        ..ResponsiveVisibility::default()
                    },
                    ..OverlayOptions::default()
                },
            )
            .unwrap();
        assert_eq!(tui.focused(), Some(root));
        tui.terminal_mut().resize(60, 10);
        tui.dispatch_terminal_event(TerminalEvent::Resize {
            columns: 60,
            rows: 10,
        });
        assert_eq!(tui.focused(), Some(overlay.component_id()));
        tui.terminal_mut().resize(40, 10);
        tui.dispatch_terminal_event(TerminalEvent::Resize {
            columns: 40,
            rows: 10,
        });
        assert_eq!(tui.focused(), Some(root));
    }
}
