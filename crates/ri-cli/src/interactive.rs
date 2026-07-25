//! Minimal interactive binding built from `ri-tui` primitives.

#![cfg(feature = "interactive")]

use std::sync::{Arc, Mutex, MutexGuard};

use ri_tui::components::{Editor, EditorOptions, Markdown};
use ri_tui::{
    Component, ConstrainedLine, EditorTheme, InputEvent, KeyEventKind, MarkdownTheme,
    RenderContext, Result,
};
use tokio::sync::mpsc;

/// Action produced by the terminal editor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InteractiveAction {
    Submit(String),
    Abort,
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EntryRole {
    User,
    Assistant,
    Notice,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranscriptEntry {
    role: EntryRole,
    text: String,
}

#[derive(Debug, Default)]
struct ViewState {
    entries: Vec<TranscriptEntry>,
    busy: bool,
    status: String,
}

/// Cloneable handle used by the asynchronous runtime loop.
#[derive(Debug, Clone)]
pub(crate) struct InteractiveHandle {
    state: Arc<Mutex<ViewState>>,
}

impl InteractiveHandle {
    pub(crate) fn user(&self, text: impl Into<String>) {
        self.push(EntryRole::User, text);
    }

    pub(crate) fn assistant(&self, text: impl Into<String>) {
        self.push(EntryRole::Assistant, text);
    }

    pub(crate) fn notice(&self, text: impl Into<String>) {
        self.push(EntryRole::Notice, text);
    }

    pub(crate) fn error(&self, text: impl Into<String>) {
        self.push(EntryRole::Error, text);
    }

    pub(crate) fn set_busy(&self, busy: bool) {
        self.lock().busy = busy;
    }

    pub(crate) fn set_status(&self, status: impl Into<String>) {
        self.lock().status = status.into();
    }

    fn push(&self, role: EntryRole, text: impl Into<String>) {
        let text = text.into();
        if !text.trim().is_empty() {
            self.lock().entries.push(TranscriptEntry { role, text });
        }
    }

    fn lock(&self) -> MutexGuard<'_, ViewState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Root chat transcript and editor component.
pub(crate) struct InteractiveComponent {
    state: Arc<Mutex<ViewState>>,
    editor: Editor,
    actions: mpsc::UnboundedSender<InteractiveAction>,
}

impl std::fmt::Debug for InteractiveComponent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("InteractiveComponent")
            .field("state", &self.state)
            .field("editor", &self.editor)
            .finish_non_exhaustive()
    }
}

/// Build the mounted component, its update handle, and its action receiver.
pub(crate) fn channel() -> (
    InteractiveComponent,
    InteractiveHandle,
    mpsc::UnboundedReceiver<InteractiveAction>,
) {
    let state = Arc::new(Mutex::new(ViewState {
        status: "Enter sends · Esc aborts · Ctrl+D exits".to_owned(),
        ..ViewState::default()
    }));
    let (actions, receiver) = mpsc::unbounded_channel();
    let submit = actions.clone();
    let mut editor = Editor::new(EditorTheme::default(), EditorOptions::default());
    editor.set_viewport_height(4);
    editor.on_submit(move |text| {
        if !text.trim().is_empty() {
            let _ = submit.send(InteractiveAction::Submit(text.to_owned()));
        }
    });
    (
        InteractiveComponent {
            state: Arc::clone(&state),
            editor,
            actions,
        },
        InteractiveHandle { state },
        receiver,
    )
}

impl Component for InteractiveComponent {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        let editor_context = RenderContext {
            width: context.width,
            height: 4,
            focused: context.focused,
        };
        let editor = self.editor.render(editor_context)?;
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let status = if state.busy {
            format!("Working… {}", state.status)
        } else {
            state.status.clone()
        };
        let status = ConstrainedLine::truncated(status, context.width, "…");
        let transcript_height = context
            .height
            .saturating_sub(editor.len().saturating_add(1));
        let mut transcript = render_transcript(&state.entries, context.width)?;
        if transcript.len() > transcript_height {
            transcript.drain(..transcript.len() - transcript_height);
        }
        drop(state);

        transcript.push(status);
        transcript.extend(editor);
        Ok(transcript)
    }

    fn handle_event(&mut self, event: &InputEvent) -> bool {
        if let InputEvent::Key(key) = event
            && key.kind != KeyEventKind::Release
        {
            if key.matches("ctrl+d") {
                let _ = self.actions.send(InteractiveAction::Quit);
                return true;
            }
            if key.matches("ctrl+c") {
                if self.editor.text().is_empty() {
                    let _ = self.actions.send(InteractiveAction::Quit);
                } else {
                    self.editor.set_text("");
                }
                return true;
            }
            if key.matches("escape") {
                let _ = self.actions.send(InteractiveAction::Abort);
                return true;
            }
        }
        self.editor.handle_event(event)
    }

    fn set_focused(&mut self, focused: bool) {
        self.editor.set_focused(focused);
    }

    fn focusable(&self) -> bool {
        true
    }

    fn invalidate(&mut self) {
        self.editor.invalidate();
    }
}

fn render_transcript(entries: &[TranscriptEntry], width: usize) -> Result<Vec<ConstrainedLine>> {
    let theme = MarkdownTheme::default();
    let mut output = Vec::new();
    for entry in entries {
        let source = match entry.role {
            EntryRole::User => format!("### You\n{}", entry.text),
            EntryRole::Assistant => entry.text.clone(),
            EntryRole::Notice => format!("> {}", entry.text),
            EntryRole::Error => format!("> Error: {}", entry.text),
        };
        let mut markdown = Markdown::new(source, theme.clone()).with_padding(1, 0);
        output.extend(markdown.render(RenderContext {
            width,
            height: usize::MAX,
            focused: false,
        })?);
        output.push(ConstrainedLine::empty(width));
    }
    if output.last().is_some() {
        output.pop();
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_never_exceeds_width() {
        let entries = vec![
            TranscriptEntry {
                role: EntryRole::User,
                text: "Please explain this very long request".to_owned(),
            },
            TranscriptEntry {
                role: EntryRole::Assistant,
                text: "A detailed response with `code`.".to_owned(),
            },
        ];
        let lines = render_transcript(&entries, 16).unwrap();
        assert!(lines.iter().all(|line| line.width() <= 16));
    }
}
