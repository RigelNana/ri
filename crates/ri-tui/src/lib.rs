//! Safe terminal UI building blocks used by `ri`.
//!
//! The crate deliberately separates terminal I/O, input framing, components,
//! layout, and rendering.  Applications can therefore use the real crossterm
//! backend or the deterministic virtual terminal without changing UI code.

pub mod ansi;
pub mod autocomplete;
pub mod component;
pub mod components;
pub mod editing;
pub mod image;
pub mod input_buffer;
pub mod keys;
pub mod line;
pub mod overlay;
pub mod renderer;
pub mod terminal;
pub mod theme;
pub mod tui;
pub mod virtual_terminal;

pub use autocomplete::{
    AutocompleteContext, AutocompleteItem, AutocompleteProvider, AutocompleteResult,
    StaticAutocomplete,
};
pub use component::{Component, ComponentId, ComponentTree, InputEvent, RenderContext};
pub use image::{
    CellDimensions, ImageCellSize, ImageDimensions, ImageProtocol, ImageRender,
    Iterm2ImageDescriptor, KittyImageDescriptor, TerminalCapabilities,
};
pub use input_buffer::{InputFrame, StdinFrameBuffer};
pub use keys::{KeyCode, KeyEvent, KeyEventKind, KeyParseMode, KeyParser, Modifiers, parse_key};
pub use line::{ConstrainedLine, LineError};
pub use overlay::{
    Margins, OverlayAnchor, OverlayId, OverlayOptions, OverlayPosition, ResponsiveVisibility,
    SizeValue,
};
pub use renderer::{CURSOR_MARKER, DifferentialRenderer, Frame, FrameCursor, RenderStats};
pub use terminal::{CrosstermTerminal, Terminal, TerminalEvent};
pub use theme::{Color, EditorTheme, MarkdownTheme, SelectTheme, Style, Theme};
pub use tui::Tui;
pub use virtual_terminal::VirtualTerminal;

/// Error type shared by rendering and component APIs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A component produced a line wider than the render area.
    #[error(transparent)]
    Line(#[from] LineError),
    /// Terminal I/O failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// A component or overlay identifier was not mounted.
    #[error("component {0:?} is not mounted")]
    MissingComponent(ComponentId),
    /// A terminal image descriptor was invalid.
    #[error("invalid terminal image: {0}")]
    InvalidImage(String),
}

/// Result alias for TUI operations.
pub type Result<T> = std::result::Result<T, Error>;
