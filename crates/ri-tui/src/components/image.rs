use crate::Result;
use crate::ansi::truncate_to_width;
use crate::component::{Component, RenderContext};
use crate::image::{
    CellDimensions, ImageDimensions, Iterm2ImageDescriptor, KittyImageDescriptor,
    TerminalCapabilities, calculate_image_cell_size, is_image_line, render_image,
};
use crate::line::ConstrainedLine;

/// Terminal image component reserving its image cell rectangle.
#[derive(Clone, Debug)]
pub struct Image {
    sequence: String,
    columns: usize,
    rows: usize,
    fallback: String,
}

impl Image {
    /// Creates an image from an already encoded terminal control sequence.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidImage`] when `sequence` is not a Kitty
    /// or iTerm2 image control sequence.
    pub fn new(sequence: impl Into<String>, columns: usize, rows: usize) -> Result<Self> {
        let sequence = sequence.into();
        if !is_image_line(&sequence) {
            return Err(crate::Error::InvalidImage(
                "sequence is not Kitty or iTerm2 graphics".to_owned(),
            ));
        }
        Ok(Self {
            sequence,
            columns: columns.max(1),
            rows: rows.max(1),
            fallback: "[image]".to_owned(),
        })
    }

    /// Encodes a Kitty image.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoded descriptor does not produce a valid
    /// terminal image sequence.
    pub fn kitty(descriptor: &KittyImageDescriptor, base64_data: &str) -> Result<Self> {
        Self::new(
            descriptor.encode(base64_data),
            descriptor.columns.map_or(1, usize::from),
            descriptor.rows.map_or(1, usize::from),
        )
    }

    /// Encodes an iTerm2 image.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoded descriptor does not produce a valid
    /// terminal image sequence.
    pub fn iterm2(
        descriptor: &Iterm2ImageDescriptor,
        base64_data: &str,
        columns: usize,
        rows: usize,
    ) -> Result<Self> {
        Self::new(descriptor.encode(base64_data), columns, rows)
    }

    /// Selects the best protocol from terminal capabilities.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidImage`] when no image protocol was
    /// detected or encoding does not produce a valid image sequence.
    pub fn from_bytes(
        capabilities: TerminalCapabilities,
        bytes: &[u8],
        dimensions: ImageDimensions,
        cells: CellDimensions,
        max_width: u16,
        max_height: Option<u16>,
        image_id: Option<u32>,
    ) -> Result<Self> {
        let protocol = capabilities.images.ok_or_else(|| {
            crate::Error::InvalidImage("terminal has no detected image protocol".to_owned())
        })?;
        let size = calculate_image_cell_size(dimensions, cells, max_width, max_height);
        let rendered = render_image(
            protocol, bytes, dimensions, cells, max_width, max_height, image_id,
        );
        Self::new(
            rendered.sequence,
            usize::from(size.columns),
            usize::from(size.rows),
        )
    }

    /// Changes plain-text fallback shown when the image is too wide.
    #[must_use]
    pub fn with_fallback(mut self, fallback: impl Into<String>) -> Self {
        self.fallback = fallback.into();
        self
    }
}

impl Component for Image {
    fn render(&mut self, context: RenderContext) -> Result<Vec<ConstrainedLine>> {
        if self.columns > context.width {
            return Ok(vec![ConstrainedLine::new(
                truncate_to_width(&self.fallback, context.width, "", false),
                context.width,
            )?]);
        }
        let mut lines = Vec::with_capacity(self.rows);
        lines.push(ConstrainedLine::new(self.sequence.clone(), context.width)?);
        lines.extend(
            (1..self.rows.min(context.height.max(1)))
                .map(|_| ConstrainedLine::empty(context.width)),
        );
        Ok(lines)
    }
}
