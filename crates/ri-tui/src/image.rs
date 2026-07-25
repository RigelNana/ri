//! Kitty and iTerm2 terminal-image descriptors.

use std::env;
use std::fmt::Write as _;
use std::io::Cursor;
use std::sync::atomic::{AtomicU32, Ordering};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::{ImageFormat, ImageReader};

static NEXT_IMAGE_ID: AtomicU32 = AtomicU32::new(1);

/// Supported inline-image protocol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageProtocol {
    /// Kitty graphics protocol (also supported by Ghostty, `WezTerm`, and Warp).
    Kitty,
    /// iTerm2 OSC 1337 inline files.
    Iterm2,
}

/// Detected terminal features.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TerminalCapabilities {
    /// Inline-image support.
    pub images: Option<ImageProtocol>,
    /// 24-bit SGR color.
    pub true_color: bool,
    /// OSC 8 hyperlinks.
    pub hyperlinks: bool,
}

impl TerminalCapabilities {
    /// Conservatively detects features from process environment variables.
    pub fn detect() -> Self {
        Self::detect_with(|name| env::var(name).ok())
    }

    /// Detects from an alternate environment source.
    pub fn detect_with(mut get: impl FnMut(&str) -> Option<String>) -> Self {
        let term = get("TERM").unwrap_or_default().to_ascii_lowercase();
        let program = get("TERM_PROGRAM").unwrap_or_default().to_ascii_lowercase();
        let emulator = get("TERMINAL_EMULATOR")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let color_term = get("COLORTERM").unwrap_or_default().to_ascii_lowercase();
        let hinted_true_color = matches!(color_term.as_str(), "truecolor" | "24bit");

        if get("TMUX").is_some() || term.starts_with("tmux") {
            return Self {
                images: None,
                true_color: hinted_true_color,
                hyperlinks: false,
            };
        }
        if term.starts_with("screen") {
            return Self {
                images: None,
                true_color: hinted_true_color,
                hyperlinks: false,
            };
        }
        if get("KITTY_WINDOW_ID").is_some() || program == "kitty" {
            return Self::full(ImageProtocol::Kitty);
        }
        if program == "ghostty"
            || term.contains("ghostty")
            || get("GHOSTTY_RESOURCES_DIR").is_some()
            || get("WEZTERM_PANE").is_some()
            || program == "wezterm"
            || program == "warpterminal"
            || get("WARP_SESSION_ID").is_some()
            || get("WARP_TERMINAL_SESSION_UUID").is_some()
        {
            return Self::full(ImageProtocol::Kitty);
        }
        if get("ITERM_SESSION_ID").is_some() || program == "iterm.app" {
            return Self::full(ImageProtocol::Iterm2);
        }
        if get("WT_SESSION").is_some() || matches!(program.as_str(), "vscode" | "alacritty") {
            return Self {
                images: None,
                true_color: true,
                hyperlinks: true,
            };
        }
        if emulator == "jetbrains-jediterm" {
            return Self {
                images: None,
                true_color: true,
                hyperlinks: false,
            };
        }
        Self {
            images: None,
            true_color: hinted_true_color,
            hyperlinks: false,
        }
    }

    fn full(protocol: ImageProtocol) -> Self {
        Self {
            images: Some(protocol),
            true_color: true,
            hyperlinks: true,
        }
    }
}

/// Terminal cell pixel dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CellDimensions {
    /// Cell width in pixels.
    pub width_px: u32,
    /// Cell height in pixels.
    pub height_px: u32,
}

impl Default for CellDimensions {
    fn default() -> Self {
        Self {
            width_px: 9,
            height_px: 18,
        }
    }
}

/// Source image pixel dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageDimensions {
    /// Pixel width.
    pub width_px: u32,
    /// Pixel height.
    pub height_px: u32,
}

/// Image footprint in terminal cells.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageCellSize {
    /// Columns.
    pub columns: u16,
    /// Rows.
    pub rows: u16,
}

/// Kitty transmission and placement descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KittyImageDescriptor {
    /// Placement columns.
    pub columns: Option<u16>,
    /// Placement rows.
    pub rows: Option<u16>,
    /// Reusable image identifier.
    pub image_id: Option<u32>,
    /// Allow Kitty's default cursor movement.
    pub move_cursor: bool,
    /// Suppress protocol replies.
    pub quiet: bool,
}

impl Default for KittyImageDescriptor {
    fn default() -> Self {
        Self {
            columns: None,
            rows: None,
            image_id: None,
            move_cursor: true,
            quiet: true,
        }
    }
}

impl KittyImageDescriptor {
    /// Encodes already-base64 image data, chunking at Kitty's 4096-byte limit.
    pub fn encode(&self, data: &str) -> String {
        const CHUNK_SIZE: usize = 4096;
        let mut parameters = vec!["a=T".to_owned(), "f=100".to_owned()];
        if self.quiet {
            parameters.push("q=2".to_owned());
        }
        if !self.move_cursor {
            parameters.push("C=1".to_owned());
        }
        if let Some(columns) = self.columns {
            parameters.push(format!("c={columns}"));
        }
        if let Some(rows) = self.rows {
            parameters.push(format!("r={rows}"));
        }
        if let Some(image_id) = self.image_id {
            parameters.push(format!("i={image_id}"));
        }
        let parameters = parameters.join(",");
        if data.len() <= CHUNK_SIZE {
            return format!("\x1b_G{parameters};{data}\x1b\\");
        }

        let mut output = String::new();
        let mut offset = 0;
        let mut first = true;
        while offset < data.len() {
            let mut end = (offset + CHUNK_SIZE).min(data.len());
            while !data.is_char_boundary(end) {
                end -= 1;
            }
            let more = end < data.len();
            if first {
                write!(
                    output,
                    "\x1b_G{parameters},m={};{}\x1b\\",
                    u8::from(more),
                    &data[offset..end]
                )
                .expect("writing to a String cannot fail");
                first = false;
            } else {
                write!(
                    output,
                    "\x1b_Gm={};{}\x1b\\",
                    u8::from(more),
                    &data[offset..end]
                )
                .expect("writing to a String cannot fail");
            }
            offset = end;
        }
        output
    }
}

/// iTerm2 inline-file descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Iterm2ImageDescriptor {
    /// Width parameter, normally a cell count.
    pub width: Option<String>,
    /// Height parameter, normally `auto`.
    pub height: Option<String>,
    /// Optional displayed filename.
    pub name: Option<String>,
    /// Preserve source aspect ratio.
    pub preserve_aspect_ratio: bool,
    /// Render inline rather than downloading.
    pub inline: bool,
}

impl Default for Iterm2ImageDescriptor {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            name: None,
            preserve_aspect_ratio: true,
            inline: true,
        }
    }
}

impl Iterm2ImageDescriptor {
    /// Encodes already-base64 image data as OSC 1337.
    pub fn encode(&self, data: &str) -> String {
        let mut parameters = vec![format!("inline={}", u8::from(self.inline))];
        if let Some(width) = &self.width {
            parameters.push(format!("width={width}"));
        }
        if let Some(height) = &self.height {
            parameters.push(format!("height={height}"));
        }
        if let Some(name) = &self.name {
            parameters.push(format!("name={}", STANDARD.encode(name.as_bytes())));
        }
        if !self.preserve_aspect_ratio {
            parameters.push("preserveAspectRatio=0".to_owned());
        }
        format!("\x1b]1337;File={}:{}\x07", parameters.join(";"), data)
    }
}

/// Result of terminal-image encoding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImageRender {
    /// Escape sequence.
    pub sequence: String,
    /// Rows reserved by the placement.
    pub rows: u16,
    /// Kitty image identifier, when applicable.
    pub image_id: Option<u32>,
}

/// Allocates a non-zero process-local Kitty image identifier.
pub fn allocate_image_id() -> u32 {
    loop {
        let id = NEXT_IMAGE_ID.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

/// Encodes image bytes for a selected terminal protocol.
pub fn render_image(
    protocol: ImageProtocol,
    bytes: &[u8],
    dimensions: ImageDimensions,
    cells: CellDimensions,
    max_width: u16,
    max_height: Option<u16>,
    image_id: Option<u32>,
) -> ImageRender {
    let data = STANDARD.encode(bytes);
    let size = calculate_image_cell_size(dimensions, cells, max_width, max_height);
    match protocol {
        ImageProtocol::Kitty => {
            let image_id = image_id.or_else(|| Some(allocate_image_id()));
            let descriptor = KittyImageDescriptor {
                columns: Some(size.columns),
                rows: Some(size.rows),
                image_id,
                move_cursor: false,
                quiet: true,
            };
            ImageRender {
                sequence: descriptor.encode(&data),
                rows: size.rows,
                image_id,
            }
        }
        ImageProtocol::Iterm2 => {
            let descriptor = Iterm2ImageDescriptor {
                width: Some(size.columns.to_string()),
                height: Some("auto".to_owned()),
                ..Iterm2ImageDescriptor::default()
            };
            ImageRender {
                sequence: descriptor.encode(&data),
                rows: size.rows,
                image_id: None,
            }
        }
    }
}

/// Calculates a bounded aspect-ratio-preserving cell footprint.
pub fn calculate_image_cell_size(
    image: ImageDimensions,
    cell: CellDimensions,
    max_width: u16,
    max_height: Option<u16>,
) -> ImageCellSize {
    let max_width = max_width.max(1);
    let max_height = max_height.map(|height| height.max(1));
    let image_width = f64::from(image.width_px.max(1));
    let image_height = f64::from(image.height_px.max(1));
    let cell_width = f64::from(cell.width_px.max(1));
    let cell_height = f64::from(cell.height_px.max(1));
    let width_scale = f64::from(max_width) * cell_width / image_width;
    let height_scale = max_height.map_or(width_scale, |height| {
        f64::from(height) * cell_height / image_height
    });
    let scale = width_scale.min(height_scale);
    let columns = ceil_cells(image_width * scale / cell_width, max_width);
    let rows_unbounded = ceil_cells(image_height * scale / cell_height, u16::MAX);
    let rows = max_height.map_or(rows_unbounded.max(1), |height| {
        rows_unbounded.clamp(1, height)
    });
    ImageCellSize { columns, rows }
}

fn ceil_cells(value: f64, maximum: u16) -> u16 {
    let bounded = value.ceil().clamp(1.0, f64::from(maximum.max(1)));
    // Clamping proves the protocol cell count is within the u16 output range.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        bounded as u16
    }
}

/// Returns true if a line contains either supported image protocol.
pub fn is_image_line(line: &str) -> bool {
    line.contains("\x1b_G") || line.contains("\x1b]1337;File=")
}

/// Deletes and frees one Kitty image.
pub fn delete_kitty_image(image_id: u32) -> String {
    format!("\x1b_Ga=d,d=I,i={image_id},q=2\x1b\\")
}

/// Deletes and frees all Kitty images.
pub const fn delete_all_kitty_images() -> &'static str {
    "\x1b_Ga=d,d=A,q=2\x1b\\"
}

/// Wraps text in an OSC 8 hyperlink.
pub fn hyperlink(text: &str, url: &str) -> String {
    let safe_url = url.replace(['\x1b', '\x07'], "");
    format!("\x1b]8;;{safe_url}\x1b\\{text}\x1b]8;;\x1b\\")
}

/// Reads dimensions from PNG, JPEG, GIF, or WebP headers.
pub fn image_dimensions(bytes: &[u8], mime_type: &str) -> Option<ImageDimensions> {
    let format = match mime_type {
        "image/png" => ImageFormat::Png,
        "image/jpeg" => ImageFormat::Jpeg,
        "image/gif" => ImageFormat::Gif,
        "image/webp" => ImageFormat::WebP,
        _ => return None,
    };
    let (width_px, height_px) = ImageReader::with_format(Cursor::new(bytes), format)
        .into_dimensions()
        .ok()?;
    Some(ImageDimensions {
        width_px,
        height_px,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kitty_chunks_and_describes_placement() {
        let descriptor = KittyImageDescriptor {
            columns: Some(3),
            rows: Some(2),
            image_id: Some(42),
            move_cursor: false,
            quiet: true,
        };
        let encoded = descriptor.encode(&"A".repeat(5000));
        assert!(encoded.starts_with("\x1b_Ga=T,f=100,q=2,C=1,c=3,r=2,i=42,m=1;"));
        assert!(encoded.contains("\x1b_Gm=0;"));
    }

    #[test]
    fn parses_png_dimensions() {
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgba8(640, 480)
            .write_to(&mut png, ImageFormat::Png)
            .expect("encode PNG");
        assert_eq!(
            image_dimensions(png.get_ref(), "image/png"),
            Some(ImageDimensions {
                width_px: 640,
                height_px: 480
            })
        );
    }

    #[test]
    fn iterm2_descriptor_escapes_name_and_dimensions() {
        let descriptor = Iterm2ImageDescriptor {
            width: Some("12".to_owned()),
            height: Some("auto".to_owned()),
            name: Some("diagram.png".to_owned()),
            preserve_aspect_ratio: true,
            inline: true,
        };
        let encoded = descriptor.encode("YWJj");
        assert!(encoded.starts_with("\x1b]1337;File=inline=1;width=12;height=auto;name="));
        assert!(encoded.ends_with(":YWJj\x07"));
        assert!(is_image_line(&encoded));
    }

    #[test]
    fn calculates_bounded_cell_size_and_parses_webp_extended() {
        assert_eq!(
            calculate_image_cell_size(
                ImageDimensions {
                    width_px: 400,
                    height_px: 200,
                },
                CellDimensions {
                    width_px: 10,
                    height_px: 20,
                },
                20,
                Some(8),
            ),
            ImageCellSize {
                columns: 20,
                rows: 5,
            }
        );
        let mut webp = std::io::Cursor::new(Vec::new());
        image::DynamicImage::new_rgb8(512, 128)
            .write_to(&mut webp, ImageFormat::WebP)
            .expect("encode WebP");
        assert_eq!(
            image_dimensions(webp.get_ref(), "image/webp"),
            Some(ImageDimensions {
                width_px: 512,
                height_px: 128,
            })
        );
    }
}
