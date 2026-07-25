//! Width-constrained rendered lines.

use std::fmt;

use crate::ansi::{truncate_to_width, visible_width};

/// Failure to satisfy a line's terminal-width contract.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum LineError {
    /// The line exceeded the component's render width.
    #[error("rendered line is {actual} cells wide, but only {maximum} are available")]
    TooWide {
        /// Measured terminal-cell width.
        actual: usize,
        /// Maximum permitted terminal-cell width.
        maximum: usize,
    },
}

/// A rendered line proven not to exceed its declared terminal width.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConstrainedLine {
    text: String,
    width: usize,
    limit: usize,
}

impl ConstrainedLine {
    /// Validates and constructs a line.
    ///
    /// # Errors
    ///
    /// Returns [`LineError::TooWide`] when the visible cell width exceeds
    /// `maximum`.
    pub fn new(text: impl Into<String>, maximum: usize) -> Result<Self, LineError> {
        let text = text.into();
        let width = visible_width(&text);
        if width > maximum {
            return Err(LineError::TooWide {
                actual: width,
                maximum,
            });
        }
        Ok(Self {
            text,
            width,
            limit: maximum,
        })
    }

    /// Constructs an empty line for a render area.
    pub fn empty(maximum: usize) -> Self {
        Self {
            text: String::new(),
            width: 0,
            limit: maximum,
        }
    }

    /// Truncates input rather than returning an overflow error.
    pub fn truncated(text: impl AsRef<str>, maximum: usize, ellipsis: &str) -> Self {
        let text = truncate_to_width(text.as_ref(), maximum, ellipsis, false);
        let width = visible_width(&text);
        Self {
            text,
            width,
            limit: maximum,
        }
    }

    /// Returns the raw text, including ANSI sequences.
    pub fn as_str(&self) -> &str {
        &self.text
    }

    /// Consumes the line.
    pub fn into_string(self) -> String {
        self.text
    }

    /// Returns its measured terminal-cell width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Returns the width used to validate the line.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Returns a copy padded to exactly the declared limit.
    pub fn padded(&self) -> String {
        let mut output = self.text.clone();
        output.push_str(&" ".repeat(self.limit - self.width));
        output
    }
}

impl AsRef<str> for ConstrainedLine {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for ConstrainedLine {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cell_overflow_not_byte_length() {
        assert!(ConstrainedLine::new("界界", 3).is_err());
        let line = ConstrainedLine::new("界", 2).unwrap();
        assert_eq!(line.width(), 2);
    }
}
