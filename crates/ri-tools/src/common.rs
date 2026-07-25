//! Shared tool input and output primitives.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A model-facing content block returned by a tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Content {
    /// UTF-8 text.
    Text {
        /// Text payload.
        text: String,
    },
    /// Base64-encoded image bytes.
    Image {
        /// Base64 payload.
        data: String,
        /// IANA media type.
        #[serde(rename = "mimeType")]
        mime_type: String,
    },
}

impl Content {
    /// Construct a text block.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }
}

/// A typed tool response.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ToolResult<D> {
    /// Model-facing content blocks.
    pub content: Vec<Content>,
    /// Structured metadata for callers and renderers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<D>,
}

impl<D> ToolResult<D> {
    /// Construct a text-only response.
    pub fn text(text: impl Into<String>, details: Option<D>) -> Self {
        Self {
            content: vec![Content::text(text)],
            details,
        }
    }

    /// Concatenate all text blocks, matching the reference tool convention.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| match block {
                Content::Text { text } => Some(text.as_str()),
                Content::Image { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
