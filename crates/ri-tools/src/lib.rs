//! Filesystem, search, edit, and shell tools for Ri agents.
//!
//! The core is independent of the agent scheduler. Callers can use the typed
//! functions directly, use [`Tools`], or provide a remote [`ExecutionEnv`].

mod common;
mod error;
mod paths;

pub mod bash;
pub mod edit;
pub mod env;
pub mod find;
pub mod grep;
pub mod ls;
pub mod mutation;
pub mod output;
pub mod read;
pub mod runtime;
pub mod truncate;
pub mod write;

pub use bash::{BashDetails, BashInput, BashOptions, BashResult, BashUpdate, bash};
pub use common::{Content, ToolResult};
pub use edit::{
    Edit, EditDetails, EditInput, EditPreview, EditResult, apply_edits_to_normalized_content, edit,
    generate_unified_patch, normalize_for_fuzzy_match, normalize_to_lf, preview_edit,
};
pub use env::{
    EnvDirEntry, EnvMetadata, ExecutionEnv, LocalExecutionEnv, OutputChunk, OutputSink,
    OutputStream, ProcessExit, ProcessRequest, WalkEntry, WalkOptions,
};
pub use error::{EnvError, ToolError};
pub use find::{FindBackend, FindDetails, FindInput, FindResult, find};
pub use grep::{GrepDetails, GrepInput, GrepResult, SearchBackend, grep};
pub use ls::{LsDetails, LsInput, LsResult, ls};
pub use mutation::{canonical_mutation_key, with_file_mutation};
pub use output::{OutputAccumulator, OutputAccumulatorOptions, OutputSnapshot};
pub use read::{ReadDetails, ReadInput, ReadResult, read};
pub use runtime::Tools;
pub use truncate::{
    DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, GREP_MAX_LINE_LENGTH, TruncatedBy, TruncationOptions,
    TruncationResult, format_size, truncate_head, truncate_line, truncate_tail,
};
pub use write::{WriteDetails, WriteInput, WriteResult, write};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Built-in tool identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum ToolName {
    /// Read files and images.
    Read,
    /// Create or replace files.
    Write,
    /// Apply targeted text edits.
    Edit,
    /// Execute shell source.
    Bash,
    /// Search file contents.
    Grep,
    /// Find paths by glob.
    Find,
    /// List a directory.
    Ls,
}

/// All built-in tools in stable prompt order.
pub const ALL_TOOL_NAMES: [ToolName; 7] = [
    ToolName::Read,
    ToolName::Bash,
    ToolName::Edit,
    ToolName::Write,
    ToolName::Grep,
    ToolName::Find,
    ToolName::Ls,
];
