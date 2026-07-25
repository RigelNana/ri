//! SDK construction failures.

use thiserror::Error;

/// Error returned while resolving or constructing a session runtime.
#[derive(Debug, Error)]
pub enum Error {
    /// A required builder choice was omitted.
    #[error("session builder is missing {0}")]
    Missing(&'static str),
    /// A model id was not present in the configured catalog.
    #[error("model {provider}/{model} is not registered")]
    ModelNotFound {
        /// Provider identifier.
        provider: String,
        /// Model identifier.
        model: String,
    },
    /// A filesystem path required by the session format is not valid UTF-8.
    #[error("path is not valid UTF-8: {0}")]
    NonUtf8Path(std::path::PathBuf),
    /// Host filesystem setup failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Built-in tool construction failed.
    #[error(transparent)]
    Tool(#[from] ri_agent::ToolError),
    /// Provider/model/auth runtime failure.
    #[error(transparent)]
    Ai(#[from] ri_ai::AiError),
    /// Session repository failure.
    #[error(transparent)]
    Session(#[from] ri_session::Error),
    /// Harness construction or operation failure.
    #[error(transparent)]
    Harness(#[from] ri_harness::Error),
}

/// SDK result alias.
pub type Result<T> = std::result::Result<T, Error>;
