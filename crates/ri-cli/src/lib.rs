//! Command-line frontend for the `ri` agent runtime.
//!
//! Argument parsing, mode selection, and I/O bindings live here so every
//! frontend mode can share one [`ri_sdk::SessionRuntime`].

pub mod app;
pub mod cli;
mod credential_store;
pub mod error;
pub mod input;
#[cfg(feature = "interactive")]
mod interactive;
pub mod mode;
pub mod modes;
pub mod output;
mod package_runtime;
#[cfg(feature = "rpc")]
pub mod rpc_io;
#[cfg(feature = "rpc")]
pub mod rpc_mode;
pub mod runtime;
pub mod sdk_adapter;

pub use cli::Cli;
pub use error::{CliError, Result};
pub use mode::{IoCapabilities, RunMode, select_mode};
