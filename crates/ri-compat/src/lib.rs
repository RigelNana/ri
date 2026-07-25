//! Explicit compatibility with Pi's public, non-secret formats.
//!
//! Every import function consumes caller-provided bytes. The crate never
//! searches `.pi`, reads `auth.json`, executes key commands, interpolates
//! environment variables, or mutates the source document.

pub mod models;
pub mod rpc;
pub mod session;
pub mod session_convert;
pub mod settings;

pub use models::*;
pub use rpc::*;
pub use session::{
    PiSession, PiSessionError, PiSessionHeader, PiSessionVersion, export_session, import_session,
    import_session_with_ids,
};
pub use session_convert::{
    SessionConversionError, native_entry_to_pi, native_header_to_pi, pi_entry_to_native,
    pi_header_to_native,
};
pub use settings::*;
