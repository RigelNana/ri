//! Transactional `SQLite` storage for [`ri_session`].
//!
//! The repository configures write-ahead logging, `FULL` synchronous writes,
//! foreign keys, and a busy timeout before applying versioned migrations.
//! Authoritative entries, monotonic sequences, the durable leaf, aggregate
//! statistics, labels, and the active branch are updated in one transaction.
//! Materialized state is rebuilt from entries whenever the database reopens.

mod migrations;
mod repository;

pub use repository::{
    MaterializedSession, PragmaSettings, SqliteRepository, SqliteSessionRepo,
    SqliteSessionRepository,
};
pub use ri_session::{Error, Result};

#[cfg(test)]
mod tests;
