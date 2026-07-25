//! Durable session resolution and interchange.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ri_session::{
    CURRENT_SESSION_VERSION, CreateOptions, ForkOptions, ForkPosition, JsonlRepository,
    ListOptions, MemoryRepository, Repository, Session, SessionEntry, SessionHeader,
    SessionMetadata, SessionSnapshot,
};

use crate::cli::{SessionForkArgs, SessionFormat, SessionImportArgs};
use crate::error::{CliError, Result};

#[derive(Clone)]
pub(super) struct SessionRepository {
    repository: Arc<dyn Repository>,
    jsonl: Option<Arc<JsonlRepository>>,
}

impl std::fmt::Debug for SessionRepository {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SessionRepository")
            .field("repository", &self.repository)
            .field("jsonl", &self.jsonl)
            .finish()
    }
}

impl SessionRepository {
    pub(super) fn durable(root: PathBuf) -> Self {
        let jsonl = Arc::new(JsonlRepository::new(root));
        Self {
            repository: jsonl.clone(),
            jsonl: Some(jsonl),
        }
    }

    pub(super) fn memory() -> Self {
        Self {
            repository: Arc::new(MemoryRepository::default()),
            jsonl: None,
        }
    }

    pub(super) async fn create(&self, options: CreateOptions) -> Result<Session> {
        self.repository
            .create(options)
            .await
            .map_err(|error| CliError::runtime("create session", error))
    }

    pub(super) async fn list(&self, cwd: Option<String>) -> Result<Vec<SessionMetadata>> {
        self.repository
            .list(ListOptions { cwd })
            .await
            .map_err(|error| CliError::runtime("list sessions", error))
    }

    pub(super) async fn open_target(&self, target: &str) -> Result<Session> {
        let path = Path::new(target);
        if tokio::fs::metadata(path).await.is_ok() {
            let jsonl = self.jsonl.as_ref().ok_or_else(|| {
                CliError::unsupported(
                    "open a session path",
                    "the invocation uses an in-memory session repository",
                )
            })?;
            return jsonl
                .open_path(path)
                .await
                .map_err(|error| CliError::runtime("open session path", error));
        }

        let id = self.resolve_id(target).await?;
        self.repository
            .open(&id)
            .await
            .map_err(|error| CliError::runtime("open session", error))
    }

    pub(super) async fn resolve_id(&self, target: &str) -> Result<String> {
        let sessions = self.list(None).await?;
        if sessions.iter().any(|session| session.id == target) {
            return Ok(target.to_owned());
        }
        let matches = sessions
            .iter()
            .filter(|session| session.id.starts_with(target))
            .map(|session| session.id.clone())
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [id] => Ok(id.clone()),
            [] => Err(CliError::NotFound {
                kind: "session",
                name: target.to_owned(),
            }),
            _ => Err(CliError::InvalidArguments(format!(
                "session prefix `{target}` is ambiguous (matches {})",
                matches.join(", ")
            ))),
        }
    }

    pub(super) async fn fork(&self, arguments: &SessionForkArgs) -> Result<Session> {
        if tokio::fs::metadata(&arguments.source).await.is_ok() {
            let source = self.open_target(&arguments.source).await?;
            return self.copy_external(source, arguments).await;
        }
        let source_id = self.resolve_id(&arguments.source).await?;
        self.repository
            .fork(
                &source_id,
                ForkOptions {
                    id: arguments.id.clone(),
                    cwd: arguments
                        .cwd
                        .as_ref()
                        .map(|path| path.to_string_lossy().into_owned()),
                    entry_id: arguments.entry.clone(),
                    position: if arguments.at {
                        ForkPosition::At
                    } else {
                        ForkPosition::Before
                    },
                    metadata: None,
                },
            )
            .await
            .map_err(|error| CliError::runtime("fork session", error))
    }

    async fn copy_external(&self, source: Session, arguments: &SessionForkArgs) -> Result<Session> {
        let snapshot = source
            .snapshot()
            .await
            .map_err(|error| CliError::runtime("read source session", error))?;
        let entries = fork_entries(&snapshot, arguments)?;
        let destination = self
            .create(CreateOptions {
                id: arguments.id.clone(),
                cwd: arguments.cwd.as_ref().map_or_else(
                    || snapshot.header().cwd.clone(),
                    |path| path.to_string_lossy().into_owned(),
                ),
                parent_session: Some(snapshot.header().id.clone()),
                metadata: snapshot.header().metadata.clone(),
            })
            .await?;
        for entry in entries {
            destination
                .append_entry(entry)
                .await
                .map_err(|error| CliError::runtime("copy fork entry", error))?;
        }
        Ok(destination)
    }

    pub(super) async fn import(&self, arguments: &SessionImportArgs) -> Result<Session> {
        let bytes = tokio::fs::read(&arguments.input)
            .await
            .map_err(|source| CliError::Io {
                operation: "read session import",
                source,
            })?;
        let imported = match arguments.format {
            SessionFormat::Native => parse_native(&bytes)?,
            SessionFormat::Pi => parse_pi(&bytes)?,
        };
        let destination = self
            .create(CreateOptions {
                id: arguments.id.clone(),
                cwd: arguments.cwd.as_ref().map_or_else(
                    || imported.header.cwd.clone(),
                    |path| path.to_string_lossy().into_owned(),
                ),
                parent_session: Some(imported.header.id),
                metadata: imported.header.metadata,
            })
            .await?;
        for entry in imported.entries {
            destination
                .append_entry(entry)
                .await
                .map_err(|error| CliError::runtime("append imported session entry", error))?;
        }
        Ok(destination)
    }
}

struct ImportedSession {
    header: SessionHeader,
    entries: Vec<SessionEntry>,
}

pub(super) async fn export(session: &Session, format: SessionFormat) -> Result<String> {
    let header = session
        .header()
        .await
        .map_err(|error| CliError::runtime("read session header", error))?;
    let entries = session
        .entries(None, None)
        .await
        .map_err(|error| CliError::runtime("read session entries", error))?
        .into_iter()
        .map(|entry| entry.entry)
        .collect::<Vec<_>>();

    match format {
        SessionFormat::Native => {
            let mut output = json_line(&header)?;
            for entry in entries {
                output.push_str(&json_line(&entry)?);
            }
            Ok(output)
        }
        SessionFormat::Pi => {
            let entries = entries
                .into_iter()
                .map(|entry| {
                    ri_compat::native_entry_to_pi(entry).map_err(|error| {
                        CliError::runtime("convert a session entry for Pi export", error)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let header = ri_compat::native_header_to_pi(header).map_err(|error| {
                CliError::runtime("convert the session header for Pi export", error)
            })?;
            let pi = ri_compat::PiSession::new(header, entries);
            let bytes = ri_compat::export_session(&pi, ri_compat::PiSessionVersion::V3)
                .map_err(|error| CliError::runtime("export Pi session", error))?;
            String::from_utf8(bytes).map_err(|error| {
                CliError::InvalidArguments(format!(
                    "Pi session exporter returned non-UTF-8 data: {error}"
                ))
            })
        }
    }
}

fn parse_native(bytes: &[u8]) -> Result<ImportedSession> {
    let text = std::str::from_utf8(bytes).map_err(|error| {
        CliError::InvalidArguments(format!("native session is not UTF-8: {error}"))
    })?;
    let mut records = text.lines().filter(|line| !line.trim().is_empty());
    let header_line = records
        .next()
        .ok_or_else(|| CliError::InvalidArguments("native session import is empty".to_owned()))?;
    let header: SessionHeader =
        serde_json::from_str(header_line).map_err(|source| CliError::Json {
            operation: "decoding a native session header",
            source,
        })?;
    if header.version != CURRENT_SESSION_VERSION {
        return Err(CliError::InvalidArguments(format!(
            "native session version {} is unsupported; expected {CURRENT_SESSION_VERSION}",
            header.version
        )));
    }
    let entries = records
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|source| CliError::Json {
                operation: if index == 0 {
                    "decoding the first native session entry"
                } else {
                    "decoding a native session entry"
                },
                source,
            })
        })
        .collect::<Result<Vec<SessionEntry>>>()?;
    Ok(ImportedSession { header, entries })
}

fn parse_pi(bytes: &[u8]) -> Result<ImportedSession> {
    let imported = ri_compat::import_session(bytes)
        .map_err(|error| CliError::runtime("import Pi session", error))?;
    let header = ri_compat::pi_header_to_native(imported.header)
        .map_err(|error| CliError::runtime("convert an imported Pi session header", error))?;
    let entries = imported
        .entries
        .into_iter()
        .map(|entry| {
            ri_compat::pi_entry_to_native(entry)
                .map_err(|error| CliError::runtime("convert an imported Pi session entry", error))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ImportedSession { header, entries })
}

fn fork_entries(
    snapshot: &SessionSnapshot,
    arguments: &SessionForkArgs,
) -> Result<Vec<SessionEntry>> {
    let Some(target_id) = arguments.entry.as_deref() else {
        return Ok(snapshot
            .entries()
            .iter()
            .map(|entry| entry.entry.clone())
            .collect());
    };
    let target = snapshot
        .entry(target_id)
        .ok_or_else(|| CliError::NotFound {
            kind: "session entry",
            name: target_id.to_owned(),
        })?;
    let effective_id = if arguments.at {
        Some(target_id)
    } else {
        let SessionEntry::Message(message) = &target.entry else {
            return Err(CliError::InvalidArguments(format!(
                "session entry `{target_id}` is not a user message"
            )));
        };
        if message
            .message
            .get("role")
            .and_then(serde_json::Value::as_str)
            != Some("user")
        {
            return Err(CliError::InvalidArguments(format!(
                "session entry `{target_id}` is not a user message"
            )));
        }
        message.base.parent_id.as_deref()
    };
    snapshot
        .path_to(effective_id)
        .map(|entries| {
            entries
                .into_iter()
                .map(|entry| entry.entry.clone())
                .collect()
        })
        .map_err(|error| CliError::runtime("select session fork path", error))
}

fn json_line(value: &impl serde::Serialize) -> Result<String> {
    let mut line = serde_json::to_string(value).map_err(|source| CliError::Json {
        operation: "encoding a native session record",
        source,
    })?;
    line.push('\n');
    Ok(line)
}
