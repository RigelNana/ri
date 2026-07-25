//! Shared temporary-file and replace semantics for small durable files.

use std::io;
use std::path::Path;

use uuid::Uuid;

/// Options for one atomic file replacement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AtomicWriteOptions {
    /// Unix permission bits applied to the temporary file before replacement.
    pub unix_mode: Option<u32>,
}

/// Writes `bytes` to a sibling temporary file and replaces `path`.
///
/// Parent directories are created automatically. On Windows, where rename
/// does not replace an existing destination, the destination is removed under
/// the caller's serialization lock before retrying the rename.
///
/// # Errors
///
/// Returns the underlying filesystem error. A failed temporary write or rename
/// is cleaned up on a best-effort basis.
pub async fn atomic_write(
    path: &Path,
    bytes: &[u8],
    options: AtomicWriteOptions,
) -> io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    tokio::fs::create_dir_all(parent).await?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("file");
    let temporary = parent.join(format!(".{name}.{}.tmp", Uuid::new_v4()));

    if let Err(error) = tokio::fs::write(&temporary, bytes).await {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error);
    }
    if let Err(error) = set_unix_mode(&temporary, options.unix_mode).await {
        let _ = tokio::fs::remove_file(&temporary).await;
        return Err(error);
    }

    match tokio::fs::rename(&temporary, path).await {
        Ok(()) => Ok(()),
        #[cfg(windows)]
        Err(_error) if tokio::fs::try_exists(path).await.unwrap_or(false) => {
            if let Err(error) = tokio::fs::remove_file(path).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error);
            }
            if let Err(error) = tokio::fs::rename(&temporary, path).await {
                let _ = tokio::fs::remove_file(&temporary).await;
                return Err(error);
            }
            Ok(())
        }
        Err(error) => {
            let _ = tokio::fs::remove_file(&temporary).await;
            Err(error)
        }
    }
}

#[cfg(unix)]
async fn set_unix_mode(path: &Path, mode: Option<u32>) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    if let Some(mode) = mode {
        tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).await?;
    }
    Ok(())
}

#[cfg(not(unix))]
#[allow(clippy::unused_async)]
async fn set_unix_mode(_path: &Path, _mode: Option<u32>) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn creates_parent_and_replaces_existing_file() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("nested/settings.json");
        atomic_write(&path, b"first", AtomicWriteOptions::default())
            .await
            .unwrap();
        atomic_write(&path, b"second", AtomicWriteOptions::default())
            .await
            .unwrap();
        assert_eq!(tokio::fs::read(&path).await.unwrap(), b"second");
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }
}
