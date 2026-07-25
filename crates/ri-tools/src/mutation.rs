//! Canonical-path file mutation serialization.

use std::collections::HashMap;
use std::ffi::OsString;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use tokio::sync::Mutex as AsyncMutex;

use crate::{EnvError, ExecutionEnv};

type LockMap = HashMap<PathBuf, Weak<AsyncMutex<()>>>;

fn locks() -> &'static Mutex<LockMap> {
    static LOCKS: OnceLock<Mutex<LockMap>> = OnceLock::new();
    LOCKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Run a mutation after all earlier mutations of the same canonical path.
///
/// Existing symbolic-link aliases share a lock. For a new path, the nearest
/// existing parent is canonicalized so aliases through symlinked directories
/// also serialize.
///
/// # Errors
///
/// Returns an error when the path cannot be canonicalized safely or the
/// operation itself fails.
pub async fn with_file_mutation<T, F, Fut>(
    env: &dyn ExecutionEnv,
    path: &Path,
    operation: F,
) -> Result<T, EnvError>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T, EnvError>>,
{
    let key = canonical_mutation_key(env, path).await?;
    let lock = {
        let mut registry = locks()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        registry
            .get(&key)
            .and_then(Weak::upgrade)
            .unwrap_or_else(|| {
                let lock = Arc::new(AsyncMutex::new(()));
                registry.insert(key.clone(), Arc::downgrade(&lock));
                lock
            })
    };

    let guard = Arc::clone(&lock).lock_owned().await;
    let result = operation().await;
    drop(guard);

    let mut registry = locks()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if Arc::strong_count(&lock) == 1
        && registry
            .get(&key)
            .and_then(Weak::upgrade)
            .is_some_and(|registered| Arc::ptr_eq(&registered, &lock))
    {
        registry.remove(&key);
    }
    result
}

/// Produce the canonical key used by the mutation serializer.
///
/// # Errors
///
/// Returns an error when an existing path component cannot be canonicalized
/// for a reason other than being absent.
pub async fn canonical_mutation_key(
    env: &dyn ExecutionEnv,
    path: &Path,
) -> Result<PathBuf, EnvError> {
    match env.canonicalize(path).await {
        Ok(canonical) => return Ok(normalize_key(&canonical)),
        Err(EnvError::Io(error))
            if !matches!(
                error.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
            ) =>
        {
            return Err(EnvError::Io(error));
        }
        Err(EnvError::Io(_)) => {}
        Err(error) => return Err(error),
    }

    let mut suffix: Vec<OsString> = Vec::new();
    let mut current = path;
    loop {
        match env.canonicalize(current).await {
            Ok(mut canonical) => {
                for component in suffix.iter().rev() {
                    canonical.push(component);
                }
                return Ok(normalize_key(&canonical));
            }
            Err(EnvError::Io(error))
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) => {}
            Err(error) => return Err(error),
        }
        let Some(name) = current.file_name() else {
            return Ok(normalize_key(path));
        };
        suffix.push(name.to_owned());
        let Some(parent) = current.parent() else {
            return Ok(normalize_key(path));
        };
        current = parent;
    }
}

fn normalize_key(path: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(path.to_string_lossy().to_lowercase())
    }
    #[cfg(not(windows))]
    {
        path.to_owned()
    }
}
