//! Path normalization shared by built-in tools.

use std::path::{Path, PathBuf};

/// Resolve a model-supplied path relative to a working directory.
pub(crate) fn resolve_path(path: &Path, cwd: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    let raw = raw.strip_prefix('@').unwrap_or(&raw);
    let raw = normalize_unicode_spaces(raw);
    let expanded = expand_home(&raw);
    let candidate = PathBuf::from(expanded);
    if candidate.is_absolute() {
        candidate
    } else {
        cwd.join(candidate)
    }
}

/// Render a path with forward slashes.
pub(crate) fn to_posix(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Return a stable path relative to `root` where possible.
pub(crate) fn relative_display(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .map_or_else(|_| to_posix(path), to_posix)
}

fn normalize_unicode_spaces(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '\u{00a0}' | '\u{2002}'..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

fn expand_home(value: &str) -> String {
    let is_home = value == "~";
    let has_separator = value
        .strip_prefix('~')
        .is_some_and(|rest| rest.starts_with('/') || rest.starts_with('\\'));
    if !is_home && !has_separator {
        return value.to_owned();
    }
    let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) else {
        return value.to_owned();
    };
    let mut path = PathBuf::from(home);
    if value.len() > 1 {
        path.push(&value[2..]);
    }
    path.to_string_lossy().into_owned()
}
