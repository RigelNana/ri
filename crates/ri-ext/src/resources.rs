//! Skill, prompt-template, and context resource discovery.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use indexmap::IndexMap;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::extension::GenerationClock;
use crate::source::{
    Diagnostic, ResourceKind, SourceInfo, SourceKind, SourceOrigin, SourceScope, canonical_key,
};

const CONTEXT_CANDIDATES: &[&str] = &["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"];
const IGNORE_FILES: &[&str] = &[".gitignore", ".ignore", ".fdignore"];
const MAX_SKILL_NAME: usize = 64;
const MAX_SKILL_DESCRIPTION: usize = 1024;

/// YAML frontmatter parsing failure.
#[derive(Debug, Error)]
pub enum FrontmatterError {
    /// YAML metadata could not be deserialized into the requested type.
    #[error("invalid YAML frontmatter: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// Parse optional YAML frontmatter and return a normalized, trimmed body.
///
/// # Errors
///
/// Returns [`FrontmatterError::Yaml`] when a present frontmatter block cannot
/// be deserialized as `T`.
pub fn parse_frontmatter<T>(content: &str) -> Result<(T, String), FrontmatterError>
where
    T: DeserializeOwned + Default,
{
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    let Some(rest) = normalized.strip_prefix("---\n") else {
        return Ok((T::default(), normalized));
    };
    let Some(end) = rest.find("\n---") else {
        return Ok((T::default(), normalized));
    };
    let yaml = &rest[..end];
    let body = rest[end + 4..].trim().to_owned();
    let frontmatter = if yaml.trim().is_empty() {
        T::default()
    } else {
        serde_yaml::from_str(yaml)?
    };
    Ok((frontmatter, body))
}

/// Path plus provenance supplied to a resource loader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourcePath {
    /// File or directory supplied to discovery.
    pub path: PathBuf,
    /// Provenance retained by resources discovered below the path.
    pub source: SourceInfo,
}

impl ResourcePath {
    /// Construct an explicitly configured resource path in the given scope.
    pub fn configured(path: impl Into<PathBuf>, scope: SourceScope) -> Self {
        let path = path.into();
        Self {
            source: SourceInfo::configured(&path, scope),
            path,
        }
    }
}

/// Agent Skill metadata. Full instructions stay on disk for progressive
/// disclosure and explicit `/skill:name` expansion.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Skill {
    /// Slash-command and model-visible skill name.
    pub name: String,
    /// Short model-visible summary.
    pub description: String,
    /// Markdown file containing the full instructions.
    pub file_path: PathBuf,
    /// Directory used to resolve relative skill references.
    pub base_dir: PathBuf,
    /// Discovery provenance.
    pub source: SourceInfo,
    /// Whether the skill is hidden from model-driven discovery.
    pub disable_model_invocation: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct SkillFrontmatter {
    name: Option<String>,
    description: Option<String>,
    #[serde(rename = "disable-model-invocation")]
    disable_model_invocation: bool,
}

/// Skill discovery output.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SkillLoadResult {
    /// Valid, first-wins skills in deterministic order.
    pub skills: Vec<Skill>,
    /// Validation, read, and collision diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

/// Discover and first-wins deduplicate skills from ordered paths.
pub fn load_skills(paths: &[ResourcePath]) -> SkillLoadResult {
    let mut ordered = paths.to_vec();
    ordered.sort_by_key(|entry| entry.source.precedence_rank());

    let mut result = SkillLoadResult::default();
    let mut by_name = IndexMap::<String, Skill>::new();
    let mut seen_files = BTreeSet::<PathBuf>::new();

    for entry in ordered {
        if !entry.path.exists() && entry.source.source.is_auto() {
            continue;
        }
        let discovered = discover_skill_path(&entry);
        result.diagnostics.extend(discovered.diagnostics);
        for skill in discovered.skills {
            let canonical = canonical_key(&skill.file_path);
            if seen_files.contains(&canonical) {
                continue;
            }
            if let Some(winner) = by_name.get(&skill.name) {
                result.diagnostics.push(Diagnostic::collision(
                    ResourceKind::Skill,
                    skill.name.clone(),
                    winner.source.clone(),
                    skill.source.clone(),
                ));
                continue;
            }
            seen_files.insert(canonical);
            by_name.insert(skill.name.clone(), skill);
        }
    }
    result.skills = by_name.into_values().collect();
    result
}

fn discover_skill_path(entry: &ResourcePath) -> SkillLoadResult {
    if !entry.path.exists() {
        return SkillLoadResult {
            skills: Vec::new(),
            diagnostics: vec![Diagnostic::warning(
                "skill path does not exist",
                entry.source.with_path(&entry.path),
            )],
        };
    }
    if entry.path.is_file() {
        if !has_markdown_extension(&entry.path) {
            return SkillLoadResult {
                skills: Vec::new(),
                diagnostics: vec![Diagnostic::warning(
                    "skill path is not a markdown file",
                    entry.source.with_path(&entry.path),
                )],
            };
        }
        return load_skill_file(&entry.path, &entry.source);
    }

    let mut visited = BTreeSet::new();
    let ignore_files = Vec::new();
    scan_skill_dir(
        &entry.path,
        &entry.path,
        true,
        &entry.source,
        &ignore_files,
        &mut visited,
    )
}

fn scan_skill_dir(
    directory: &Path,
    root: &Path,
    include_root_files: bool,
    source: &SourceInfo,
    inherited_ignore_files: &[PathBuf],
    visited: &mut BTreeSet<PathBuf>,
) -> SkillLoadResult {
    let mut result = SkillLoadResult::default();
    let canonical = canonical_key(directory);
    if !visited.insert(canonical) {
        return result;
    }
    let mut ignore_files = inherited_ignore_files.to_vec();
    ignore_files.extend(
        IGNORE_FILES
            .iter()
            .map(|name| directory.join(name))
            .filter(|path| path.is_file()),
    );
    let ignore = build_ignore_matcher(root, &ignore_files, source, &mut result.diagnostics);

    let mut entries = match fs::read_dir(directory) {
        Ok(entries) => {
            let mut collected = Vec::new();
            for entry in entries {
                match entry {
                    Ok(entry) => collected.push(entry),
                    Err(error) => result.diagnostics.push(Diagnostic::warning(
                        format!("failed to read skill directory entry: {error}"),
                        source.with_path(directory),
                    )),
                }
            }
            collected
        }
        Err(error) => {
            result.diagnostics.push(Diagnostic::warning(
                format!("failed to read skill directory: {error}"),
                source.with_path(directory),
            ));
            return result;
        }
    };
    entries.sort_by_key(std::fs::DirEntry::file_name);

    // A directory's SKILL.md defines that directory as one skill root and
    // prevents accidental nested-skill discovery below it.
    if let Some(skill_file) = entries.iter().find_map(|entry| {
        (entry.file_name().to_string_lossy() == "SKILL.md")
            .then(|| entry.path())
            .filter(|path| path.is_file() && !is_ignored(path, false, ignore.as_ref()))
    }) {
        return load_skill_file(&skill_file, source);
    }

    for entry in entries {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with('.') || name == "node_modules" {
            continue;
        }
        let path = entry.path();
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => {
                result.diagnostics.push(Diagnostic::warning(
                    format!("failed to inspect skill path: {error}"),
                    source.with_path(&path),
                ));
                continue;
            }
        };
        if is_ignored(&path, metadata.is_dir(), ignore.as_ref()) {
            continue;
        }
        if metadata.is_dir() {
            let nested = scan_skill_dir(&path, root, false, source, &ignore_files, visited);
            result.skills.extend(nested.skills);
            result.diagnostics.extend(nested.diagnostics);
        } else if include_root_files && metadata.is_file() && has_markdown_extension(&path) {
            let loaded = load_skill_file(&path, source);
            result.skills.extend(loaded.skills);
            result.diagnostics.extend(loaded.diagnostics);
        }
    }
    result
}

fn load_skill_file(path: &Path, source: &SourceInfo) -> SkillLoadResult {
    let mut result = SkillLoadResult::default();
    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(error) => {
            result.diagnostics.push(Diagnostic::warning(
                format!("failed to read skill: {error}"),
                source.with_path(path),
            ));
            return result;
        }
    };
    let (frontmatter, _body) = match parse_frontmatter::<SkillFrontmatter>(&content) {
        Ok(parsed) => parsed,
        Err(error) => {
            result.diagnostics.push(Diagnostic::warning(
                error.to_string(),
                source.with_path(path),
            ));
            return result;
        }
    };
    let base_dir = path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let fallback_name = base_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned();
    let name = frontmatter.name.unwrap_or(fallback_name);
    for message in validate_skill_name(&name) {
        result
            .diagnostics
            .push(Diagnostic::warning(message, source.with_path(path)));
    }
    let Some(description) = frontmatter.description else {
        result.diagnostics.push(Diagnostic::warning(
            "description is required",
            source.with_path(path),
        ));
        return result;
    };
    if description.trim().is_empty() {
        result.diagnostics.push(Diagnostic::warning(
            "description is required",
            source.with_path(path),
        ));
        return result;
    }
    if description.chars().count() > MAX_SKILL_DESCRIPTION {
        result.diagnostics.push(Diagnostic::warning(
            format!(
                "description exceeds {MAX_SKILL_DESCRIPTION} characters ({})",
                description.chars().count()
            ),
            source.with_path(path),
        ));
    }
    result.skills.push(Skill {
        name,
        description,
        file_path: path.to_path_buf(),
        base_dir,
        source: source.with_path(path),
        disable_model_invocation: frontmatter.disable_model_invocation,
    });
    result
}

fn validate_skill_name(name: &str) -> Vec<String> {
    let mut messages = Vec::new();
    let length = name.chars().count();
    if length > MAX_SKILL_NAME {
        messages.push(format!(
            "name exceeds {MAX_SKILL_NAME} characters ({length})"
        ));
    }
    if name.is_empty()
        || !name.chars().all(|character| {
            character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
        })
    {
        messages.push(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
                .to_owned(),
        );
    }
    if name.starts_with('-') || name.ends_with('-') {
        messages.push("name must not start or end with a hyphen".to_owned());
    }
    if name.contains("--") {
        messages.push("name must not contain consecutive hyphens".to_owned());
    }
    messages
}

/// Format discoverable skills for progressive disclosure in a system prompt.
pub fn format_skills_for_prompt(skills: &[Skill]) -> String {
    let visible = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect::<Vec<_>>();
    if visible.is_empty() {
        return String::new();
    }
    let mut lines = vec![
        String::new(),
        String::new(),
        "The following skills provide specialized instructions for specific tasks.".to_owned(),
        "Use the read tool to load a skill's file when the task matches its description."
            .to_owned(),
        "When a skill references a relative path, resolve it against the skill directory."
            .to_owned(),
        String::new(),
        "<available_skills>".to_owned(),
    ];
    for skill in visible {
        lines.push("  <skill>".to_owned());
        lines.push(format!("    <name>{}</name>", escape_xml(&skill.name)));
        lines.push(format!(
            "    <description>{}</description>",
            escape_xml(&skill.description)
        ));
        lines.push(format!(
            "    <location>{}</location>",
            escape_xml(&skill.file_path.to_string_lossy())
        ));
        lines.push("  </skill>".to_owned());
    }
    lines.push("</available_skills>".to_owned());
    lines.join("\n")
}

/// Expand `/skill:name optional instructions`. Unknown skills return `Ok(None)`.
///
/// # Errors
///
/// Returns an I/O error when the selected skill file can no longer be read.
pub fn expand_skill_command(
    input: &str,
    skills: &[Skill],
) -> Result<Option<String>, std::io::Error> {
    let Some(remainder) = input.strip_prefix("/skill:") else {
        return Ok(None);
    };
    let (name, arguments) = remainder
        .split_once(char::is_whitespace)
        .map_or((remainder, ""), |(name, arguments)| {
            (name, arguments.trim())
        });
    let Some(skill) = skills.iter().find(|skill| skill.name == name) else {
        return Ok(None);
    };
    let content = fs::read_to_string(&skill.file_path)?;
    let body = match parse_frontmatter::<SkillFrontmatter>(&content) {
        Ok((_, body)) => body,
        Err(_) => content,
    };
    let mut expanded = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.file_path.display(),
        skill.base_dir.display(),
        body.trim()
    );
    if !arguments.is_empty() {
        expanded.push_str("\n\n");
        expanded.push_str(arguments);
    }
    Ok(Some(expanded))
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn has_markdown_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
}

fn build_ignore_matcher(
    root: &Path,
    files: &[PathBuf],
    source: &SourceInfo,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<Gitignore> {
    let mut builder = GitignoreBuilder::new(root);
    for file in files {
        if let Some(error) = builder.add(file) {
            diagnostics.push(Diagnostic::warning(
                format!("failed to load ignore file: {error}"),
                source.with_path(file),
            ));
        }
    }
    match builder.build() {
        Ok(matcher) => Some(matcher),
        Err(error) => {
            diagnostics.push(Diagnostic::warning(
                format!("failed to compile ignore rules: {error}"),
                source.with_path(root),
            ));
            None
        }
    }
}

fn is_ignored(path: &Path, directory: bool, matcher: Option<&Gitignore>) -> bool {
    matcher.is_some_and(|matcher| {
        matcher
            .matched_path_or_any_parents(path, directory)
            .is_ignore()
    })
}

// ---------------------------------------------------------------------------
// Prompt templates and argument expansion
// ---------------------------------------------------------------------------

/// Markdown prompt template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromptTemplate {
    /// Slash-command name derived from the filename.
    pub name: String,
    /// Human-readable command description.
    pub description: String,
    /// Optional argument hint for command completion.
    pub argument_hint: Option<String>,
    /// Prompt body after frontmatter removal.
    pub content: String,
    /// Source markdown file.
    pub file_path: PathBuf,
    /// Discovery provenance.
    pub source: SourceInfo,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct PromptFrontmatter {
    description: Option<String>,
    #[serde(rename = "argument-hint")]
    argument_hint: Option<String>,
}

/// Prompt discovery output.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PromptLoadResult {
    /// Valid, first-wins templates in deterministic order.
    pub prompts: Vec<PromptTemplate>,
    /// Read, parse, and collision diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

/// Load direct `.md` children from directories and first-wins deduplicate by
/// file stem. Prompt directories are intentionally non-recursive.
pub fn load_prompt_templates(paths: &[ResourcePath]) -> PromptLoadResult {
    let mut ordered = paths.to_vec();
    ordered.sort_by_key(|entry| entry.source.precedence_rank());
    let mut output = PromptLoadResult::default();
    let mut prompts = IndexMap::<String, PromptTemplate>::new();
    let mut seen_files = BTreeSet::<PathBuf>::new();

    for entry in ordered {
        if !entry.path.exists() {
            if entry.source.source.is_auto() {
                continue;
            }
            output.diagnostics.push(Diagnostic::warning(
                "prompt template path does not exist",
                entry.source.with_path(&entry.path),
            ));
            continue;
        }
        let files = if entry.path.is_dir() {
            match direct_markdown_files(&entry.path) {
                Ok(files) => files,
                Err(error) => {
                    output.diagnostics.push(Diagnostic::warning(
                        format!("failed to read prompt template directory: {error}"),
                        entry.source.with_path(&entry.path),
                    ));
                    Vec::new()
                }
            }
        } else if has_markdown_extension(&entry.path) {
            vec![entry.path.clone()]
        } else {
            output.diagnostics.push(Diagnostic::warning(
                "prompt template path is not a markdown file",
                entry.source.with_path(&entry.path),
            ));
            Vec::new()
        };
        for file in files {
            let canonical = canonical_key(&file);
            if !seen_files.insert(canonical) {
                continue;
            }
            match load_prompt_file(&file, &entry.source) {
                Ok(prompt) => {
                    if let Some(winner) = prompts.get(&prompt.name) {
                        output.diagnostics.push(Diagnostic::collision(
                            ResourceKind::Prompt,
                            prompt.name.clone(),
                            winner.source.clone(),
                            prompt.source.clone(),
                        ));
                    } else {
                        prompts.insert(prompt.name.clone(), prompt);
                    }
                }
                Err(message) => output
                    .diagnostics
                    .push(Diagnostic::warning(message, entry.source.with_path(&file))),
            }
        }
    }
    output.prompts = prompts.into_values().collect();
    output
}

fn direct_markdown_files(directory: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(directory)?
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && has_markdown_extension(path))
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

fn load_prompt_file(path: &Path, source: &SourceInfo) -> Result<PromptTemplate, String> {
    let raw = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let (frontmatter, body) =
        parse_frontmatter::<PromptFrontmatter>(&raw).map_err(|error| error.to_string())?;
    let name = path
        .file_stem()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "prompt filename is not valid UTF-8".to_owned())?
        .to_owned();
    let description = frontmatter.description.unwrap_or_else(|| {
        let first = body
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or_default();
        let mut chars = first.chars();
        let prefix = chars.by_ref().take(60).collect::<String>();
        if chars.next().is_some() {
            format!("{prefix}...")
        } else {
            prefix
        }
    });
    Ok(PromptTemplate {
        name,
        description,
        argument_hint: frontmatter.argument_hint.filter(|hint| !hint.is_empty()),
        content: body,
        file_path: path.to_path_buf(),
        source: source.with_path(path),
    })
}

/// Parse command arguments with Pi's intentionally small quote grammar.
pub fn parse_command_args(input: &str) -> Vec<String> {
    let mut arguments = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    for character in input.chars() {
        if let Some(expected) = quote {
            if character == expected {
                quote = None;
            } else {
                current.push(character);
            }
        } else if character == '"' || character == '\'' {
            quote = Some(character);
        } else if character.is_whitespace() {
            if !current.is_empty() {
                arguments.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }
    if !current.is_empty() {
        arguments.push(current);
    }
    arguments
}

/// Substitute positional, aggregate, default, and slice placeholders in one
/// pass. Inserted argument/default values are never rescanned.
pub fn substitute_args(template: &str, arguments: &[String]) -> String {
    let all = arguments.join(" ");
    let mut output = String::with_capacity(template.len());
    let mut index = 0;
    while index < template.len() {
        let Some(character) = template[index..].chars().next() else {
            break;
        };
        if character != '$' {
            output.push(character);
            index += character.len_utf8();
            continue;
        }

        let braced_end = template[index..]
            .strip_prefix("${")
            .and_then(|rest| rest.find('}'))
            .map(|offset| index + 2 + offset);
        if let Some(end) = braced_end {
            let expression = &template[index + 2..end];
            if let Some(replacement) = expand_braced_placeholder(expression, arguments, &all) {
                output.push_str(&replacement);
                index = end + 1;
                continue;
            }
        }
        if template[index..].starts_with("$ARGUMENTS") {
            output.push_str(&all);
            index += "$ARGUMENTS".len();
            continue;
        }
        if template[index..].starts_with("$@") {
            output.push_str(&all);
            index += 2;
            continue;
        }
        let digits = template[index + 1..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>();
        if !digits.is_empty() {
            let position = digits.parse::<usize>().unwrap_or_default();
            if position > 0 {
                output.push_str(
                    arguments
                        .get(position - 1)
                        .map(String::as_str)
                        .unwrap_or_default(),
                );
            }
            index += 1 + digits.len();
            continue;
        }
        output.push('$');
        index += 1;
    }
    output
}

fn expand_braced_placeholder(expression: &str, arguments: &[String], all: &str) -> Option<String> {
    if let Some((target, default)) = expression.split_once(":-") {
        let value = match target {
            "@" | "ARGUMENTS" => all,
            _ if target.chars().all(|character| character.is_ascii_digit()) => {
                let position = target.parse::<usize>().ok()?;
                if position == 0 {
                    ""
                } else {
                    arguments
                        .get(position - 1)
                        .map(String::as_str)
                        .unwrap_or_default()
                }
            }
            _ => return None,
        };
        return Some(if value.is_empty() { default } else { value }.to_owned());
    }
    let slice = expression.strip_prefix("@:")?;
    let mut parts = slice.split(':');
    let position = parts.next()?.parse::<usize>().ok()?;
    let start = position.saturating_sub(1);
    let length = parts.next().map(str::parse::<usize>).transpose().ok()?;
    if parts.next().is_some() {
        return None;
    }
    let end = length.map_or(arguments.len(), |length| {
        start.saturating_add(length).min(arguments.len())
    });
    if start >= arguments.len() {
        Some(String::new())
    } else {
        Some(arguments[start..end].join(" "))
    }
}

/// Expand `/template arguments`, returning `None` for a non-template.
pub fn expand_prompt_template(input: &str, templates: &[PromptTemplate]) -> Option<String> {
    let remainder = input.strip_prefix('/')?;
    let boundary = remainder
        .char_indices()
        .find_map(|(index, character)| character.is_whitespace().then_some(index));
    let (name, raw_arguments) = boundary.map_or((remainder, ""), |index| {
        (&remainder[..index], remainder[index..].trim_start())
    });
    let template = templates.iter().find(|template| template.name == name)?;
    let arguments = parse_command_args(raw_arguments);
    Some(substitute_args(&template.content, &arguments))
}

// ---------------------------------------------------------------------------
// Context files, SYSTEM, APPEND_SYSTEM, and aggregate loader
// ---------------------------------------------------------------------------

/// Loaded AGENTS/CLAUDE context file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextResource {
    /// Context file path.
    pub path: PathBuf,
    /// Complete context text.
    pub content: String,
    /// Discovery provenance.
    pub source: SourceInfo,
}

/// Explicit prompt input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromptInput {
    /// Prompt text supplied directly by the host.
    Literal(String),
    /// Prompt text loaded from a file.
    File(PathBuf),
}

impl PromptInput {
    fn resolve(&self) -> Result<String, std::io::Error> {
        match self {
            Self::Literal(value) => Ok(value.clone()),
            Self::File(path) => fs::read_to_string(path),
        }
    }
}

/// Resource loader configuration.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ResourceLoaderOptions {
    /// Current project working directory.
    pub cwd: PathBuf,
    /// User configuration root (for example `~/.ri`).
    pub agent_dir: PathBuf,
    /// User home directory used to classify `.agents/skills` provenance.
    pub home_dir: Option<PathBuf>,
    /// Whether trust-gated project resources may be loaded.
    pub project_trusted: bool,
    /// Explicit skill paths from settings.
    pub configured_skill_paths: Vec<ResourcePath>,
    /// Explicit prompt paths from settings.
    pub configured_prompt_paths: Vec<ResourcePath>,
    /// Skill paths exported by resolved packages.
    pub package_skill_paths: Vec<ResourcePath>,
    /// Prompt paths exported by resolved packages.
    pub package_prompt_paths: Vec<ResourcePath>,
    /// Additional host- or extension-contributed skill paths.
    pub additional_skill_paths: Vec<ResourcePath>,
    /// Additional host- or extension-contributed prompt paths.
    pub additional_prompt_paths: Vec<ResourcePath>,
    /// Explicit system-prompt replacement, which has highest precedence.
    pub explicit_system_prompt: Option<PromptInput>,
    /// Explicit system-prompt suffixes, which replace discovered suffixes.
    pub explicit_append_system: Vec<PromptInput>,
    /// Whether conventional skill directories are scanned.
    pub discover_skills: bool,
    /// Whether conventional prompt directories are scanned.
    pub discover_prompts: bool,
    /// Whether hierarchical AGENTS/CLAUDE context files are scanned.
    pub discover_context: bool,
}

impl ResourceLoaderOptions {
    /// Create fail-closed loader options for a project and user config root.
    pub fn new(cwd: impl Into<PathBuf>, agent_dir: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            agent_dir: agent_dir.into(),
            home_dir: None,
            project_trusted: false,
            configured_skill_paths: Vec::new(),
            configured_prompt_paths: Vec::new(),
            package_skill_paths: Vec::new(),
            package_prompt_paths: Vec::new(),
            additional_skill_paths: Vec::new(),
            additional_prompt_paths: Vec::new(),
            explicit_system_prompt: None,
            explicit_append_system: Vec::new(),
            discover_skills: true,
            discover_prompts: true,
            discover_context: true,
        }
    }
}

/// Immutable result of one deterministic resource load.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceSnapshot {
    /// Runtime generation that produced this snapshot.
    pub generation: u64,
    /// Discovered skills.
    pub skills: Vec<Skill>,
    /// Discovered prompt templates.
    pub prompts: Vec<PromptTemplate>,
    /// Ordered global-to-local context chain.
    pub context: Vec<ContextResource>,
    /// Highest-precedence system-prompt replacement.
    pub system_prompt: Option<String>,
    /// Ordered system-prompt suffixes.
    pub append_system: Vec<String>,
    /// Non-fatal discovery and validation diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

/// Reloadable resource loader sharing the runtime generation clock.
#[derive(Debug)]
pub struct ResourceLoader {
    options: ResourceLoaderOptions,
    clock: GenerationClock,
    snapshot: ResourceSnapshot,
}

impl ResourceLoader {
    /// Create a reloadable loader at the clock's current generation.
    pub fn new(options: ResourceLoaderOptions, clock: GenerationClock) -> Self {
        let generation = clock.current();
        Self {
            options,
            clock,
            snapshot: ResourceSnapshot {
                generation,
                ..ResourceSnapshot::default()
            },
        }
    }

    /// Mutably access options used by the next reload.
    pub fn options_mut(&mut self) -> &mut ResourceLoaderOptions {
        &mut self.options
    }

    /// Return the active immutable resource snapshot.
    pub fn snapshot(&self) -> &ResourceSnapshot {
        &self.snapshot
    }

    /// Invalidate old contexts and replace the complete resource snapshot.
    pub fn reload(&mut self) {
        let generation = self.clock.advance();
        self.snapshot = load_resource_snapshot(&self.options, generation);
    }
}

/// Load a standalone snapshot without changing a generation clock.
pub fn load_resource_snapshot(
    options: &ResourceLoaderOptions,
    generation: u64,
) -> ResourceSnapshot {
    let mut diagnostics = Vec::new();
    let skill_paths = collect_skill_paths(options);
    let prompt_paths = collect_prompt_paths(options);
    let skill_result = load_skills(&skill_paths);
    let prompt_result = load_prompt_templates(&prompt_paths);
    diagnostics.extend(skill_result.diagnostics);
    diagnostics.extend(prompt_result.diagnostics);

    let context = if options.discover_context {
        load_context_chain(options, &mut diagnostics)
    } else {
        Vec::new()
    };
    let system_prompt = resolve_system_prompt(options, &mut diagnostics);
    let append_system = resolve_append_system(options, &mut diagnostics);

    ResourceSnapshot {
        generation,
        skills: skill_result.skills,
        prompts: prompt_result.prompts,
        context,
        system_prompt,
        append_system,
        diagnostics,
    }
}

fn collect_skill_paths(options: &ResourceLoaderOptions) -> Vec<ResourcePath> {
    let mut paths = Vec::new();
    paths.extend(options.configured_skill_paths.clone());
    if options.discover_skills {
        if options.project_trusted {
            let project_root = options.cwd.join(".ri");
            paths.push(ResourcePath {
                path: project_root.join("skills"),
                source: SourceInfo::auto(
                    project_root.join("skills"),
                    SourceScope::Project,
                    &project_root,
                ),
            });
            for agents_dir in ancestor_agents_skill_dirs(&options.cwd, options.home_dir.as_deref())
            {
                paths.push(ResourcePath {
                    path: agents_dir.clone(),
                    source: SourceInfo::auto(
                        &agents_dir,
                        SourceScope::Project,
                        agents_dir.parent().unwrap_or(&agents_dir),
                    ),
                });
            }
        }
        paths.push(ResourcePath {
            path: options.agent_dir.join("skills"),
            source: SourceInfo::auto(
                options.agent_dir.join("skills"),
                SourceScope::User,
                &options.agent_dir,
            ),
        });
        if let Some(home) = &options.home_dir {
            let agents = home.join(".agents").join("skills");
            paths.push(ResourcePath {
                path: agents.clone(),
                source: SourceInfo::auto(&agents, SourceScope::User, home.join(".agents")),
            });
        }
    }
    paths.extend(options.additional_skill_paths.clone());
    paths.extend(options.package_skill_paths.clone());
    dedupe_resource_paths(paths, options.project_trusted)
}

fn collect_prompt_paths(options: &ResourceLoaderOptions) -> Vec<ResourcePath> {
    let mut paths = Vec::new();
    paths.extend(options.configured_prompt_paths.clone());
    if options.discover_prompts {
        if options.project_trusted {
            let project_root = options.cwd.join(".ri");
            paths.push(ResourcePath {
                path: project_root.join("prompts"),
                source: SourceInfo::auto(
                    project_root.join("prompts"),
                    SourceScope::Project,
                    &project_root,
                ),
            });
        }
        paths.push(ResourcePath {
            path: options.agent_dir.join("prompts"),
            source: SourceInfo::auto(
                options.agent_dir.join("prompts"),
                SourceScope::User,
                &options.agent_dir,
            ),
        });
    }
    paths.extend(options.additional_prompt_paths.clone());
    paths.extend(options.package_prompt_paths.clone());
    dedupe_resource_paths(paths, options.project_trusted)
}

fn dedupe_resource_paths(paths: Vec<ResourcePath>, project_trusted: bool) -> Vec<ResourcePath> {
    let mut paths = paths;
    paths.sort_by_key(|entry| entry.source.precedence_rank());
    let mut seen = BTreeSet::new();
    paths
        .into_iter()
        .filter(|entry| project_trusted || entry.source.scope != SourceScope::Project)
        .filter(|entry| seen.insert(canonical_key(&entry.path)))
        .collect()
}

fn ancestor_agents_skill_dirs(cwd: &Path, home_dir: Option<&Path>) -> Vec<PathBuf> {
    let mut directories = Vec::new();
    let mut current = cwd.to_path_buf();
    let home = home_dir.map(canonical_key);
    loop {
        let is_home = home.as_ref() == Some(&canonical_key(&current));
        if !is_home {
            directories.push(current.join(".agents").join("skills"));
        }
        if is_home {
            break;
        }
        if !current.pop() {
            break;
        }
    }
    directories
}

fn load_context_chain(
    options: &ResourceLoaderOptions,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<ContextResource> {
    let mut context = Vec::new();
    let mut seen = BTreeSet::new();
    if let Some(resource) = load_context_from_dir(
        &options.agent_dir,
        SourceScope::User,
        &SourceKind::Configured,
        diagnostics,
    ) {
        seen.insert(canonical_key(&resource.path));
        context.push(resource);
    }
    if !options.project_trusted {
        return context;
    }

    let mut ancestors = Vec::new();
    let mut current = options.cwd.clone();
    loop {
        ancestors.push(current.clone());
        if !current.pop() {
            break;
        }
    }
    ancestors.reverse();
    for directory in ancestors {
        if let Some(resource) = load_context_from_dir(
            &directory,
            SourceScope::Project,
            &SourceKind::Auto,
            diagnostics,
        )
        .filter(|resource| seen.insert(canonical_key(&resource.path)))
        {
            context.push(resource);
        }
    }
    context
}

fn load_context_from_dir(
    directory: &Path,
    scope: SourceScope,
    source_kind: &SourceKind,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<ContextResource> {
    for filename in CONTEXT_CANDIDATES {
        let path = directory.join(filename);
        if !path.exists() {
            continue;
        }
        let source = SourceInfo {
            path: path.clone(),
            source: source_kind.clone(),
            scope,
            origin: SourceOrigin::TopLevel,
            base_dir: Some(directory.to_path_buf()),
        };
        match fs::read_to_string(&path) {
            Ok(content) => {
                return Some(ContextResource {
                    path,
                    content,
                    source,
                });
            }
            Err(error) => diagnostics.push(Diagnostic::warning(
                format!("failed to read context file: {error}"),
                source,
            )),
        }
    }
    None
}

fn resolve_system_prompt(
    options: &ResourceLoaderOptions,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<String> {
    if let Some(input) = &options.explicit_system_prompt {
        return resolve_prompt_input(input, ResourceKind::SystemPrompt, diagnostics);
    }
    if options.project_trusted {
        let project = options.cwd.join(".ri").join("SYSTEM.md");
        if project.exists() {
            return resolve_prompt_input(
                &PromptInput::File(project),
                ResourceKind::SystemPrompt,
                diagnostics,
            );
        }
    }
    let global = options.agent_dir.join("SYSTEM.md");
    global.exists().then(|| {
        resolve_prompt_input(
            &PromptInput::File(global),
            ResourceKind::SystemPrompt,
            diagnostics,
        )
    })?
}

fn resolve_append_system(
    options: &ResourceLoaderOptions,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<String> {
    if !options.explicit_append_system.is_empty() {
        return options
            .explicit_append_system
            .iter()
            .filter_map(|input| {
                resolve_prompt_input(input, ResourceKind::SystemPrompt, diagnostics)
            })
            .collect();
    }
    if options.project_trusted {
        let project = options.cwd.join(".ri").join("APPEND_SYSTEM.md");
        if project.exists() {
            return resolve_prompt_input(
                &PromptInput::File(project),
                ResourceKind::SystemPrompt,
                diagnostics,
            )
            .into_iter()
            .collect();
        }
    }
    let global = options.agent_dir.join("APPEND_SYSTEM.md");
    if global.exists() {
        resolve_prompt_input(
            &PromptInput::File(global),
            ResourceKind::SystemPrompt,
            diagnostics,
        )
        .into_iter()
        .collect()
    } else {
        Vec::new()
    }
}

fn resolve_prompt_input(
    input: &PromptInput,
    kind: ResourceKind,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<String> {
    match input.resolve() {
        Ok(value) => Some(value),
        Err(error) => {
            let path = match input {
                PromptInput::Literal(_) => PathBuf::from("<literal>"),
                PromptInput::File(path) => path.clone(),
            };
            diagnostics.push(Diagnostic::error(
                format!("failed to read {kind:?}: {error}"),
                SourceInfo::configured(path, SourceScope::Temporary),
            ));
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn write(path: &Path, value: &str) {
        fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        fs::write(path, value).expect("write");
    }

    #[test]
    fn skill_validation_warns_but_only_missing_description_skips() {
        let temp = tempdir().expect("tempdir");
        let invalid = temp.path().join("Bad Name").join("SKILL.md");
        write(
            &invalid,
            "---\nname: Bad Name\ndescription: works\n---\nbody",
        );
        let missing = temp.path().join("missing").join("SKILL.md");
        write(&missing, "---\nname: missing\n---\nbody");
        let result = load_skills(&[ResourcePath::configured(
            temp.path(),
            SourceScope::Temporary,
        )]);
        assert_eq!(result.skills.len(), 1);
        assert_eq!(result.skills[0].name, "Bad Name");
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("invalid characters"))
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("description is required"))
        );
    }

    #[test]
    fn root_skill_stops_nested_discovery_and_ignore_files_apply() {
        let temp = tempdir().expect("tempdir");
        write(
            &temp.path().join("SKILL.md"),
            "---\nname: root\ndescription: root\n---\nroot",
        );
        write(
            &temp.path().join("nested").join("SKILL.md"),
            "---\nname: nested\ndescription: nested\n---\nnested",
        );
        let result = load_skills(&[ResourcePath::configured(
            temp.path(),
            SourceScope::Temporary,
        )]);
        assert_eq!(
            result
                .skills
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            vec!["root"]
        );

        fs::remove_file(temp.path().join("SKILL.md")).expect("remove");
        write(&temp.path().join(".gitignore"), "nested/\n");
        let result = load_skills(&[ResourcePath::configured(
            temp.path(),
            SourceScope::Temporary,
        )]);
        assert!(result.skills.is_empty());
    }

    #[test]
    fn first_skill_name_wins_by_source_precedence() {
        let temp = tempdir().expect("tempdir");
        let project = temp.path().join("project").join("SKILL.md");
        let user = temp.path().join("user").join("SKILL.md");
        write(
            &project,
            "---\nname: same\ndescription: project\n---\nproject",
        );
        write(&user, "---\nname: same\ndescription: user\n---\nuser");
        let result = load_skills(&[
            ResourcePath::configured(&user, SourceScope::User),
            ResourcePath::configured(&project, SourceScope::Project),
        ]);
        assert_eq!(result.skills[0].description, "project");
        assert_eq!(result.diagnostics.len(), 1);
    }

    #[test]
    fn home_agent_skills_keep_user_provenance() {
        let temp = tempdir().expect("tempdir");
        let home = temp.path().join("home");
        let cwd = home.join("project");
        fs::create_dir_all(&cwd).expect("cwd");
        write(
            &home
                .join(".agents")
                .join("skills")
                .join("home")
                .join("SKILL.md"),
            "---\nname: home\ndescription: user skill\n---\nbody",
        );
        let mut options = ResourceLoaderOptions::new(&cwd, temp.path().join("agent"));
        options.home_dir = Some(home);
        options.discover_prompts = false;
        options.discover_context = false;
        let snapshot = load_resource_snapshot(&options, 0);
        let skill = snapshot
            .skills
            .iter()
            .find(|skill| skill.name == "home")
            .expect("home skill");
        assert_eq!(skill.source.scope, SourceScope::User);
        assert_eq!(skill.source.source, SourceKind::Auto);
    }

    #[test]
    fn configured_project_resources_are_trust_gated() {
        let temp = tempdir().expect("tempdir");
        let project = temp.path().join("project").join("SKILL.md");
        let user = temp.path().join("user").join("SKILL.md");
        write(
            &project,
            "---\nname: project\ndescription: project skill\n---\nbody",
        );
        write(&user, "---\nname: user\ndescription: user skill\n---\nbody");
        let mut options = ResourceLoaderOptions::new(temp.path(), temp.path().join("agent"));
        options.discover_skills = false;
        options.discover_prompts = false;
        options.discover_context = false;
        options.configured_skill_paths = vec![
            ResourcePath::configured(&project, SourceScope::Project),
            ResourcePath::configured(&user, SourceScope::User),
        ];
        let untrusted = load_resource_snapshot(&options, 0);
        assert_eq!(
            untrusted
                .skills
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            vec!["user"]
        );

        options.project_trusted = true;
        let trusted = load_resource_snapshot(&options, 1);
        assert_eq!(trusted.skills.len(), 2);
    }

    #[test]
    fn skill_prompt_uses_progressive_disclosure_and_expands_explicitly() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("skill").join("SKILL.md");
        write(
            &path,
            "---\nname: test\ndescription: A <test> & skill\n---\nFull body",
        );
        let result = load_skills(&[ResourcePath::configured(&path, SourceScope::Temporary)]);
        let prompt = format_skills_for_prompt(&result.skills);
        assert!(prompt.contains("A &lt;test&gt; &amp; skill"));
        assert!(!prompt.contains("Full body"));
        let expanded = expand_skill_command("/skill:test extra", &result.skills)
            .expect("read")
            .expect("known");
        assert!(expanded.contains("Full body"));
        assert!(expanded.ends_with("\n\nextra"));
    }

    #[test]
    fn prompt_templates_are_non_recursive_and_first_wins() {
        let temp = tempdir().expect("tempdir");
        let project = temp.path().join("project");
        let user = temp.path().join("user");
        write(&project.join("commit.md"), "Project");
        write(&user.join("commit.md"), "User");
        write(&project.join("nested").join("ignored.md"), "Ignored");
        let result = load_prompt_templates(&[
            ResourcePath::configured(&user, SourceScope::User),
            ResourcePath::configured(&project, SourceScope::Project),
        ]);
        assert_eq!(result.prompts.len(), 1);
        assert_eq!(result.prompts[0].content, "Project");
        assert_eq!(result.diagnostics.len(), 1);
    }

    #[test]
    fn argument_parser_and_substitution_match_reference_semantics() {
        let arguments = parse_command_args("cmd \"first arg\" second");
        assert_eq!(arguments, vec!["cmd", "first arg", "second"]);
        assert_eq!(
            substitute_args("$1 | $@ | ${@:2} | ${@:2:1} | ${4:-fallback}", &arguments),
            "cmd | cmd first arg second | first arg second | first arg | fallback"
        );
        assert_eq!(
            substitute_args("$ARGUMENTS", &["$1".to_owned(), "$ARGUMENTS".to_owned()]),
            "$1 $ARGUMENTS"
        );
        assert_eq!(substitute_args("$0/$99", &arguments), "/");
    }

    #[test]
    fn prompt_expansion_supports_newline_arguments() {
        let template = PromptTemplate {
            name: "test".to_owned(),
            description: String::new(),
            argument_hint: None,
            content: "$1 -- ${@:2}".to_owned(),
            file_path: PathBuf::from("test.md"),
            source: SourceInfo::inline("test"),
        };
        assert_eq!(
            expand_prompt_template("/test label\nlong description", &[template]).as_deref(),
            Some("label -- long description")
        );
    }

    #[test]
    fn context_chain_is_global_then_root_to_cwd_and_trust_gated() {
        let temp = tempdir().expect("tempdir");
        let agent = temp.path().join("agent");
        let root = temp.path().join("project");
        let cwd = root.join("child");
        write(&agent.join("AGENTS.md"), "global");
        write(&root.join("AGENTS.md"), "root");
        write(&cwd.join("CLAUDE.md"), "child");
        let mut options = ResourceLoaderOptions::new(&cwd, &agent);
        options.project_trusted = true;
        options.discover_skills = false;
        options.discover_prompts = false;
        let trusted = load_resource_snapshot(&options, 1);
        assert_eq!(
            trusted
                .context
                .iter()
                .map(|resource| resource.content.as_str())
                .collect::<Vec<_>>(),
            vec!["global", "root", "child"]
        );
        options.project_trusted = false;
        let untrusted = load_resource_snapshot(&options, 2);
        assert_eq!(
            untrusted
                .context
                .iter()
                .map(|resource| resource.content.as_str())
                .collect::<Vec<_>>(),
            vec!["global"]
        );
    }

    #[test]
    fn system_and_append_precedence_is_deterministic() {
        let temp = tempdir().expect("tempdir");
        let agent = temp.path().join("agent");
        let cwd = temp.path().join("project");
        write(&agent.join("SYSTEM.md"), "global system");
        write(&agent.join("APPEND_SYSTEM.md"), "global append");
        write(&cwd.join(".ri").join("SYSTEM.md"), "project system");
        write(&cwd.join(".ri").join("APPEND_SYSTEM.md"), "project append");
        let mut options = ResourceLoaderOptions::new(&cwd, &agent);
        options.project_trusted = true;
        options.discover_context = false;
        options.discover_skills = false;
        options.discover_prompts = false;
        let project = load_resource_snapshot(&options, 1);
        assert_eq!(project.system_prompt.as_deref(), Some("project system"));
        assert_eq!(project.append_system, vec!["project append"]);

        options.explicit_system_prompt = Some(PromptInput::Literal("explicit".to_owned()));
        options.explicit_append_system = vec![
            PromptInput::Literal("one".to_owned()),
            PromptInput::Literal("two".to_owned()),
        ];
        let explicit = load_resource_snapshot(&options, 2);
        assert_eq!(explicit.system_prompt.as_deref(), Some("explicit"));
        assert_eq!(explicit.append_system, vec!["one", "two"]);
    }

    #[test]
    fn resource_reload_advances_shared_generation() {
        let temp = tempdir().expect("tempdir");
        let clock = GenerationClock::default();
        let mut options = ResourceLoaderOptions::new(temp.path(), temp.path());
        options.discover_context = false;
        options.discover_skills = false;
        options.discover_prompts = false;
        let mut loader = ResourceLoader::new(options, clock.clone());
        let before = clock.current();
        loader.reload();
        assert_eq!(loader.snapshot().generation, before + 1);
        assert_eq!(clock.current(), before + 1);
    }
}
