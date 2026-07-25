//! Immutable resources shared by session construction and frontends.

use std::fs;
use std::sync::Arc;

use ri_agent::Tool;
use ri_harness::{PromptTemplate, Resources, Skill};

/// Loaded prompt resources and executable tools.
///
/// Discovery and trust remain owned by `ri-ext`; this type is the immutable
/// resolved snapshot installed into a session.
#[derive(Clone, Default)]
pub struct ResourceRuntime {
    resources: Arc<Resources>,
    tools: Arc<[Arc<dyn Tool>]>,
    active_tool_names: Arc<[String]>,
    generation: u64,
    system_prompt: Option<Arc<str>>,
    append_system: Arc<[String]>,
    diagnostics: Arc<[ri_ext::Diagnostic]>,
}

impl std::fmt::Debug for ResourceRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResourceRuntime")
            .field("resources", &self.resources)
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.definition().name.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("active_tool_names", &self.active_tool_names)
            .field("generation", &self.generation)
            .field("system_prompt", &self.system_prompt)
            .field("append_system", &self.append_system)
            .field("diagnostics", &self.diagnostics)
            .finish()
    }
}

impl ResourceRuntime {
    /// Creates a resolved snapshot. All tools are active initially.
    pub fn new(resources: Resources, tools: Vec<Arc<dyn Tool>>) -> Self {
        let active_tool_names = tools
            .iter()
            .map(|tool| tool.definition().name.clone())
            .collect::<Vec<_>>();
        Self {
            resources: Arc::new(resources),
            tools: tools.into(),
            active_tool_names: active_tool_names.into(),
            generation: 0,
            system_prompt: None,
            append_system: Arc::new([]),
            diagnostics: Arc::new([]),
        }
    }

    /// Creates a resolved snapshot with an explicit active subset.
    pub fn with_active_tools(
        resources: Resources,
        tools: Vec<Arc<dyn Tool>>,
        active_tool_names: Vec<String>,
    ) -> Self {
        Self {
            resources: Arc::new(resources),
            tools: tools.into(),
            active_tool_names: active_tool_names.into(),
            generation: 0,
            system_prompt: None,
            append_system: Arc::new([]),
            diagnostics: Arc::new([]),
        }
    }

    /// Converts one deterministic `ri-ext` resource snapshot.
    ///
    /// Skill bodies are read eagerly so a harness turn captures immutable
    /// content rather than observing a file that changes mid-turn.
    ///
    /// # Errors
    /// Returns an error when a discovered skill body cannot be read.
    pub fn from_snapshot(
        snapshot: ri_ext::ResourceSnapshot,
        tools: Vec<Arc<dyn Tool>>,
    ) -> std::io::Result<Self> {
        let skills = snapshot
            .skills
            .iter()
            .map(|skill| {
                let content = fs::read_to_string(&skill.file_path)?;
                Ok(Skill {
                    name: skill.name.clone(),
                    description: skill.description.clone(),
                    content: strip_frontmatter(&content).trim().to_owned(),
                    source: skill.file_path.display().to_string(),
                    disable_model_invocation: skill.disable_model_invocation,
                })
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        let prompt_templates = snapshot
            .prompts
            .iter()
            .map(|prompt| PromptTemplate {
                name: prompt.name.clone(),
                description: (!prompt.description.is_empty()).then(|| prompt.description.clone()),
                content: prompt.content.clone(),
                source: prompt.file_path.display().to_string(),
            })
            .collect::<Vec<_>>();
        let context = snapshot
            .context
            .iter()
            .map(|resource| resource.content.clone())
            .collect::<Vec<_>>();
        let active_tool_names = tools
            .iter()
            .map(|tool| tool.definition().name.clone())
            .collect::<Vec<_>>();
        Ok(Self {
            resources: Arc::new(Resources::new(skills, prompt_templates, context)),
            tools: tools.into(),
            active_tool_names: active_tool_names.into(),
            generation: snapshot.generation,
            system_prompt: snapshot.system_prompt.map(Into::into),
            append_system: snapshot.append_system.into(),
            diagnostics: snapshot.diagnostics.into(),
        })
    }

    /// Prompt skills and templates.
    pub fn resources(&self) -> &Arc<Resources> {
        &self.resources
    }

    /// Executable tools in deterministic registration order.
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }

    /// Tool names enabled for new sessions.
    pub fn active_tool_names(&self) -> &[String] {
        &self.active_tool_names
    }

    /// Resource-loader generation captured by this runtime.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Diagnostics produced by `ri-ext` discovery and trust filtering.
    pub fn diagnostics(&self) -> &[ri_ext::Diagnostic] {
        &self.diagnostics
    }

    /// Resolves the model-visible base prompt from an optional SDK override,
    /// resource-provided prompt fragments, and the discoverable skill catalog.
    pub fn resolve_system_prompt(&self, explicit: Option<&str>) -> String {
        let mut parts = Vec::new();
        if let Some(base) = explicit.or(self.system_prompt.as_deref())
            && !base.trim().is_empty()
        {
            parts.push(base.to_owned());
        }
        parts.extend(
            self.append_system
                .iter()
                .filter(|fragment| !fragment.trim().is_empty())
                .cloned(),
        );
        let skills = format_skill_catalog(&self.resources.skills);
        if !skills.is_empty() {
            parts.push(skills);
        }
        parts.join("\n\n")
    }
}

fn strip_frontmatter(content: &str) -> &str {
    let Some(rest) = content.strip_prefix("---") else {
        return content;
    };
    let Some(after_open) = rest
        .strip_prefix("\r\n")
        .or_else(|| rest.strip_prefix('\n'))
    else {
        return content;
    };
    let mut offset = 0;
    for line in after_open.split_inclusive('\n') {
        offset += line.len();
        if line.trim_end_matches(['\r', '\n']).trim() == "---" {
            return &after_open[offset..];
        }
    }
    content
}

fn format_skill_catalog(skills: &[Skill]) -> String {
    let visible = skills
        .iter()
        .filter(|skill| !skill.disable_model_invocation)
        .collect::<Vec<_>>();
    if visible.is_empty() {
        return String::new();
    }
    let mut output = String::from(
        "The following skills provide specialized instructions for matching tasks.\n\
         Invoke one explicitly with `/skill:name`.\n\n<available_skills>",
    );
    for skill in visible {
        output.push_str("\n  <skill>\n    <name>");
        output.push_str(&xml_escape(&skill.name));
        output.push_str("</name>\n    <description>");
        output.push_str(&xml_escape(&skill.description));
        output.push_str("</description>\n    <location>");
        output.push_str(&xml_escape(&skill.source));
        output.push_str("</location>\n  </skill>");
    }
    output.push_str("\n</available_skills>");
    output
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}
