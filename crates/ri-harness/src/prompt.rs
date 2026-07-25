//! Resource expansion for the canonical prompt pipeline.

use std::path::Path;

use ri_ext::{parse_command_args, substitute_args};

use crate::types::{PromptTemplate, Resources, Skill};

/// Resource selected by a slash invocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExpandedResource {
    /// A `/skill:name` invocation.
    Skill {
        /// Stable skill name.
        name: String,
        /// Skill source path or identifier.
        source: String,
    },
    /// A `/name` prompt-template invocation.
    PromptTemplate {
        /// Stable template name.
        name: String,
        /// Template source path or identifier.
        source: String,
    },
}

/// Result of applying Pi-compatible slash-resource expansion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceExpansion {
    /// Model-visible text after expansion, or the original input when unmatched.
    pub text: String,
    /// Resource responsible for the expansion.
    pub resource: Option<ExpandedResource>,
}

/// Expands `/skill:name` and prompt-template invocations.
///
/// Extension commands and input interception intentionally run before this
/// helper. Unknown slash commands are left untouched for the model.
///
pub fn expand_resources(input: &str, resources: &Resources) -> ResourceExpansion {
    if !input.starts_with('/') {
        return unchanged(input);
    }
    let (command, arguments) = split_invocation(input);
    if let Some(name) = command.strip_prefix("/skill:") {
        let Some(skill) = resources.skills.iter().find(|skill| skill.name == name) else {
            return unchanged(input);
        };
        return ResourceExpansion {
            text: format_skill(skill, arguments),
            resource: Some(ExpandedResource::Skill {
                name: skill.name.clone(),
                source: skill.source.clone(),
            }),
        };
    }
    let name = command.trim_start_matches('/');
    let Some(template) = resources
        .prompt_templates
        .iter()
        .find(|template| template.name == name)
    else {
        return unchanged(input);
    };
    ResourceExpansion {
        text: format_template(template, arguments),
        resource: Some(ExpandedResource::PromptTemplate {
            name: template.name.clone(),
            source: template.source.clone(),
        }),
    }
}

fn unchanged(input: &str) -> ResourceExpansion {
    ResourceExpansion {
        text: input.to_owned(),
        resource: None,
    }
}

/// Formats an explicit skill invocation.
pub fn format_skill(skill: &Skill, additional_instructions: &str) -> String {
    let base_dir = Path::new(&skill.source)
        .parent()
        .map_or_else(|| ".".to_owned(), |path| path.display().to_string());
    let mut prompt = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        xml_escape(&skill.name),
        xml_escape(&skill.source),
        xml_escape(&base_dir),
        skill.content
    );
    let instructions = additional_instructions.trim();
    if !instructions.is_empty() {
        prompt.push_str("\n\n");
        prompt.push_str(instructions);
    }
    prompt
}

/// Formats Pi positional, aggregate, default, and slice placeholders.
pub fn format_template(template: &PromptTemplate, arguments: &str) -> String {
    substitute_args(&template.content, &parse_command_args(arguments))
}

fn split_invocation(input: &str) -> (&str, &str) {
    input
        .find(char::is_whitespace)
        .map_or((input, ""), |index| {
            (&input[..index], input[index..].trim())
        })
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_quoted_template_arguments() {
        let template = PromptTemplate {
            name: "review".into(),
            description: None,
            content: "Review $1; all=$ARGUMENTS; tail=${@:2}; default=${4:-none}".into(),
            source: "test".into(),
        };
        assert_eq!(
            format_template(&template, "\"src/lib.rs\" 'the spec'"),
            "Review src/lib.rs; all=src/lib.rs the spec; tail=the spec; default=none"
        );
    }

    #[test]
    fn replacement_values_are_not_recursively_expanded() {
        let template = PromptTemplate {
            name: "literal".into(),
            description: None,
            content: "$1 | $ARGUMENTS".into(),
            source: "test".into(),
        };
        assert_eq!(format_template(&template, "'$2' '$@'"), "$2 | $2 $@");
    }

    #[test]
    fn unknown_commands_are_preserved() {
        assert_eq!(
            expand_resources("/unknown x", &Resources::default()),
            ResourceExpansion {
                text: "/unknown x".into(),
                resource: None,
            }
        );
    }

    #[test]
    fn skill_includes_source_and_extra_instructions() {
        let skill = Skill {
            name: "audit".into(),
            description: "audit".into(),
            content: "Inspect carefully.".into(),
            source: "C:/skills/audit/SKILL.md".into(),
            disable_model_invocation: false,
        };
        let output = format_skill(&skill, "focus on races");
        assert!(output.contains("Inspect carefully."));
        assert!(output.contains("focus on races"));
        assert!(output.contains("location=\"C:/skills/audit/SKILL.md\""));
        assert!(output.contains("References are relative to C:/skills/audit."));
        assert!(!output.contains("Additional instructions:"));
    }
}
