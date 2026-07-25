//! Adapters from the typed built-in tools into the agent tool contract.

use std::{path::PathBuf, sync::Arc};

use ri_agent::{
    FnTool, Tool, ToolError as AgentToolError, ToolExecutionMode, ToolResult as AgentToolResult,
};
use ri_ai::{ImageContent, InputContent, TextContent};
use serde::Serialize;
use tokio::sync::mpsc;

/// Builds the production local coding tools rooted at `cwd`.
///
/// # Errors
///
/// Returns an error when a generated input schema cannot be represented as
/// JSON.
pub fn local_tools(cwd: PathBuf) -> Result<Vec<Arc<dyn Tool>>, AgentToolError> {
    let tools = Arc::new(ri_tools::Tools::local(cwd));
    let mut output: Vec<Arc<dyn Tool>> = Vec::with_capacity(7);

    let runtime = Arc::clone(&tools);
    output.push(Arc::new(FnTool::typed::<ri_tools::ReadInput, _, _>(
        "read",
        "Read",
        "Read a UTF-8 text file or supported image. Text may be selected by line offset and limit.",
        move |context, input| {
            let runtime = Arc::clone(&runtime);
            async move {
                runtime
                    .read_with_cancellation(input, &context.cancellation)
                    .await
                    .map_err(|error| tool_error(&error))
                    .and_then(convert_result)
            }
        },
    )?));

    let runtime = Arc::clone(&tools);
    output.push(Arc::new(
        FnTool::typed::<ri_tools::BashInput, _, _>(
            "bash",
            "Bash",
            "Run a shell command in the session working directory and stream its combined output. There is no default timeout: foreground servers such as `npm run dev` keep running until cancelled, explicitly timed out, or exited.",
            move |context, input| {
                let runtime = Arc::clone(&runtime);
                async move {
                    let (update_tx, mut update_rx) = mpsc::unbounded_channel();
                    let on_update: ri_tools::BashUpdate = Arc::new(move |update| {
                        // A closed receiver means the enclosing tool future was
                        // dropped, so there is no remaining update consumer.
                        drop(update_tx.send(update));
                    });
                    let execution = runtime.bash_with_cancellation(
                        input,
                        &context.cancellation,
                        Some(on_update),
                    );
                    tokio::pin!(execution);

                    let result = loop {
                        tokio::select! {
                            result = &mut execution => break result,
                            Some(update) = update_rx.recv() => {
                                context
                                    .updates
                                    .send(convert_result(update)?)
                                    .await
                                    .map_err(|error| AgentToolError::message(error.to_string()))?;
                            }
                        }
                    };
                    while let Ok(update) = update_rx.try_recv() {
                        context
                            .updates
                            .send(convert_result(update)?)
                            .await
                            .map_err(|error| AgentToolError::message(error.to_string()))?;
                    }
                    result.map_err(|error| tool_error(&error)).and_then(convert_result)
                }
            },
        )?
        .with_execution_mode(ToolExecutionMode::Sequential),
    ));

    let runtime = Arc::clone(&tools);
    output.push(Arc::new(
        FnTool::typed::<ri_tools::EditInput, _, _>(
            "edit",
            "Edit",
            "Apply an exact, targeted text replacement to an existing file.",
            move |context, input| {
                let runtime = Arc::clone(&runtime);
                async move {
                    runtime
                        .edit_with_cancellation(input, &context.cancellation)
                        .await
                        .map_err(|error| tool_error(&error))
                        .and_then(convert_result)
                }
            },
        )?
        .with_execution_mode(ToolExecutionMode::Sequential),
    ));

    let runtime = Arc::clone(&tools);
    output.push(Arc::new(
        FnTool::typed::<ri_tools::WriteInput, _, _>(
            "write",
            "Write",
            "Create or replace a UTF-8 text file.",
            move |context, input| {
                let runtime = Arc::clone(&runtime);
                async move {
                    runtime
                        .write_with_cancellation(input, &context.cancellation)
                        .await
                        .map_err(|error| tool_error(&error))
                        .and_then(convert_result)
                }
            },
        )?
        .with_execution_mode(ToolExecutionMode::Sequential),
    ));

    let runtime = Arc::clone(&tools);
    output.push(Arc::new(FnTool::typed::<ri_tools::GrepInput, _, _>(
        "grep",
        "Grep",
        "Search file contents with a regular expression or literal pattern.",
        move |context, input| {
            let runtime = Arc::clone(&runtime);
            async move {
                runtime
                    .grep_with_cancellation(input, &context.cancellation)
                    .await
                    .map_err(|error| tool_error(&error))
                    .and_then(convert_result)
            }
        },
    )?));

    let runtime = Arc::clone(&tools);
    output.push(Arc::new(FnTool::typed::<ri_tools::FindInput, _, _>(
        "find",
        "Find",
        "Find files and directories by glob pattern.",
        move |context, input| {
            let runtime = Arc::clone(&runtime);
            async move {
                runtime
                    .find_with_cancellation(input, &context.cancellation)
                    .await
                    .map_err(|error| tool_error(&error))
                    .and_then(convert_result)
            }
        },
    )?));

    let runtime = tools;
    output.push(Arc::new(FnTool::typed::<ri_tools::LsInput, _, _>(
        "ls",
        "List",
        "List the immediate contents of a directory.",
        move |context, input| {
            let runtime = Arc::clone(&runtime);
            async move {
                runtime
                    .ls_with_cancellation(input, &context.cancellation)
                    .await
                    .map_err(|error| tool_error(&error))
                    .and_then(convert_result)
            }
        },
    )?));

    Ok(output)
}

fn tool_error(error: &ri_tools::ToolError) -> AgentToolError {
    AgentToolError::message(error.to_string())
}

fn convert_result<D>(result: ri_tools::ToolResult<D>) -> Result<AgentToolResult, AgentToolError>
where
    D: Serialize,
{
    let content = result
        .content
        .into_iter()
        .map(|block| match block {
            ri_tools::Content::Text { text } => InputContent::Text(TextContent::new(text)),
            ri_tools::Content::Image { data, mime_type } => {
                InputContent::Image(ImageContent { data, mime_type })
            }
        })
        .collect();
    let details = result
        .details
        .map(serde_json::to_value)
        .transpose()
        .map_err(|error| AgentToolError::message(error.to_string()))?;
    Ok(AgentToolResult {
        content,
        details,
        ..AgentToolResult::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_tools_have_reference_order_and_scheduling() {
        let tools = local_tools(PathBuf::from(".")).expect("construct built-in tools");
        let names = tools
            .iter()
            .map(|tool| tool.definition().name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            ["read", "bash", "edit", "write", "grep", "find", "ls"]
        );
        assert_eq!(tools[0].execution_mode(), None);
        assert_eq!(
            tools[1].execution_mode(),
            Some(ToolExecutionMode::Sequential)
        );
        assert_eq!(
            tools[2].execution_mode(),
            Some(ToolExecutionMode::Sequential)
        );
        assert_eq!(
            tools[3].execution_mode(),
            Some(ToolExecutionMode::Sequential)
        );
    }
}
