extern crate self as ri;

pub use ri_agent as agent;

#[doc(hidden)]
pub mod __private {
    pub use async_trait::async_trait;
}

#[derive(schemars::JsonSchema, serde::Deserialize)]
struct AddArguments {
    left: i64,
    right: i64,
}

/// Adds two integers.
#[ri_macros::tool(label = "Add", execution = "sequential")]
async fn add(
    _context: agent::ToolCallContext,
    arguments: AddArguments,
) -> Result<agent::ToolResult, agent::ToolError> {
    Ok(agent::ToolResult::text(
        (arguments.left + arguments.right).to_string(),
    ))
}

#[ri_macros::tool]
fn negate(arguments: AddArguments) -> Result<agent::ToolResult, agent::ToolError> {
    Ok(agent::ToolResult::text((-arguments.left).to_string()))
}

fn main() {
    let tool: std::sync::Arc<dyn agent::Tool> = add_tool().expect("generated tool");
    assert_eq!(tool.definition().name, "add");
    let sync_tool: std::sync::Arc<dyn agent::Tool> = negate_tool().expect("generated sync tool");
    assert_eq!(sync_tool.definition().name, "negate");
}
