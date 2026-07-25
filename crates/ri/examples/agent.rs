//! Complete multi-turn agent base with a custom tool, skill, prompt template,
//! durable history, and streaming events.
//!
//! Run:
//! `$env:OPENAI_API_KEY='...'; $env:OPENAI_ENDPOINT='https://api.openai.com/v1'; cargo run -p ri --example agent`

use std::env;
use std::io::{self, Write};

use ri::agent::{AgentEvent as LoopEvent, ToolError, ToolResult};
use ri::ai::{AssistantMessageEvent, InputContent, Message};
use ri::{
    Agent, AgentEvent, AgentEvents, ApiKey, BuiltinProvider, ExpandedResource, HarnessEvent,
    PromptTemplate, Skill, Url,
};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::broadcast::error::RecvError;

#[derive(Deserialize, JsonSchema)]
struct LookupNote {
    /// Topic whose application-owned note should be returned.
    topic: String,
}

#[ri::tool(
    name = "lookup_note",
    label = "Lookup note",
    description = "Looks up an application-owned note by topic."
)]
fn lookup_note(input: LookupNote) -> Result<ToolResult, ToolError> {
    let LookupNote { topic } = input;
    let note = match topic.as_str() {
        "architecture" => "The service uses ports-and-adapters architecture.",
        "release" => "Releases require tests, clippy, and a signed tag.",
        topic => return Err(ToolError::message(format!("No note exists for {topic:?}."))),
    };
    Ok(ToolResult::text(note))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = ApiKey::new(env::var("OPENAI_API_KEY")?)?;
    let endpoint = Url::parse(&env::var("OPENAI_ENDPOINT")?)?;
    let agent = Agent::builder(
        BuiltinProvider::OpenAi,
        "gpt-5.6-sol",
        api_key,
    )
    .endpoint(endpoint)
    .thinking_level(ri_ai::ThinkingLevel::High)
    .system_prompt("You are a concise assistant. Use tools when their data is relevant.")
    .tool(lookup_note_tool()?)
    // Remove this line for a non-coding agent. The base has no implicit tools.
    .coding_tools()
    .skill(Skill::new(
        "analyst",
        "Analyze a subject using explicit assumptions and evidence.",
        "State the question, list verified evidence, then give a bounded conclusion.",
        "inline:analyst",
    ))
    .prompt_template(
        PromptTemplate::new(
            "review",
            "Review $1. Focus on: ${@:2}. If no focus was supplied, check correctness and clarity.",
            "inline:review",
        )
        .description("Review a target with optional focus areas."),
    )
    .context("Application notes are private workspace facts; do not invent missing notes.")
    .build()
    .await?;

    let event_task = tokio::spawn(print_events(agent.subscribe()));
    let stdin = io::stdin();

    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if stdin.read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input == "/exit" {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let prompt_agent = agent.clone();
        let input = input.to_owned();
        let mut prompt = tokio::spawn(async move { prompt_agent.prompt(input).await });
        let response = tokio::select! {
            result = &mut prompt => result??,
            signal = tokio::signal::ctrl_c() => {
                signal?;
                eprintln!("\n[abort] cancelling the active model/tool operation...");
                agent.session().abort().await?;
                prompt.await??
            }
        };
        println!(
            "[response] stop={:?} tokens={} cost=${:.6}",
            response.stop_reason, response.usage.total_tokens, response.usage.cost.total
        );
    }

    drop(agent);
    event_task.await??;
    Ok(())
}

async fn print_events(mut events: AgentEvents) -> io::Result<()> {
    loop {
        match events.recv().await {
            Ok(AgentEvent::Runtime(event)) => print_runtime_event(event),
            Ok(AgentEvent::Loop(event)) => print_loop_event(&event)?,
            Err(RecvError::Lagged(count)) => {
                eprintln!("[events:lagged] dropped {count} events");
            }
            Err(RecvError::Closed) => return Ok(()),
        }
    }
}

fn print_loop_event(event: &LoopEvent<Message>) -> io::Result<()> {
    match event {
        LoopEvent::AgentStart => println!("[agent:start]"),
        LoopEvent::AgentEnd { messages } => {
            println!("[agent:end] new_messages={}", messages.len());
        }
        LoopEvent::TurnStart => println!("[turn:start]"),
        LoopEvent::TurnEnd { tool_results, .. } => {
            println!("[turn:end] tool_results={}", tool_results.len());
        }
        LoopEvent::MessageUpdate {
            assistant_event: AssistantMessageEvent::TextStart { .. },
            ..
        } => {
            print!("assistant> ");
            io::stdout().flush()?;
        }
        LoopEvent::MessageUpdate {
            assistant_event:
                AssistantMessageEvent::TextDelta { delta, .. }
                | AssistantMessageEvent::ThinkingDelta { delta, .. }
                | AssistantMessageEvent::ToolcallDelta { delta, .. },
            ..
        } => {
            print!("{delta}");
            io::stdout().flush()?;
        }
        LoopEvent::MessageUpdate {
            assistant_event:
                AssistantMessageEvent::TextEnd { .. } | AssistantMessageEvent::ThinkingEnd { .. },
            ..
        } => println!(),
        LoopEvent::MessageUpdate {
            assistant_event: AssistantMessageEvent::ThinkingStart { .. },
            ..
        } => {
            print!("thinking> ");
            io::stdout().flush()?;
        }
        LoopEvent::MessageUpdate {
            assistant_event: AssistantMessageEvent::ToolcallStart { .. },
            ..
        } => {
            print!("tool-call-json> ");
            io::stdout().flush()?;
        }
        LoopEvent::MessageUpdate {
            assistant_event: AssistantMessageEvent::ToolcallEnd { tool_call, .. },
            ..
        } => println!(
            "\n[tool:requested] {} {}",
            tool_call.name, tool_call.arguments
        ),
        LoopEvent::MessageUpdate {
            assistant_event: AssistantMessageEvent::Done { reason, .. },
            ..
        } => println!("[message:done] reason={reason:?}"),
        LoopEvent::MessageUpdate {
            assistant_event: AssistantMessageEvent::Error { reason, error },
            ..
        } => println!(
            "[message:error] reason={reason:?} error={}",
            error.error_message.as_deref().unwrap_or("provider error")
        ),
        LoopEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            arguments,
        } => println!("[tool:start] id={tool_call_id} name={tool_name} args={arguments}"),
        LoopEvent::ToolExecutionUpdate {
            tool_call_id,
            tool_name,
            partial_result,
            ..
        } => print_tool_result(
            &format!("tool:update id={tool_call_id} name={tool_name}"),
            partial_result,
        ),
        LoopEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => {
            println!("[tool:end] id={tool_call_id} name={tool_name} error={is_error}");
            print_tool_result("tool:result", result);
        }
        LoopEvent::MessageStart { .. }
        | LoopEvent::MessageEnd { .. }
        | LoopEvent::MessageUpdate {
            assistant_event: AssistantMessageEvent::Start { .. },
            ..
        } => {}
    }
    Ok(())
}

fn print_tool_result(label: &str, result: &ToolResult) {
    println!("[{label}]");
    for block in &result.content {
        match block {
            InputContent::Text(text) => println!("{}", text.text),
            InputContent::Image(image) => {
                println!(
                    "<image mime={} base64_bytes={}>",
                    image.mime_type,
                    image.data.len()
                );
            }
        }
    }
    if let Some(details) = &result.details {
        println!("details={details}");
    }
    if let Some(usage) = &result.usage {
        println!("usage_tokens={}", usage.total_tokens);
    }
}

fn print_runtime_event(event: HarnessEvent) {
    match event {
        HarnessEvent::ResourceExpanded { resource, text } => {
            let (kind, name, source) = match resource {
                ExpandedResource::Skill { name, source } => ("skill", name, source),
                ExpandedResource::PromptTemplate { name, source } => {
                    ("prompt-template", name, source)
                }
            };
            println!("[resource:expanded] kind={kind} name={name} source={source}\n{text}");
        }
        HarnessEvent::PromptAccepted { operation } => {
            println!("[prompt:accepted] operation={operation}");
        }
        HarnessEvent::QueueUpdated(lengths) => println!(
            "[queue] steer={} follow_up={} next_turn={}",
            lengths.steer, lengths.follow_up, lengths.next_turn
        ),
        HarnessEvent::MessagePersisted { entry_id, role } => {
            println!("[persisted] entry={entry_id} role={role}");
        }
        HarnessEvent::SavePoint {
            operation,
            had_pending_writes,
        } => println!("[save-point] operation={operation} wrote={had_pending_writes}"),
        HarnessEvent::RetryScheduled {
            operation,
            attempt,
            max_attempts,
            delay,
            error,
        } => println!(
            "[retry:scheduled] operation={operation:?} attempt={attempt}/{max_attempts} delay_ms={} error={error}",
            delay.as_millis()
        ),
        HarnessEvent::RetryAttemptStarted { kind, reason } => {
            println!("[retry:start] kind={kind:?} reason={reason:?}");
        }
        HarnessEvent::RetryFinished {
            operation,
            success,
            attempt,
            final_error,
        } => println!(
            "[retry:end] operation={operation:?} success={success} attempt={attempt} error={final_error:?}"
        ),
        HarnessEvent::CompactionStarted { reason } => {
            println!("[compaction:start] reason={reason:?}");
        }
        HarnessEvent::CompactionFinished {
            reason,
            result,
            aborted,
            will_retry,
            error_message,
        } => {
            println!(
                "[compaction:end] reason={reason:?} aborted={aborted} will_retry={will_retry} error={error_message:?}"
            );
            if let Some(result) = result {
                println!(
                    "tokens={} -> {} first_kept={} from_hook={}\nsummary:\n{}",
                    result.tokens_before,
                    result.estimated_tokens_after,
                    result.first_kept_entry_id,
                    result.from_hook,
                    result.summary
                );
            }
        }
        HarnessEvent::BranchNavigated {
            old_leaf,
            new_leaf,
            summary_entry,
        } => println!("[branch] old={old_leaf:?} new={new_leaf:?} summary={summary_entry:?}"),
        HarnessEvent::SessionReplacing { old_session_id } => {
            println!("[session:replacing] old={old_session_id}");
        }
        HarnessEvent::SessionReplaced {
            session_id,
            generation,
        } => println!("[session:replaced] id={session_id} generation={generation}"),
        HarnessEvent::Settled {
            operation,
            next_turn,
        } => println!("[settled] operation={operation} next_turn={next_turn}"),
    }
}
