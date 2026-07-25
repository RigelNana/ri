//! Creates one durable runtime and four frontend views over it.

use std::sync::Arc;

use ri_ai::{InMemoryCredentialStore, Models, SystemAuthContext, catalog::builtin_providers};
use ri_harness::PromptOptions;
use ri_sdk::{FrontendMode, ModelRuntime, SessionBuilder};
use ri_session::{CreateOptions, FileRepository};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = std::env::var("RI_PROVIDER").unwrap_or_else(|_| "anthropic".to_owned());
    let model = std::env::var("RI_MODEL").unwrap_or_else(|_| "claude-sonnet-4-6".to_owned());
    let session_dir = std::env::var_os("RI_SESSION_DIR").map_or_else(
        || {
            std::env::current_dir()
                .expect("cwd")
                .join(".ri")
                .join("sessions")
        },
        std::path::PathBuf::from,
    );

    let models = Models::with_providers(
        Arc::new(InMemoryCredentialStore::default()),
        Arc::new(SystemAuthContext),
        builtin_providers(),
    );
    let runtime = SessionBuilder::new(Arc::new(ModelRuntime::new(models)))
        .catalog_model(provider, model)
        .create_session(
            Arc::new(FileRepository::new(session_dir)),
            CreateOptions::new(std::env::current_dir()?.display().to_string()),
        )
        .system_prompt("You are a concise coding assistant.")
        .build()
        .await?;

    // These are presentation handles, not independent agents or transcripts.
    let _interactive = runtime.frontend(FrontendMode::Interactive);
    let print = runtime.frontend(FrontendMode::Print);
    let _json = runtime.frontend(FrontendMode::Json);
    let _rpc = runtime.frontend(FrontendMode::Rpc);

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if !prompt.is_empty() {
        let outcome = print.prompt(prompt, PromptOptions::interactive()).await?;
        println!("{outcome:?}");
    }
    Ok(())
}
