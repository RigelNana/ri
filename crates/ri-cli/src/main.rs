//! Standalone `ri` command-line entrypoint.

use std::io::IsTerminal as _;
use std::process::ExitCode;

use ri_cli::app;
use ri_cli::output::{Output, is_broken_pipe};
use ri_cli::sdk_adapter;
use ri_cli::{Cli, CliError, IoCapabilities};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> ExitCode {
    let cli = match Cli::try_parse_compatible_from(std::env::args_os()) {
        Ok(cli) => cli,
        Err(error) => {
            let exit = if error.use_stderr() { 2 } else { 0 };
            let _ = error.print();
            return ExitCode::from(exit);
        }
    };
    init_tracing(cli.verbose);
    let io = IoCapabilities::new(
        std::io::stdin().is_terminal(),
        std::io::stdout().is_terminal(),
    );
    let output = Output::stdio();

    let prepared = match app::prepare_run(&cli, io).await {
        Ok(prepared) => prepared,
        Err(error) => return report(error, &output).await,
    };
    let runtime = match sdk_adapter::build(&cli, io, &output).await {
        Ok(runtime) => runtime,
        Err(error) => return report(error, &output).await,
    };
    match app::execute_prepared(cli, runtime, output.clone(), prepared).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => report(error, &output).await,
    }
}

fn init_tracing(verbose: bool) {
    let fallback = if verbose { "info" } else { "warn" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(fallback));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .try_init();
}

async fn report(error: CliError, output: &Output) -> ExitCode {
    if is_broken_pipe(&error) {
        return ExitCode::SUCCESS;
    }
    let code = match &error {
        CliError::Interrupted => 130,
        CliError::InvalidArguments(_)
        | CliError::InvalidConfig { .. }
        | CliError::FeatureUnavailable { .. }
        | CliError::MissingPrompt
        | CliError::NotFound { .. } => 2,
        _ => 1,
    };
    let _ = output.stderr_line(&format!("error: {error}")).await;
    ExitCode::from(code)
}
