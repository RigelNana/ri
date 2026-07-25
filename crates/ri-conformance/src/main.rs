//! Command-line entry point for the conformance harness.

use std::io::Write;
use std::path::{Path, PathBuf};

use clap::{Parser, Subcommand};
use ri_conformance::{Request, compare, encode, execute, load, normalize, read_input, validate};

#[derive(Debug, Parser)]
#[command(name = "ri-conformance", version, about)]
struct Cli {
    /// Workspace root.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// Manifest path relative to the workspace root.
    #[arg(long, default_value = "conformance/manifest.yaml")]
    manifest: PathBuf,

    #[command(subcommand)]
    command: Action,
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Validate the manifest and reference test inventory.
    Validate,
    /// Generate the stable reference-test inventory.
    Inventory,
    /// Canonicalize an arbitrary JSON value.
    Normalize {
        /// Read from this file instead of standard input.
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Execute a versioned dual-runner request.
    Run {
        /// Read from this file instead of standard input.
        #[arg(long)]
        input: Option<PathBuf>,
    },
    /// Compare every fixture using Rust and reference runners.
    Compare,
}

fn main() {
    if let Err(error) = entry() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn entry() -> ri_conformance::Result<()> {
    let cli = Cli::parse();
    let root = absolute(&cli.root);
    let manifest_path = root.join(&cli.manifest);
    let manifest = load(manifest_path)?;
    match cli.command {
        Action::Validate => {
            let inventory = validate(&root, &manifest)?;
            println!(
                "validated {} features and {} reference tests",
                manifest.features.len(),
                inventory.len()
            );
        }
        Action::Inventory => {
            let inventory = validate(&root, &manifest)?;
            ri_conformance::write_inventory(&root, &manifest, &inventory)?;
            println!(
                "wrote {} mappings to {}",
                inventory.len(),
                manifest.test_inventory.output.display()
            );
        }
        Action::Normalize { input } => {
            let bytes = read_input(input.as_deref())?;
            let value = serde_json::from_slice(&bytes)?;
            write_json(&normalize(value))?;
        }
        Action::Run { input } => {
            let bytes = read_input(input.as_deref())?;
            let request: Request = serde_json::from_slice(&bytes)?;
            write_json(&execute(request)?)?;
        }
        Action::Compare => {
            let count = compare(&root, &manifest)?;
            println!("compared {count} fixtures");
        }
    }
    Ok(())
}

fn absolute(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}

fn write_json(value: &serde_json::Value) -> ri_conformance::Result<()> {
    std::io::stdout()
        .lock()
        .write_all(&encode(value)?)
        .map_err(|source| ri_conformance::Error::Io {
            path: PathBuf::from("<stdout>"),
            source,
        })
}
