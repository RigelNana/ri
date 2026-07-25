//! Black-box behavior checks for the `ri` executable.

use std::io::{BufRead as _, Read as _, Write as _};
use std::process::{Command, Stdio};

fn ri(arguments: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(arguments)
        .output()
        .expect("ri subprocess should start")
}

#[test]
fn help_is_successful_and_stays_on_stdout() {
    let output = ri(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("native coding agent"));
    assert!(stdout.contains("--mode"));
    assert!(output.stderr.is_empty());
}

#[test]
fn version_is_successful() {
    let output = ri(&["--version"]);
    assert!(output.status.success());
    assert!(String::from_utf8(output.stdout).unwrap().starts_with("ri "));
}

#[test]
fn parse_errors_are_nonzero_and_stay_on_stderr() {
    let output = ri(&["--mode", "yaml"]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("invalid value")
    );
}

#[test]
fn missing_prompt_fails_before_runtime_construction() {
    let output = ri(&["--print"]);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr)
            .unwrap()
            .contains("no prompt was provided")
    );
}

#[test]
fn local_package_round_trip_uses_real_settings_and_resolver() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = temporary.path().join("workspace");
    let agent_dir = temporary.path().join("agent");
    let package = temporary.path().join("fixture-package");
    std::fs::create_dir_all(&workspace).unwrap();
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(
        package.join("ri-package.toml"),
        r#"
[package]
name = "fixture"
version = "1.0.0"

[resources]
prompts = ["prompt.md"]
"#,
    )
    .unwrap();
    std::fs::write(package.join("prompt.md"), "Review $@").unwrap();
    let source = package.to_string_lossy().into_owned();

    let run = |arguments: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_ri"))
            .args(arguments)
            .current_dir(&workspace)
            .env("RI_AGENT_DIR", &agent_dir)
            .output()
            .expect("ri subprocess should start")
    };

    let installed = run(&["--offline", "install", &source]);
    assert!(
        installed.status.success(),
        "{}",
        String::from_utf8_lossy(&installed.stderr)
    );
    assert!(String::from_utf8_lossy(&installed.stdout).contains("installed local:"));

    let listed = run(&["--offline", "list", "--json"]);
    assert!(
        listed.status.success(),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    let packages: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(packages[0]["name"], "fixture");
    assert_eq!(packages[0]["version"], "1.0.0");

    let removed = run(&["--offline", "remove", &source]);
    assert!(
        removed.status.success(),
        "{}",
        String::from_utf8_lossy(&removed.stderr)
    );
    let settings: serde_json::Value =
        serde_json::from_slice(&std::fs::read(agent_dir.join("settings.json")).unwrap()).unwrap();
    assert_eq!(settings["packages"], serde_json::json!([]));
}

#[test]
fn resource_enablement_persists_without_runtime_construction() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = temporary.path().join("workspace");
    let agent_dir = temporary.path().join("agent");
    std::fs::create_dir_all(&workspace).unwrap();
    let run = |arguments: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_ri"))
            .args(arguments)
            .current_dir(&workspace)
            .env("RI_AGENT_DIR", &agent_dir)
            .output()
            .unwrap()
    };

    let disabled = run(&["--offline", "resource", "disable", "tool", "bash"]);
    assert!(
        disabled.status.success(),
        "{}",
        String::from_utf8_lossy(&disabled.stderr)
    );
    let listed = run(&["--offline", "resource", "list", "--kind", "tool", "--json"]);
    let resources: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    let bash = resources
        .as_array()
        .unwrap()
        .iter()
        .filter(|resource| resource["name"] == "bash")
        .collect::<Vec<_>>();
    assert_eq!(bash.len(), 1);
    assert_eq!(bash[0]["enabled"], false);
    let listed = run(&[
        "--offline",
        "resource",
        "list",
        "--kind",
        "tool",
        "--scope",
        "global",
        "--json",
    ]);
    let resources: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(resources[0]["name"], "bash");
    assert_eq!(resources[0]["enabled"], false);

    let enabled = run(&["--offline", "resource", "enable", "tool", "bash"]);
    assert!(
        enabled.status.success(),
        "{}",
        String::from_utf8_lossy(&enabled.stderr)
    );
    let listed = run(&["--offline", "resource", "list", "--kind", "tool", "--json"]);
    let resources: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert!(
        resources
            .as_array()
            .unwrap()
            .iter()
            .any(|resource| resource["name"] == "bash" && resource["enabled"] == true)
    );
}

#[cfg(feature = "rpc")]
#[test]
fn rpc_mode_uses_strict_jsonl_and_shared_runtime_state() {
    let temporary = tempfile::tempdir().unwrap();
    let workspace = temporary.path().join("workspace");
    let agent_dir = temporary.path().join("agent");
    std::fs::create_dir_all(&workspace).unwrap();

    let listed = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args(["--offline", "model", "list", "--all", "--json"])
        .current_dir(&workspace)
        .env("RI_AGENT_DIR", &agent_dir)
        .output()
        .unwrap();
    assert!(
        listed.status.success(),
        "{}",
        String::from_utf8_lossy(&listed.stderr)
    );
    let models: serde_json::Value = serde_json::from_slice(&listed.stdout).unwrap();
    let model = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["provider"] == "anthropic")
        .expect("the built-in Anthropic catalog should contain a model");
    let selector = format!(
        "{}/{}",
        model["provider"].as_str().unwrap(),
        model["id"].as_str().unwrap()
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_ri"))
        .args([
            "--offline",
            "--mode",
            "rpc",
            "--no-session",
            "--model",
            &selector,
            "--api-key",
            "test-key",
        ])
        .current_dir(&workspace)
        .env("RI_AGENT_DIR", &agent_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(
        child.stdin.as_mut().unwrap(),
        r#"{{"id":"state-1","type":"get_state"}}"#
    )
    .unwrap();
    let mut line = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    drop(child.stdin.take());
    let status = child.wait().unwrap();
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(status.success(), "{}", stderr);
    assert!(!line.contains('\r'));
    assert_eq!(line.matches('\n').count(), 1);
    let response: serde_json::Value = serde_json::from_str(&line).unwrap();
    assert_eq!(response["id"], "state-1");
    assert_eq!(response["type"], "response");
    assert_eq!(response["success"], true);
    assert_eq!(response["command"], "get_state");
    assert_eq!(
        response["data"]["model"]["provider"],
        model["provider"].clone()
    );
    assert_eq!(response["data"]["model"]["id"], model["id"].clone());
}
