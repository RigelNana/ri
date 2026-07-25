//! Pi session, settings, models, and RPC compatibility coverage.

use proptest::prelude::*;
use ri_compat::{
    PiSessionVersion, PiTransport, export_models, export_session, export_settings, import_models,
    import_session, import_session_with_ids, import_settings,
};
use ri_rpc::{AgentMessage, ClientFrame, QueueMode, SessionEntry};
use serde_json::{Value, json};

#[test]
fn v1_migrates_ids_compaction_pointer_and_hook_message() {
    let input = concat!(
        r#"{"type":"session","id":"s","timestamp":"2026-01-01T00:00:00Z","cwd":"/tmp"}"#,
        "\r\n",
        r#"{"type":"message","timestamp":"2026-01-01T00:00:01Z","message":{"role":"user","content":"hello","timestamp":1}}"#,
        "\r\n",
        r#"{"type":"message","timestamp":"2026-01-01T00:00:02Z","message":{"role":"hookMessage","customType":"ext","content":"context","display":true,"timestamp":2}}"#,
        "\r\n",
        r#"{"type":"compaction","timestamp":"2026-01-01T00:00:03Z","summary":"summary","firstKeptEntryIndex":1,"tokensBefore":100}"#
    );
    let mut sequence = 0;
    let session = import_session_with_ids(input.as_bytes(), || {
        sequence += 1;
        format!("id{sequence:06}")
    })
    .unwrap();

    assert_eq!(session.source_version, PiSessionVersion::V1);
    assert_eq!(session.entries[0].id(), "id000001");
    assert_eq!(session.entries[1].parent_id(), Some("id000001"));
    match &session.entries[1] {
        SessionEntry::Message {
            message: AgentMessage::Custom { custom_type, .. },
            ..
        } => assert_eq!(custom_type, "ext"),
        entry => panic!("unexpected migrated entry: {entry:?}"),
    }
    match &session.entries[2] {
        SessionEntry::Compaction {
            first_kept_entry_id,
            ..
        } => assert_eq!(first_kept_entry_id.as_deref(), Some("id000001")),
        entry => panic!("unexpected compaction entry: {entry:?}"),
    }

    let exported = export_session(&session, PiSessionVersion::V1).unwrap();
    let text = String::from_utf8(exported.clone()).unwrap();
    assert!(!text.lines().next().unwrap().contains("\"version\""));
    assert!(text.contains("\"firstKeptEntryIndex\":1"));
    assert!(text.contains("\"role\":\"hookMessage\""));

    sequence = 0;
    let reimported = import_session_with_ids(&exported, || {
        sequence += 1;
        format!("id{sequence:06}")
    })
    .unwrap();
    assert_eq!(reimported, session);
}

#[test]
fn v2_and_v3_export_round_trip_without_semantic_loss() {
    let input = concat!(
        r#"{"type":"session","version":3,"id":"s","timestamp":"2026-01-01T00:00:00Z","cwd":"/tmp","parentSession":"parent.jsonl"}"#,
        "\n",
        r#"{"type":"message","id":"a","parentId":null,"timestamp":"2026-01-01T00:00:01Z","message":{"role":"custom","customType":"ext","content":"ctx","display":false,"details":{"x":1},"timestamp":1}}"#,
        "\n",
        r#"{"type":"session_info","id":"b","parentId":"a","timestamp":"2026-01-01T00:00:02Z","name":"demo"}"#,
        "\n"
    );
    let session = import_session(input.as_bytes()).unwrap();

    for version in [PiSessionVersion::V2, PiSessionVersion::V3] {
        let exported = export_session(&session, version).unwrap();
        let reimported = import_session(&exported).unwrap();
        assert_eq!(reimported.header, session.header);
        assert_eq!(reimported.entries, session.entries);
        assert_eq!(reimported.source_version, version);
    }
}

#[test]
fn settings_import_applies_all_reference_migrations() {
    let settings = import_settings(
        br#"{
            "queueMode": "all",
            "websockets": true,
            "skills": {
                "enableSkillCommands": false,
                "customDirectories": ["skills/a"]
            },
            "retry": {"maxDelayMs": 1234, "maxRetries": 2},
            "futureSetting": {"kept": true}
        }"#,
    )
    .unwrap();

    assert_eq!(settings.steering_mode, Some(QueueMode::All));
    assert_eq!(settings.transport, Some(PiTransport::Websocket));
    assert_eq!(settings.skills, Some(vec!["skills/a".to_owned()]));
    assert_eq!(settings.enable_skill_commands, Some(false));
    assert_eq!(
        settings
            .retry
            .as_ref()
            .and_then(|retry| retry.provider.as_ref())
            .and_then(|provider| provider.max_retry_delay_ms),
        Some(1234)
    );
    assert_eq!(
        settings.extra.get("futureSetting"),
        Some(&json!({"kept": true}))
    );

    let exported = export_settings(&settings).unwrap();
    assert_eq!(import_settings(exported.as_bytes()).unwrap(), settings);
}

#[test]
fn models_import_accepts_comments_but_never_resolves_key_sources() {
    let input = br#"{
        // key expressions remain inert strings
        "providers": {
            "local": {
                "baseUrl": "http://localhost:11434/v1", /* URL comment */
                "api": "openai-completions",
                "apiKey": "!print-secret",
                "headers": {"x-key": "$MODEL_KEY"},
                "models": [{
                    "id": "model",
                    "reasoning": true,
                    "thinkingLevelMap": {"off": null, "max": "max"},
                    "input": ["text", "image"],
                    "compat": {
                        "supportsDeveloperRole": false,
                        "supportsOpenAIGrammarTools": true
                    }
                }]
            }
        }
    }"#;
    let models = import_models(input).unwrap();
    let provider = &models.providers["local"];
    assert_eq!(
        provider.api_key.as_ref().unwrap().0,
        "!print-secret".to_owned()
    );
    assert_eq!(
        provider.headers.as_ref().unwrap()["x-key"].0,
        "$MODEL_KEY".to_owned()
    );
    assert_eq!(
        provider.models.as_ref().unwrap()[0]
            .compat
            .as_ref()
            .unwrap()
            .supports_open_ai_grammar_tools,
        Some(true)
    );

    let exported = export_models(&models).unwrap();
    assert_eq!(import_models(exported.as_bytes()).unwrap(), models);
}

#[test]
fn pi_rpc_codec_reexports_strict_typed_framing() {
    let frames = ri_compat::decode_pi_client_jsonl(
        "{\"id\":\"r\",\"type\":\"prompt\",\"message\":\"a\u{2028}b\u{2029}c\"}\r\n".as_bytes(),
    )
    .unwrap();
    assert_eq!(frames.len(), 1);
    assert!(matches!(&frames[0], ClientFrame::Request(_)));

    let encoded = ri_compat::encode_pi_client_jsonl(frames).unwrap();
    assert!(encoded.ends_with(b"\n"));
    assert!(
        encoded
            .windows(3)
            .any(|bytes| bytes == "\u{2028}".as_bytes())
    );
    assert!(
        encoded
            .windows(3)
            .any(|bytes| bytes == "\u{2029}".as_bytes())
    );
}

proptest! {
    #[test]
    fn v3_user_text_round_trips(text in any::<String>()) {
        let records = [
            json!({
                "type": "session",
                "version": 3,
                "id": "s",
                "timestamp": "2026-01-01T00:00:00Z",
                "cwd": "/tmp"
            }),
            json!({
                "type": "message",
                "id": "a",
                "parentId": Value::Null,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "role": "user",
                    "content": text,
                    "timestamp": 1
                }
            }),
        ];
        let mut input = Vec::new();
        for record in records {
            serde_json::to_writer(&mut input, &record).unwrap();
            input.push(b'\n');
        }
        let imported = import_session(&input).unwrap();
        let exported = export_session(&imported, PiSessionVersion::V3).unwrap();
        let reimported = import_session(&exported).unwrap();
        prop_assert_eq!(reimported, imported);
    }
}
