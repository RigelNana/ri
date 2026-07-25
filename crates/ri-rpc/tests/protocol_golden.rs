//! Golden wire-format coverage for the complete typed RPC protocol.

use std::collections::BTreeSet;

use ri_rpc::{
    AssistantMessageEvent, ClientFrame, CommandName, Event, ExtensionUiRequest,
    ExtensionUiResponse, InvalidClientMessage, Request, RequestId, Response, ServerFrame,
};

#[test]
fn all_command_variants_match_pi_wire_names() {
    let cases = [
        (r#"{"type":"prompt","message":"hi"}"#, CommandName::Prompt),
        (
            r#"{"type":"steer","message":"hi","images":[{"type":"image","data":"AA==","mimeType":"image/png"}]}"#,
            CommandName::Steer,
        ),
        (
            r#"{"type":"follow_up","message":"later"}"#,
            CommandName::FollowUp,
        ),
        (r#"{"type":"abort"}"#, CommandName::Abort),
        (
            r#"{"type":"new_session","parentSession":"parent.jsonl"}"#,
            CommandName::NewSession,
        ),
        (r#"{"type":"get_state"}"#, CommandName::GetState),
        (
            r#"{"type":"set_model","provider":"p","modelId":"m"}"#,
            CommandName::SetModel,
        ),
        (r#"{"type":"cycle_model"}"#, CommandName::CycleModel),
        (
            r#"{"type":"get_available_models"}"#,
            CommandName::GetAvailableModels,
        ),
        (
            r#"{"type":"set_thinking_level","level":"xhigh"}"#,
            CommandName::SetThinkingLevel,
        ),
        (
            r#"{"type":"cycle_thinking_level"}"#,
            CommandName::CycleThinkingLevel,
        ),
        (
            r#"{"type":"get_available_thinking_levels"}"#,
            CommandName::GetAvailableThinkingLevels,
        ),
        (
            r#"{"type":"set_steering_mode","mode":"all"}"#,
            CommandName::SetSteeringMode,
        ),
        (
            r#"{"type":"set_follow_up_mode","mode":"one-at-a-time"}"#,
            CommandName::SetFollowUpMode,
        ),
        (
            r#"{"type":"compact","customInstructions":"focus"}"#,
            CommandName::Compact,
        ),
        (
            r#"{"type":"set_auto_compaction","enabled":true}"#,
            CommandName::SetAutoCompaction,
        ),
        (
            r#"{"type":"set_auto_retry","enabled":false}"#,
            CommandName::SetAutoRetry,
        ),
        (r#"{"type":"abort_retry"}"#, CommandName::AbortRetry),
        (
            r#"{"type":"bash","command":"echo hi","excludeFromContext":true}"#,
            CommandName::Bash,
        ),
        (r#"{"type":"abort_bash"}"#, CommandName::AbortBash),
        (
            r#"{"type":"get_session_stats"}"#,
            CommandName::GetSessionStats,
        ),
        (
            r#"{"type":"export_html","outputPath":"out.html"}"#,
            CommandName::ExportHtml,
        ),
        (
            r#"{"type":"switch_session","sessionPath":"s.jsonl"}"#,
            CommandName::SwitchSession,
        ),
        (r#"{"type":"fork","entryId":"e1"}"#, CommandName::Fork),
        (r#"{"type":"clone"}"#, CommandName::Clone),
        (
            r#"{"type":"get_fork_messages"}"#,
            CommandName::GetForkMessages,
        ),
        (
            r#"{"type":"get_entries","since":"e1"}"#,
            CommandName::GetEntries,
        ),
        (r#"{"type":"get_tree"}"#, CommandName::GetTree),
        (
            r#"{"type":"get_last_assistant_text"}"#,
            CommandName::GetLastAssistantText,
        ),
        (
            r#"{"type":"set_session_name","name":"demo"}"#,
            CommandName::SetSessionName,
        ),
        (r#"{"type":"get_messages"}"#, CommandName::GetMessages),
        (r#"{"type":"get_commands"}"#, CommandName::GetCommands),
    ];

    assert_eq!(cases.len(), 32);
    let mut names = BTreeSet::new();
    for (json, expected) in cases {
        let request: Request = serde_json::from_str(json).unwrap();
        assert_eq!(request.command.name(), expected);
        names.insert(expected);
        let encoded = serde_json::to_string(&request).unwrap();
        assert_eq!(serde_json::from_str::<Request>(&encoded).unwrap(), request);
    }
    assert_eq!(names.len(), 32);
}

#[test]
fn all_response_variants_decode_and_round_trip() {
    let model = r#"{"id":"m","name":"M","api":"api","provider":"p","baseUrl":"https://example.test","reasoning":false,"input":["text"],"contextWindow":8,"maxTokens":4,"cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0}}"#;
    let session_snapshot = format!(
        r#"{{"model":{model},"thinkingLevel":"off","isStreaming":false,"isCompacting":false,"steeringMode":"all","followUpMode":"one-at-a-time","sessionId":"s","autoCompactionEnabled":true,"messageCount":0,"pendingMessageCount":0}}"#
    );
    let compaction = r#"{"summary":"s","firstKeptEntryId":"e","tokensBefore":1}"#;
    let usage_totals = r#"{"sessionId":"s","userMessages":0,"assistantMessages":0,"toolCalls":0,"toolResults":0,"totalMessages":0,"tokens":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"total":0},"cost":0.0}"#;

    let mut records = vec![
        success("prompt", None),
        success("steer", None),
        success("follow_up", None),
        success("abort", None),
        success("new_session", Some(r#"{"cancelled":false}"#)),
        success("get_state", Some(&session_snapshot)),
        success("set_model", Some(model)),
        success("cycle_model", Some("null")),
        success(
            "get_available_models",
            Some(&format!(r#"{{"models":[{model}]}}"#)),
        ),
        success("set_thinking_level", None),
        success("cycle_thinking_level", Some(r#"{"level":"high"}"#)),
        success(
            "get_available_thinking_levels",
            Some(r#"{"levels":["off","high"]}"#),
        ),
        success("set_steering_mode", None),
        success("set_follow_up_mode", None),
        success("compact", Some(compaction)),
        success("set_auto_compaction", None),
        success("set_auto_retry", None),
        success("abort_retry", None),
        success(
            "bash",
            Some(r#"{"output":"ok","exitCode":0,"cancelled":false,"truncated":false}"#),
        ),
        success("abort_bash", None),
        success("get_session_stats", Some(usage_totals)),
        success("export_html", Some(r#"{"path":"out.html"}"#)),
        success("switch_session", Some(r#"{"cancelled":false}"#)),
        success("fork", Some(r#"{"text":"hi","cancelled":false}"#)),
        success("clone", Some(r#"{"cancelled":false}"#)),
        success("get_fork_messages", Some(r#"{"messages":[]}"#)),
        success("get_entries", Some(r#"{"entries":[],"leafId":null}"#)),
        success("get_tree", Some(r#"{"tree":[],"leafId":null}"#)),
        success("get_last_assistant_text", Some(r#"{"text":null}"#)),
        success("set_session_name", None),
        success("get_messages", Some(r#"{"messages":[]}"#)),
        success("get_commands", Some(r#"{"commands":[]}"#)),
    ];
    records.push(
        r#"{"id":"r","type":"response","command":"future_command","success":false,"error":"no"}"#
            .to_owned(),
    );

    assert_eq!(records.len(), 33);
    for record in records {
        let response: Response = serde_json::from_str(&record).unwrap_or_else(|error| {
            panic!("failed to decode {record}: {error}");
        });
        let encoded = serde_json::to_string(&response).unwrap();
        assert_eq!(
            serde_json::from_str::<Response>(&encoded).unwrap(),
            response
        );
    }
}

fn success(command: &str, data: Option<&str>) -> String {
    match data {
        Some(data) => format!(
            r#"{{"id":"r","type":"response","command":"{command}","success":true,"data":{data}}}"#
        ),
        None => format!(r#"{{"id":"r","type":"response","command":"{command}","success":true}}"#),
    }
}

#[test]
fn all_event_variants_decode_and_round_trip() {
    let user = r#"{"role":"user","content":"hi","timestamp":1}"#;
    let assistant = r#"{"role":"assistant","content":[{"type":"text","text":"ok"}],"api":"api","provider":"p","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}},"stopReason":"stop","timestamp":2}"#;
    let tool_result = r#"{"content":[{"type":"text","text":"ok"}],"details":null}"#;
    let entry = format!(
        r#"{{"type":"message","id":"e","parentId":null,"timestamp":"2026-01-01T00:00:00Z","message":{user}}}"#
    );
    let compaction = r#"{"summary":"s","firstKeptEntryId":"e","tokensBefore":1}"#;
    let records = vec![
        r#"{"type":"agent_start"}"#.to_owned(),
        format!(r#"{{"type":"agent_end","messages":[{assistant}],"willRetry":false}}"#),
        r#"{"type":"agent_settled"}"#.to_owned(),
        r#"{"type":"turn_start"}"#.to_owned(),
        format!(r#"{{"type":"turn_end","message":{assistant},"toolResults":[]}}"#),
        format!(r#"{{"type":"message_start","message":{user}}}"#),
        format!(
            r#"{{"type":"message_update","message":{assistant},"assistantMessageEvent":{{"type":"text_delta","contentIndex":0,"delta":"x","partial":{assistant}}}}}"#
        ),
        format!(r#"{{"type":"message_end","message":{assistant}}}"#),
        r#"{"type":"tool_execution_start","toolCallId":"c","toolName":"bash","args":{"command":"x"}}"#.to_owned(),
        format!(
            r#"{{"type":"tool_execution_update","toolCallId":"c","toolName":"bash","args":{{"command":"x"}},"partialResult":{tool_result}}}"#
        ),
        format!(
            r#"{{"type":"tool_execution_end","toolCallId":"c","toolName":"bash","result":{tool_result},"isError":false}}"#
        ),
        r#"{"type":"bash_execution_update","id":"r","delta":"x"}"#.to_owned(),
        r#"{"type":"queue_update","steering":["a"],"followUp":["b"]}"#.to_owned(),
        r#"{"type":"compaction_start","reason":"threshold"}"#.to_owned(),
        format!(
            r#"{{"type":"compaction_end","reason":"manual","result":{compaction},"aborted":false,"willRetry":false}}"#
        ),
        r#"{"type":"auto_retry_start","attempt":1,"maxAttempts":3,"delayMs":10,"errorMessage":"busy"}"#.to_owned(),
        r#"{"type":"auto_retry_end","success":false,"attempt":3,"finalError":"busy"}"#.to_owned(),
        r#"{"type":"summarization_retry_scheduled","attempt":1,"maxAttempts":3,"delayMs":10,"errorMessage":"busy"}"#.to_owned(),
        r#"{"type":"summarization_retry_attempt_start","source":"branchSummary"}"#.to_owned(),
        r#"{"type":"summarization_retry_finished"}"#.to_owned(),
        format!(r#"{{"type":"entry_appended","entry":{entry}}}"#),
        r#"{"type":"session_info_changed","name":"demo"}"#.to_owned(),
        r#"{"type":"thinking_level_changed","level":"max"}"#.to_owned(),
        r#"{"type":"extension_error","extensionPath":"x.ts","event":"tool_call","error":"boom"}"#.to_owned(),
    ];

    assert_eq!(records.len(), 24);
    for record in records {
        let event: Event = serde_json::from_str(&record).unwrap_or_else(|error| {
            panic!("failed to decode {record}: {error}");
        });
        let frame = ServerFrame::Event(event.clone());
        let encoded = serde_json::to_string(&frame).unwrap();
        assert_eq!(
            serde_json::from_str::<ServerFrame>(&encoded).unwrap(),
            frame
        );
    }
}

#[test]
fn all_assistant_delta_variants_decode_and_round_trip() {
    let assistant = r#"{"role":"assistant","content":[],"api":"api","provider":"p","model":"m","usage":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0,"totalTokens":0,"cost":{"input":0.0,"output":0.0,"cacheRead":0.0,"cacheWrite":0.0,"total":0.0}},"stopReason":"stop","timestamp":2}"#;
    let records = [
        format!(r#"{{"type":"start","partial":{assistant}}}"#),
        format!(r#"{{"type":"text_start","contentIndex":0,"partial":{assistant}}}"#),
        format!(r#"{{"type":"text_delta","contentIndex":0,"delta":"x","partial":{assistant}}}"#),
        format!(r#"{{"type":"text_end","contentIndex":0,"content":"x","partial":{assistant}}}"#),
        format!(r#"{{"type":"thinking_start","contentIndex":0,"partial":{assistant}}}"#),
        format!(
            r#"{{"type":"thinking_delta","contentIndex":0,"delta":"x","partial":{assistant}}}"#
        ),
        format!(
            r#"{{"type":"thinking_end","contentIndex":0,"content":"x","partial":{assistant}}}"#
        ),
        format!(r#"{{"type":"toolcall_start","contentIndex":0,"partial":{assistant}}}"#),
        format!(
            r#"{{"type":"toolcall_delta","contentIndex":0,"delta":"{{","partial":{assistant}}}"#
        ),
        format!(
            r#"{{"type":"toolcall_end","contentIndex":0,"toolCall":{{"type":"toolCall","id":"c","name":"bash","arguments":{{"command":"x"}}}},"partial":{assistant}}}"#
        ),
        format!(r#"{{"type":"done","reason":"stop","message":{assistant}}}"#),
        format!(r#"{{"type":"error","reason":"aborted","error":{assistant}}}"#),
    ];

    for record in records {
        let event: AssistantMessageEvent = serde_json::from_str(&record).unwrap_or_else(|error| {
            panic!("failed to decode {record}: {error}");
        });
        let encoded = serde_json::to_string(&event).unwrap();
        assert_eq!(
            serde_json::from_str::<AssistantMessageEvent>(&encoded).unwrap(),
            event
        );
    }
}

#[test]
fn unknown_command_preserves_a_valid_request_id() {
    let frame: ClientFrame =
        serde_json::from_str(r#"{"id":"test","type":"future_command","x":1}"#).unwrap();
    assert_eq!(
        frame,
        ClientFrame::Invalid(InvalidClientMessage {
            id: Some(RequestId::new("test")),
            command: "future_command".to_owned(),
            error: "Unknown command: future_command".to_owned(),
        })
    );
}

#[test]
fn every_extension_ui_method_and_result_round_trips() {
    let requests = [
        r#"{"type":"extension_ui_request","id":"1","method":"select","title":"Pick","options":["a"],"timeout":10}"#,
        r#"{"type":"extension_ui_request","id":"2","method":"confirm","title":"Sure?","message":"Body","timeout":10}"#,
        r#"{"type":"extension_ui_request","id":"3","method":"input","title":"Value","placeholder":"x","timeout":10}"#,
        r#"{"type":"extension_ui_request","id":"4","method":"editor","title":"Edit","prefill":"x"}"#,
        r#"{"type":"extension_ui_request","id":"5","method":"notify","message":"Hi","notifyType":"warning"}"#,
        r#"{"type":"extension_ui_request","id":"6","method":"setStatus","statusKey":"k","statusText":"v"}"#,
        r#"{"type":"extension_ui_request","id":"7","method":"setWidget","widgetKey":"k","widgetLines":["v"],"widgetPlacement":"belowEditor"}"#,
        r#"{"type":"extension_ui_request","id":"8","method":"setTitle","title":"Title"}"#,
        r#"{"type":"extension_ui_request","id":"9","method":"set_editor_text","text":"Text"}"#,
    ];
    for json in requests {
        let request: ExtensionUiRequest = serde_json::from_str(json).unwrap();
        let encoded = serde_json::to_string(&request).unwrap();
        assert_eq!(
            serde_json::from_str::<ExtensionUiRequest>(&encoded).unwrap(),
            request
        );
    }

    let responses = [
        r#"{"type":"extension_ui_response","id":"1","value":"a"}"#,
        r#"{"type":"extension_ui_response","id":"2","confirmed":true}"#,
        r#"{"type":"extension_ui_response","id":"3","cancelled":true}"#,
    ];
    for json in responses {
        let response: ExtensionUiResponse = serde_json::from_str(json).unwrap();
        let encoded = serde_json::to_string(&response).unwrap();
        assert_eq!(
            serde_json::from_str::<ExtensionUiResponse>(&encoded).unwrap(),
            response
        );
    }
}
