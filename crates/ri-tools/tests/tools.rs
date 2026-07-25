//! Integration coverage for local built-in tools.

use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ri_tools::{
    BashInput, Content, DEFAULT_MAX_LINES, Edit, EditInput, FindInput, GREP_MAX_LINE_LENGTH,
    GrepInput, LsInput, ReadInput, ToolError, Tools, WriteInput,
};
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

fn tiny_bmp() -> Vec<u8> {
    let mut bytes = vec![0_u8; 58];
    bytes[0..2].copy_from_slice(b"BM");
    bytes[2..6].copy_from_slice(&58_u32.to_le_bytes());
    bytes[10..14].copy_from_slice(&54_u32.to_le_bytes());
    bytes[14..18].copy_from_slice(&40_u32.to_le_bytes());
    bytes[18..22].copy_from_slice(&1_i32.to_le_bytes());
    bytes[22..26].copy_from_slice(&1_i32.to_le_bytes());
    bytes[26..28].copy_from_slice(&1_u16.to_le_bytes());
    bytes[28..30].copy_from_slice(&24_u16.to_le_bytes());
    bytes[34..38].copy_from_slice(&4_u32.to_le_bytes());
    bytes[56] = 0xff;
    bytes
}

#[tokio::test]
async fn read_supports_offsets_truncation_and_image_magic() {
    let directory = tempdir().unwrap();
    let text_path = directory.path().join("large.txt");
    let text = (1..=DEFAULT_MAX_LINES + 1)
        .map(|line| format!("line {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&text_path, text).unwrap();
    let tools = Tools::local(directory.path());

    let result = tools.read(ReadInput::new("large.txt")).await.unwrap();
    assert!(result.text_content().contains("line 2000"));
    assert!(!result.text_content().contains("line 2001"));
    assert!(result.text_content().contains("Use offset=2001"));
    assert!(result.details.unwrap().truncation.unwrap().truncated);

    let result = tools
        .read(ReadInput {
            path: "large.txt".into(),
            offset: Some(2001),
            limit: Some(1),
        })
        .await
        .unwrap();
    assert_eq!(result.text_content(), "line 2001");

    let png_path = directory.path().join("image.data");
    fs::write(&png_path, b"\x89PNG\r\n\x1a\npayload").unwrap();
    let result = tools.read(ReadInput::new("image.data")).await.unwrap();
    assert!(matches!(
        &result.content[1],
        Content::Image { mime_type, .. } if mime_type == "image/png"
    ));

    fs::write(directory.path().join("image.bmp"), tiny_bmp()).unwrap();
    let result = tools.read(ReadInput::new("image.bmp")).await.unwrap();
    assert!(result.text_content().contains("converted from image/bmp"));
    assert!(matches!(
        &result.content[1],
        Content::Image { mime_type, data } if mime_type == "image/png" && data.starts_with("iVBOR")
    ));

    image::RgbImage::from_pixel(2_001, 1, image::Rgb([1, 2, 3]))
        .save(directory.path().join("wide.png"))
        .unwrap();
    let result = tools.read(ReadInput::new("wide.png")).await.unwrap();
    assert!(
        result
            .text_content()
            .contains("resized from 2001x1 to 2000x1")
    );
    assert!(matches!(
        &result.content[1],
        Content::Image { mime_type, data } if mime_type == "image/png" && data.starts_with("iVBOR")
    ));
}

#[tokio::test]
async fn write_creates_parents_and_reports_utf8_bytes() {
    let directory = tempdir().unwrap();
    let tools = Tools::local(directory.path());
    let result = tools
        .write(WriteInput {
            path: "nested/file.txt".into(),
            content: "é".to_owned(),
        })
        .await
        .unwrap();
    assert_eq!(
        fs::read_to_string(directory.path().join("nested/file.txt")).unwrap(),
        "é"
    );
    assert_eq!(result.details.unwrap().bytes_written, 2);
}

#[tokio::test]
async fn edit_preserves_bom_crlf_and_supports_fuzzy_multi_edit() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("edit.txt");
    fs::write(
        &path,
        "\u{feff}keep  \r\nconsole.log(\u{2018}hello\u{2019});\r\nhello\u{00a0}world\r\n",
    )
    .unwrap();
    let tools = Tools::local(directory.path());
    let result = tools
        .edit(EditInput {
            path: "edit.txt".into(),
            edits: vec![
                Edit {
                    old_text: "console.log('hello');\n".to_owned(),
                    new_text: "console.log('world');\n".to_owned(),
                },
                Edit {
                    old_text: "hello world\n".to_owned(),
                    new_text: "hello universe\n".to_owned(),
                },
            ],
        })
        .await
        .unwrap();

    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        "\u{feff}keep  \r\nconsole.log('world');\r\nhello universe\r\n"
    );
    let details = result.details.unwrap();
    assert!(details.patch.starts_with("--- edit.txt\n+++ edit.txt\n"));
    assert!(details.patch.contains("+console.log('world');"));
}

#[tokio::test]
async fn parallel_edits_of_one_canonical_file_are_serialized() {
    let directory = tempdir().unwrap();
    fs::write(directory.path().join("shared.txt"), "alpha\nbeta\n").unwrap();
    let tools = Tools::local(directory.path());
    let left = {
        let tools = tools.clone();
        tokio::spawn(async move {
            tools
                .edit(EditInput {
                    path: "shared.txt".into(),
                    edits: vec![Edit {
                        old_text: "alpha".to_owned(),
                        new_text: "ALPHA".to_owned(),
                    }],
                })
                .await
        })
    };
    let right = {
        let tools = tools.clone();
        tokio::spawn(async move {
            tools
                .edit(EditInput {
                    path: "shared.txt".into(),
                    edits: vec![Edit {
                        old_text: "beta".to_owned(),
                        new_text: "BETA".to_owned(),
                    }],
                })
                .await
        })
    };
    left.await.unwrap().unwrap();
    right.await.unwrap().unwrap();
    assert_eq!(
        fs::read_to_string(directory.path().join("shared.txt")).unwrap(),
        "ALPHA\nBETA\n"
    );
}

#[tokio::test]
async fn grep_supports_context_limits_gitignore_and_line_caps() {
    let directory = tempdir().unwrap();
    fs::write(directory.path().join(".gitignore"), "ignored.txt\n").unwrap();
    fs::write(directory.path().join("ignored.txt"), "match hidden\n").unwrap();
    fs::write(
        directory.path().join("kept.txt"),
        format!("before\nmatch {}\nafter\nmatch two\n", "x".repeat(600)),
    )
    .unwrap();
    let tools = Tools::local(directory.path());
    let result = tools
        .grep(GrepInput {
            pattern: "match".to_owned(),
            path: None,
            glob: Some("*.txt".to_owned()),
            ignore_case: false,
            literal: false,
            context: 1,
            limit: Some(1),
        })
        .await
        .unwrap();
    let output = result.text_content();
    assert!(output.contains("kept.txt-1- before"), "{output}");
    assert!(output.contains("kept.txt:2: match"), "{output}");
    assert!(!output.contains("ignored.txt"));
    assert!(output.contains("matches limit reached"));
    assert!(output.contains(&"x".repeat(GREP_MAX_LINE_LENGTH - "match ".len())));
    assert!(result.details.unwrap().lines_truncated);
}

#[tokio::test]
async fn find_honors_nested_ignore_rules_and_path_globs() {
    let directory = tempdir().unwrap();
    fs::create_dir_all(directory.path().join("a/deep")).unwrap();
    fs::create_dir_all(directory.path().join("b")).unwrap();
    fs::write(directory.path().join("a/.gitignore"), "ignored.txt\n").unwrap();
    fs::write(directory.path().join("a/deep/ignored.txt"), "").unwrap();
    fs::write(directory.path().join("a/deep/kept.spec.ts"), "").unwrap();
    fs::write(directory.path().join("b/ignored.txt"), "").unwrap();
    let tools = Tools::local(directory.path());

    let result = tools
        .find(FindInput {
            pattern: "a/**/*.spec.ts".to_owned(),
            path: None,
            limit: None,
        })
        .await
        .unwrap();
    assert!(result.text_content().contains("a/deep/kept.spec.ts"));

    let result = tools
        .find(FindInput {
            pattern: "**/*.txt".to_owned(),
            path: None,
            limit: None,
        })
        .await
        .unwrap();
    assert!(!result.text_content().contains("a/deep/ignored.txt"));
    assert!(result.text_content().contains("b/ignored.txt"));
}

#[tokio::test]
async fn ls_includes_dotfiles_sorts_and_marks_directories() {
    let directory = tempdir().unwrap();
    fs::write(directory.path().join(".hidden"), "").unwrap();
    fs::create_dir(directory.path().join("Alpha")).unwrap();
    fs::write(directory.path().join("beta"), "").unwrap();
    let tools = Tools::local(directory.path());
    let result = tools.ls(LsInput::default()).await.unwrap();
    assert_eq!(result.text_content(), ".hidden\nAlpha/\nbeta");
}

#[tokio::test]
async fn bash_streams_output_and_enforces_timeout_and_cancellation() {
    let directory = tempdir().unwrap();
    let tools = Tools::local(directory.path());
    let updates = Arc::new(Mutex::new(Vec::new()));
    let callback = {
        let updates = Arc::clone(&updates);
        Arc::new(move |result: ri_tools::BashResult| {
            updates.lock().unwrap().push(result.text_content());
        })
    };
    let result = tools
        .bash_with_cancellation(
            BashInput::new("printf 'one\\ntwo\\n'"),
            &CancellationToken::new(),
            Some(callback),
        )
        .await
        .unwrap();
    assert!(result.text_content().ends_with("one\ntwo\n"));
    assert!(!updates.lock().unwrap().is_empty());

    let error = tools
        .bash(BashInput {
            command: "sleep 5".to_owned(),
            timeout: Some(0.05),
        })
        .await
        .unwrap_err();
    assert!(matches!(error, ToolError::CommandTimedOut { .. }));

    let cancellation = CancellationToken::new();
    let child = cancellation.clone();
    let tools_for_task = tools.clone();
    let task = tokio::spawn(async move {
        tools_for_task
            .bash_with_cancellation(BashInput::new("sleep 5"), &child, None)
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancellation.cancel();
    assert!(matches!(
        task.await.unwrap().unwrap_err(),
        ToolError::CommandCancelled { .. }
    ));
}

#[tokio::test]
async fn pre_cancelled_filesystem_operation_does_not_mutate() {
    let directory = tempdir().unwrap();
    let tools = Tools::local(directory.path());
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let result = tools
        .write_with_cancellation(
            WriteInput {
                path: "cancelled.txt".into(),
                content: "no".to_owned(),
            },
            &cancellation,
        )
        .await;
    assert!(result.is_err());
    assert!(!directory.path().join("cancelled.txt").exists());
}
