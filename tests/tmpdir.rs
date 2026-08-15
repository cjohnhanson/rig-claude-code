//! The one test that mutates `TMPDIR`.
//!
//! It lives alone because `tempfile` reads `TMPDIR` to place every temporary
//! directory, including the ones other tests build their fixtures in. Pointing
//! it at a path that does not exist breaks any test that starts while it is
//! set, so cargo's one-binary-per-file rule is what keeps it contained.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::FakeClaude;
use rig_claude_code::ClaudeCodeModel;
use rig_core::OneOrMany;
use rig_core::completion::{CompletionModel, CompletionRequest, Message};

#[tokio::test]
async fn reports_a_system_prompt_file_it_cannot_create() {
    // The system prompt travels in a private file rather than argv. When that
    // file cannot be created, the turn must say so plainly rather than fail
    // later with something about the CLI.
    let fake = FakeClaude::printing(r#"{"is_error":false,"result":"ok"}"#);
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let request = CompletionRequest {
        model: None,
        preamble: Some("Be terse.".to_owned()),
        chat_history: OneOrMany::one(Message::user("hi")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("TMPDIR", "/nonexistent/rig-claude-code");
    }
    let result = model.completion(request).await;
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("TMPDIR");
    }

    let error = result.unwrap_err().to_string();
    assert!(error.contains("file for the system prompt"), "{error}");
}
