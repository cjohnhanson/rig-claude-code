//! Tests that mutate process-wide environment variables.
//!
//! These live in their own file on purpose. Cargo compiles each integration
//! file into its own binary, so nothing else runs in this process while a
//! variable is being set. `set_var` is unsafe in edition 2024 precisely
//! because another thread may be reading the environment, and spawning a child
//! reads the whole environment table — so a mutex shared with the main suite
//! would order the mutating tests against each other while doing nothing about
//! the thirty other tests spawning children beside them.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::FakeClaude;
use rig::prelude::*;
use rig_claude_code::{BINARY_ENV, ClaudeCodeClient, ClaudeCodeModel, DEFAULT_BINARY};
use rig_core::OneOrMany;
use rig_core::completion::{CompletionModel, CompletionRequest, Message};

fn envelope() -> &'static str {
    r#"{"is_error":false,"subtype":"success","result":"ok"}"#
}

fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: OneOrMany::one(Message::user("hi")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Serializes every test in this file.
///
/// Cargo's one-binary-per-file rule keeps these tests away from the rest of
/// the suite, but the harness still runs the tests *within* a binary in
/// parallel, so without this they clobber each other's variables.
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Set a variable, run `body`, and restore whatever was there before.
///
/// Restoring matters: a developer with a real `RIG_CLAUDE_CODE_BIN` set should
/// not have it clobbered for the rest of the process.
fn with_var<T>(name: &str, value: Option<&str>, body: impl FnOnce() -> T) -> T {
    let previous = std::env::var(name).ok();

    #[allow(unsafe_code)]
    unsafe {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }

    let outcome = body();

    #[allow(unsafe_code)]
    unsafe {
        match previous {
            Some(previous) => std::env::set_var(name, previous),
            None => std::env::remove_var(name),
        }
    }

    outcome
}

#[tokio::test]
async fn from_env_resolves_the_binary_from_the_environment() {
    let _guard = ENV_LOCK.lock().await;
    let fake = FakeClaude::printing("2.1.233 (Claude Code)\n");
    let path = fake.path();

    let client = with_var(BINARY_ENV, Some(&path), ClaudeCodeClient::from_env);

    assert_eq!(client.unwrap().binary(), path);
}

#[tokio::test]
async fn from_env_falls_back_to_the_binary_name_on_path() {
    let _guard = ENV_LOCK.lock().await;
    // The documented default, and the branch every user without the variable
    // set takes. Whether a real `claude` is installed is not this test's
    // business: either it resolves and the client names the default, or it
    // does not and the error names the default.
    let outcome = with_var(BINARY_ENV, None, ClaudeCodeClient::from_env);

    match outcome {
        Ok(client) => assert_eq!(client.binary(), DEFAULT_BINARY),
        Err(error) => assert!(error.to_string().contains(DEFAULT_BINARY), "{error}"),
    }
}

#[tokio::test]
async fn from_env_rejects_a_binary_it_cannot_run() {
    let _guard = ENV_LOCK.lock().await;
    let fake = FakeClaude::printing("");
    let missing = fake.missing_path().display().to_string();

    let error = with_var(BINARY_ENV, Some(&missing), ClaudeCodeClient::from_env).unwrap_err();

    let rendered = error.to_string();
    assert!(rendered.contains(&missing), "{rendered}");
    assert!(
        rendered.contains("cannot run the claude binary"),
        "{rendered}"
    );
}

#[tokio::test]
async fn from_env_rejects_a_binary_that_runs_and_fails() {
    let _guard = ENV_LOCK.lock().await;
    // Any executable file used to pass this check, so `/bin/false` yielded a
    // happy client that failed one prompt later — the outcome the method
    // exists to prevent.
    let fake = FakeClaude::failing("not claude", 1);
    let path = fake.path();

    let error = with_var(BINARY_ENV, Some(&path), ClaudeCodeClient::from_env).unwrap_err();

    let rendered = error.to_string();
    assert!(rendered.contains("exited with"), "{rendered}");
    assert!(rendered.contains("not claude"), "{rendered}");
}

#[tokio::test]
async fn removes_the_nested_session_marker_from_a_turn() {
    let _guard = ENV_LOCK.lock().await;
    let fake = FakeClaude::printing(envelope());
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    // The variable must stay set across the spawn, so `with_var` cannot wrap
    // an await here; set it, run the turn, restore it.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("CLAUDECODE", "1");
    }
    let result = model.completion(request()).await;
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("CLAUDECODE");
    }
    result.unwrap();

    assert!(
        !fake.saw_nested_marker(),
        "a claude launched from inside a Claude Code session would treat \
         itself as nested"
    );
}

#[tokio::test]
async fn removes_the_nested_session_marker_from_a_stream() {
    use futures::StreamExt as _;

    let _guard = ENV_LOCK.lock().await;
    let fake = FakeClaude::printing("{\"type\":\"result\",\"result\":\"ok\"}\n");
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("CLAUDECODE", "1");
    }
    let mut stream = model.stream(request()).await.unwrap();
    while stream.next().await.is_some() {}
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("CLAUDECODE");
    }

    assert!(!fake.saw_nested_marker());
}
