//! End-to-end tests against a scripted stand-in for the `claude` binary.
//!
//! These exercise the parts a unit test cannot reach: process spawning, the
//! argument vector as the operating system actually receives it, the child's
//! environment, exit-status handling, and the agent runtime driving the model.
//!
//! The fake is a shell script, so the suite is Unix-only. The library itself
//! is not.

#![cfg(unix)]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod common;

use common::FakeClaude;
use rig::completion::{Chat, Message, Prompt};
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, ClaudeCodeModel};
use rig_core::OneOrMany;
use rig_core::completion::{CompletionModel, CompletionRequest};

/// A minimal envelope carrying `text`.
fn envelope(text: &str) -> String {
    format!(
        r#"{{"is_error":false,"subtype":"success","result":"{text}","session_id":"s-42",
            "usage":{{"input_tokens":11,"output_tokens":5,"cache_read_input_tokens":1,
                      "cache_creation_input_tokens":2,
                      "output_tokens_details":{{"thinking_tokens":3}}}}}}"#
    )
}

fn request(prompt: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        preamble: None,
        chat_history: OneOrMany::one(Message::user(prompt)),
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

#[tokio::test]
async fn returns_the_assistant_text() {
    let fake = FakeClaude::printing(&envelope("pong"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let response = model.completion(request("ping")).await.unwrap();

    match response.choice.first() {
        rig_core::completion::AssistantContent::Text(text) => assert_eq!(text.text, "pong"),
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn reports_token_usage() {
    let fake = FakeClaude::printing(&envelope("pong"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let response = model.completion(request("ping")).await.unwrap();

    assert_eq!(response.usage.input_tokens, 11);
    assert_eq!(response.usage.output_tokens, 5);
    assert_eq!(response.usage.total_tokens, 16);
    assert_eq!(response.usage.cached_input_tokens, 1);
    assert_eq!(response.usage.cache_creation_input_tokens, 2);
    assert_eq!(response.usage.reasoning_tokens, 3);
    assert_eq!(response.message_id.as_deref(), Some("s-42"));
}

#[tokio::test]
async fn passes_the_prompt_as_the_final_argument() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    model.completion(request("the prompt")).await.unwrap();

    assert_eq!(fake.argv().last().map(String::as_str), Some("the prompt"));
}

#[tokio::test]
async fn passes_the_lean_flags() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let model = ClaudeCodeModel::new("sonnet").with_binary(fake.path());

    model.completion(request("hi")).await.unwrap();

    let argv = fake.argv();
    assert!(argv.contains(&"-p".to_owned()), "{argv:?}");
    assert_eq!(fake.value_after("--output-format").as_deref(), Some("json"));
    assert_eq!(fake.value_after("--model").as_deref(), Some("sonnet"));
    assert_eq!(fake.value_after("--tools").as_deref(), Some(""));
    assert_eq!(fake.value_after("--setting-sources").as_deref(), Some(""));
    assert!(argv.contains(&"--strict-mcp-config".to_owned()), "{argv:?}");
    assert!(
        argv.contains(&"--disable-slash-commands".to_owned()),
        "{argv:?}"
    );
    assert!(
        !argv.contains(&"--bare".to_owned()),
        "--bare would force API-key auth"
    );
}

#[tokio::test]
async fn a_flag_like_prompt_reaches_the_child_as_the_prompt() {
    // Verified against Claude Code 2.1.233: without the `--` separator,
    // `claude -p '--version'` prints its version and never answers the
    // prompt. Prompt text is attacker-influenceable in most deployments, so
    // a prompt of `--dangerously-skip-permissions` must not become a flag.
    let fake = FakeClaude::printing(&envelope("ok"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    model
        .completion(request("--dangerously-skip-permissions"))
        .await
        .unwrap();

    let argv = fake.argv();
    let separator = argv
        .iter()
        .position(|arg| arg == "--")
        .expect("an argument separator must precede the prompt");
    assert_eq!(
        argv.get(separator + 1).map(String::as_str),
        Some("--dangerously-skip-permissions")
    );
    assert_eq!(separator + 2, argv.len(), "nothing may follow the prompt");
}

#[tokio::test]
async fn preserves_an_empty_argument_through_the_process_boundary() {
    // `--tools ""` only strips the built-in tools if the empty string survives
    // as its own argument. A shell-quoting mistake would silently drop it and
    // leave every built-in tool enabled.
    let fake = FakeClaude::printing(&envelope("ok"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    model.completion(request("hi")).await.unwrap();

    let argv = fake.argv();
    let tools = argv.iter().position(|arg| arg == "--tools").unwrap();
    assert_eq!(argv.get(tools + 1).map(String::as_str), Some(""));
    assert_eq!(
        argv.get(tools + 2).map(String::as_str),
        Some("--strict-mcp-config"),
        "the empty value must not collapse into the next flag"
    );
}

#[tokio::test]
async fn removes_the_nested_session_marker() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    // Deliberate: this mutates process-wide state. The assertion is that the
    // marker does not reach the child even when this process has it set. The
    // library forbids unsafe; the test harness does not.
    let _guard = ENV_LOCK.lock().await;
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("CLAUDECODE", "1");
    }
    let result = model.completion(request("hi")).await;
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("CLAUDECODE");
    }
    result.unwrap();

    assert!(
        !fake.saw_nested_marker(),
        "CLAUDECODE must be stripped from the child environment"
    );
}

#[tokio::test]
async fn passes_a_preamble_as_the_system_prompt() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let mut req = request("hi");
    req.preamble = Some("Be terse.".to_owned());
    model.completion(req).await.unwrap();

    assert_eq!(
        fake.value_after("--system-prompt").as_deref(),
        Some("Be terse.")
    );
}

#[tokio::test]
async fn surfaces_a_non_zero_exit_with_its_stderr() {
    let fake = FakeClaude::failing("credit balance too low", 3);
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let error = model
        .completion(request("hi"))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("credit balance too low"), "{error}");
    assert!(error.contains("exited with"), "{error}");
}

#[tokio::test]
async fn surfaces_a_failed_turn_envelope() {
    let fake = FakeClaude::printing(
        r#"{"is_error":true,"subtype":"error_during_execution","result":"tool loop"}"#,
    );
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let error = model
        .completion(request("hi"))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("error_during_execution"), "{error}");
    assert!(error.contains("tool loop"), "{error}");
}

#[tokio::test]
async fn surfaces_unparseable_output() {
    let fake = FakeClaude::printing("this is not json");
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let error = model
        .completion(request("hi"))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("unparseable claude output"), "{error}");
    assert!(error.contains("this is not json"), "{error}");
}

#[tokio::test]
async fn surfaces_a_missing_binary() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let model =
        ClaudeCodeModel::new("haiku").with_binary(fake.missing_path().display().to_string());

    let error = model
        .completion(request("hi"))
        .await
        .unwrap_err()
        .to_string();

    assert!(error.contains("cannot run"), "{error}");
}

#[tokio::test]
async fn from_val_builds_a_working_client() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let client = ClaudeCodeClient::from_val(fake.path()).unwrap();

    let response = client
        .completion_model("haiku")
        .completion(request("hi"))
        .await
        .unwrap();

    match response.choice.first() {
        rig_core::completion::AssistantContent::Text(text) => assert_eq!(text.text, "ok"),
        other => panic!("expected text, got {other:?}"),
    }
}

/// Serializes the tests that mutate process-wide environment variables.
///
/// `set_var` is unsafe in edition 2024 precisely because other threads may be
/// reading the environment. The test binary is multi-threaded, so these tests
/// take a lock rather than trusting that they happen not to overlap. The lock
/// is async-aware because it is held across the awaited child process, which
/// is the whole window during which the variable must stay set.
static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
async fn from_env_resolves_the_binary_from_the_environment() {
    let fake = FakeClaude::printing("2.1.233 (Claude Code)\n");
    let path = fake.path();

    let client = {
        let _guard = ENV_LOCK.lock().await;
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(rig_claude_code::BINARY_ENV, &path);
        }
        let client = ClaudeCodeClient::from_env();
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var(rig_claude_code::BINARY_ENV);
        }
        client
    };

    assert_eq!(client.unwrap().binary(), path);
}

#[tokio::test]
async fn from_env_rejects_a_binary_it_cannot_run() {
    let fake = FakeClaude::printing("");
    let missing = fake.missing_path().display().to_string();

    let result = {
        let _guard = ENV_LOCK.lock().await;
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(rig_claude_code::BINARY_ENV, &missing);
        }
        let result = ClaudeCodeClient::from_env();
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var(rig_claude_code::BINARY_ENV);
        }
        result
    };

    let error = result.unwrap_err().to_string();
    assert!(error.contains(&missing), "{error}");
    assert!(error.contains("cannot run the claude binary"), "{error}");
}

#[tokio::test]
async fn reports_the_binary_version() {
    let fake = FakeClaude::printing("2.1.233 (Claude Code)\n");
    let client = ClaudeCodeClient::new(fake.path());

    assert_eq!(client.version().await.unwrap(), "2.1.233 (Claude Code)");
}

#[tokio::test]
async fn version_reports_an_unrunnable_binary() {
    let fake = FakeClaude::printing("");
    let client = ClaudeCodeClient::new(fake.missing_path().display().to_string());

    let error = client.version().await.unwrap_err().to_string();

    assert!(error.contains("cannot run the claude binary"), "{error}");
}

#[tokio::test]
async fn drives_an_agent_end_to_end() {
    let fake = FakeClaude::printing(&envelope("a forecast"));
    let client = ClaudeCodeClient::new(fake.path());

    let agent = client.agent("haiku").preamble("Be terse.").build();
    let answer = agent.prompt("Report on Dogger.").await.unwrap();

    assert_eq!(answer, "a forecast");
    assert_eq!(
        fake.value_after("--system-prompt").as_deref(),
        Some("Be terse.")
    );
}

#[tokio::test]
async fn threads_history_through_the_agent() {
    let fake = FakeClaude::printing(&envelope("second answer"));
    let client = ClaudeCodeClient::new(fake.path());
    let agent = client.agent("haiku").build();

    let mut history = vec![
        Message::user("first question"),
        Message::assistant("first answer"),
    ];
    let answer = agent.chat("second question", &mut history).await.unwrap();

    assert_eq!(answer, "second answer");

    let prompt = fake.argv().last().cloned().unwrap();
    assert!(prompt.contains("user: first question"), "{prompt}");
    assert!(prompt.contains("assistant: first answer"), "{prompt}");
    assert!(prompt.ends_with("second question"), "{prompt}");

    assert_eq!(
        history.len(),
        4,
        "chat appends the committed user and assistant turns"
    );
}

#[tokio::test]
async fn an_agent_with_context_documents_sends_them() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let client = ClaudeCodeClient::new(fake.path());

    let agent = client
        .agent("haiku")
        .context("a flurbo is a green alien")
        .build();
    agent.prompt("what is a flurbo?").await.unwrap();

    let prompt = fake.argv().last().cloned().unwrap();
    assert!(prompt.contains("a flurbo is a green alien"), "{prompt}");
}

#[tokio::test]
async fn an_agent_with_a_tool_fails_with_a_usable_message() {
    let fake = FakeClaude::printing(&envelope("ok"));
    let client = ClaudeCodeClient::new(fake.path());

    let agent = client.agent("haiku").tool(tools::Add).build();
    let error = agent.prompt("add 2 and 2").await.unwrap_err().to_string();

    assert!(error.contains("tool definitions"), "{error}");
    assert!(error.contains("--mcp-config"), "{error}");
}

// --- streaming -----------------------------------------------------------

/// A frame stream that produces `text` in two deltas, after one thinking
/// delta and the frames a real CLI interleaves.
fn frame_stream(text_a: &str, text_b: &str) -> String {
    format!(
        r#"{{"type":"system","subtype":"init","session_id":"s-7"}}
{{"type":"stream_event","event":{{"type":"message_start","message":{{}}}}}}
{{"type":"stream_event","event":{{"type":"content_block_delta","index":0,"delta":{{"type":"thinking_delta","thinking":"pondering"}}}}}}
{{"type":"rate_limit_event","rate_limit_info":{{"status":"allowed"}}}}
{{"type":"stream_event","event":{{"type":"content_block_start","index":1,"content_block":{{"type":"text","text":""}}}}}}
{{"type":"stream_event","event":{{"type":"content_block_delta","index":1,"delta":{{"type":"text_delta","text":"{text_a}"}}}}}}
{{"type":"stream_event","event":{{"type":"content_block_delta","index":1,"delta":{{"type":"text_delta","text":"{text_b}"}}}}}}
{{"type":"stream_event","event":{{"type":"message_stop"}}}}
{{"type":"result","is_error":false,"subtype":"success","result":"{text_a}{text_b}","session_id":"s-7","usage":{{"input_tokens":8,"output_tokens":4,"output_tokens_details":{{"thinking_tokens":2}}}}}}
"#
    )
}

/// Collect a stream's text, reasoning, and terminal error, if any.
async fn drain(
    mut stream: rig_core::streaming::StreamingCompletionResponse<rig_claude_code::CliResult>,
) -> (String, String, Option<String>) {
    use futures::StreamExt as _;
    use rig_core::streaming::StreamedAssistantContent;

    let (mut text, mut reasoning, mut failure) = (String::new(), String::new(), None);
    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamedAssistantContent::Text(chunk)) => text.push_str(&chunk.text),
            Ok(StreamedAssistantContent::ReasoningDelta { reasoning: r, .. }) => {
                reasoning.push_str(&r);
            }
            Ok(_) => {}
            Err(error) => {
                failure = Some(error.to_string());
                break;
            }
        }
    }
    (text, reasoning, failure)
}

#[tokio::test]
async fn streams_text_deltas_in_order() {
    let fake = FakeClaude::printing(&frame_stream("1 2 ", "3 4 5"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let stream = model.stream(request("count")).await.unwrap();
    let (text, _, failure) = drain(stream).await;

    assert_eq!(failure, None);
    assert_eq!(text, "1 2 3 4 5");
}

#[tokio::test]
async fn streams_thinking_as_reasoning() {
    let fake = FakeClaude::printing(&frame_stream("a", "b"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let stream = model.stream(request("count")).await.unwrap();
    let (_, reasoning, failure) = drain(stream).await;

    assert_eq!(failure, None);
    assert_eq!(reasoning, "pondering");
}

#[tokio::test]
async fn a_stream_asks_for_frames_and_partials() {
    let fake = FakeClaude::printing(&frame_stream("a", "b"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let stream = model.stream(request("count")).await.unwrap();
    drain(stream).await;

    assert_eq!(
        fake.value_after("--output-format").as_deref(),
        Some("stream-json")
    );
    let argv = fake.argv();
    assert!(
        argv.contains(&"--include-partial-messages".to_owned()),
        "{argv:?}"
    );
    assert!(argv.contains(&"--verbose".to_owned()), "{argv:?}");
}

#[tokio::test]
async fn a_stream_carries_usage_on_its_final_response() {
    use futures::StreamExt as _;

    let fake = FakeClaude::printing(&frame_stream("a", "b"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let mut stream = model.stream(request("count")).await.unwrap();
    while stream.next().await.is_some() {}

    let final_response = stream.response.as_ref().expect("a final response");
    assert_eq!(final_response.usage.input_tokens, 8);
    assert_eq!(final_response.result.as_deref(), Some("ab"));
    assert_eq!(stream.message_id.as_deref(), Some("s-7"));
}

#[tokio::test]
async fn a_stream_surfaces_a_failed_turn() {
    let fake = FakeClaude::printing(
        "{\"type\":\"result\",\"is_error\":true,\"subtype\":\"error_max_turns\",\"result\":\"gave up\"}\n",
    );
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let stream = model.stream(request("hi")).await.unwrap();
    let (_, _, failure) = drain(stream).await;

    let failure = failure.expect("a failed turn should surface");
    assert!(failure.contains("error_max_turns"), "{failure}");
}

#[tokio::test]
async fn a_stream_that_ends_without_a_terminal_frame_fails() {
    let fake = FakeClaude::printing(
        "{\"type\":\"stream_event\",\"event\":{\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}}\n",
    );
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let stream = model.stream(request("hi")).await.unwrap();
    let (text, _, failure) = drain(stream).await;

    assert_eq!(text, "partial", "deltas before the truncation still arrive");
    let failure = failure.expect("a truncated stream should fail");
    assert!(
        failure.contains("without a terminal result frame"),
        "{failure}"
    );
}

#[tokio::test]
async fn a_stream_surfaces_a_non_zero_exit_with_its_stderr() {
    let fake = FakeClaude::failing("session limit reached", 1);
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let stream = model.stream(request("hi")).await.unwrap();
    let (_, _, failure) = drain(stream).await;

    let failure = failure.expect("a non-zero exit should surface");
    assert!(failure.contains("session limit reached"), "{failure}");
    assert!(failure.contains("exited with"), "{failure}");
}

#[tokio::test]
async fn a_stream_ignores_noise_on_stdout() {
    let mut frames = String::from("this line is not json\n");
    frames.push_str(&frame_stream("ok", ""));
    let fake = FakeClaude::printing(&frames);
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let stream = model.stream(request("hi")).await.unwrap();
    let (text, _, failure) = drain(stream).await;

    assert_eq!(failure, None);
    assert_eq!(text, "ok");
}

#[tokio::test]
async fn a_stream_removes_the_nested_session_marker() {
    let fake = FakeClaude::printing(&frame_stream("a", "b"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    // Deliberate: the assertion is that the marker does not reach the child
    // even when this process has it set. The library forbids unsafe; the test
    // harness does not.
    let _guard = ENV_LOCK.lock().await;
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("CLAUDECODE", "1");
    }
    let stream = model.stream(request("hi")).await.unwrap();
    drain(stream).await;
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("CLAUDECODE");
    }

    assert!(!fake.saw_nested_marker());
}

// --- structured output ---------------------------------------------------

#[tokio::test]
async fn an_output_schema_reaches_the_json_schema_flag() {
    let fake = FakeClaude::printing(&envelope("{}"));
    let model = ClaudeCodeModel::new("haiku").with_binary(fake.path());

    let mut req = request("describe a person");
    req.output_schema = Some(schemars::schema_for!(Person));
    model.completion(req).await.unwrap();

    let sent = fake.value_after("--json-schema").expect("--json-schema");
    let parsed: serde_json::Value = serde_json::from_str(&sent).unwrap();
    assert!(
        parsed
            .get("properties")
            .and_then(|p| p.get("name"))
            .is_some(),
        "{parsed}"
    );
}

#[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
struct Person {
    name: String,
    age: u8,
}

mod tools {
    use rig::tool::{Tool, ToolContext};
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct AddArgs {
        pub left: i64,
        pub right: i64,
    }

    #[derive(Debug, thiserror::Error)]
    #[error("addition failed")]
    pub struct AddError;

    pub struct Add;

    impl Tool for Add {
        const NAME: &'static str = "add";
        type Args = AddArgs;
        type Output = i64;
        type Error = AddError;

        fn description(&self) -> String {
            "Add two integers".to_owned()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "left": { "type": "integer" },
                    "right": { "type": "integer" }
                },
                "required": ["left", "right"]
            })
        }

        async fn call(
            &self,
            _context: &mut ToolContext,
            args: Self::Args,
        ) -> Result<Self::Output, Self::Error> {
            Ok(args.left + args.right)
        }
    }
}
