//! Translation of a rig [`CompletionRequest`] into a `claude` command line.
//!
//! This module performs no IO. Building the command is a pure function so the
//! exact argument vector can be asserted in tests without running anything.

use rig_core::OneOrMany;
use rig_core::completion::{CompletionError, CompletionRequest};
use rig_core::message::{AssistantContent, Message, UserContent};

/// A `claude` invocation, ready for the caller to spawn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CommandSpec {
    /// The binary to run.
    pub(crate) program: String,
    /// Arguments, in the order they are passed.
    pub(crate) args: Vec<String>,
}

/// Flags that strip the invocation down to a bare model call.
///
/// Without these the CLI loads its full agent system prompt, the project's
/// `CLAUDE.md`, configured MCP servers, and skills. Measured against Claude
/// Code 2.1.233, a one-word prompt costs about 42,000 input tokens with the
/// defaults and about 165 with these flags.
///
/// `--bare` looks like it belongs here and does not: it forces authentication
/// through `ANTHROPIC_API_KEY` and never reads the subscription credential,
/// which defeats the purpose of this crate.
const LEAN_FLAGS: &[&str] = &[
    "--tools",
    "",
    "--strict-mcp-config",
    "--setting-sources",
    "",
    "--disable-slash-commands",
];

/// Which output format the invocation should ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// One JSON envelope on stdout when the turn finishes.
    Blocking,
    /// Newline-delimited JSON frames as the turn proceeds.
    Streaming,
}

impl Mode {
    /// The output-format flags this mode needs.
    ///
    /// `--verbose` is not decoration in streaming mode: without it the CLI
    /// declines to emit the frame stream in print mode.
    fn flags(self) -> &'static [&'static str] {
        match self {
            Self::Blocking => &["--output-format", "json"],
            Self::Streaming => &[
                "--output-format",
                "stream-json",
                "--include-partial-messages",
                "--verbose",
            ],
        }
    }
}

/// Build the command for `request`.
///
/// `default_model` is used unless the request overrides it.
///
/// # Errors
///
/// Returns [`CompletionError::RequestError`] when the request asks for
/// something the CLI cannot express. Each such setting fails loudly rather
/// than being dropped, because a silently ignored `max_tokens` or tool
/// definition is far harder to diagnose than a rejected request.
pub(crate) fn build(
    binary: &str,
    default_model: &str,
    request: &CompletionRequest,
    mode: Mode,
) -> Result<CommandSpec, CompletionError> {
    reject_unsupported(request)?;

    let model = request
        .model
        .clone()
        .unwrap_or_else(|| default_model.to_owned());

    let mut args = vec!["-p".to_owned()];
    args.extend(mode.flags().iter().map(|flag| (*flag).to_owned()));
    args.push("--model".to_owned());
    args.push(model);
    args.extend(LEAN_FLAGS.iter().map(|flag| (*flag).to_owned()));

    if let Some(schema) = &request.output_schema {
        args.push("--json-schema".to_owned());
        args.push(serde_json::to_string(schema)?);
    }

    if let Some(system) = render_system(request) {
        args.push("--system-prompt".to_owned());
        args.push(system);
    }

    // Everything after `--` is positional, whatever it looks like. Without
    // it, a prompt that begins with a dash is parsed as a flag: `claude -p
    // '--version'` prints the version and never answers, and a prompt of
    // `--dangerously-skip-permissions` would be *obeyed* rather than
    // answered. Prompt text is attacker-influenceable in most real
    // deployments, so this separator is a security boundary, not a nicety.
    args.push("--".to_owned());
    args.push(render_prompt(request));

    Ok(CommandSpec {
        program: binary.to_owned(),
        args,
    })
}

/// Reject request settings the CLI has no way to express.
fn reject_unsupported(request: &CompletionRequest) -> Result<(), CompletionError> {
    if !request.tools.is_empty() {
        return unsupported(
            "tool definitions",
            "the CLI accepts no tool definitions as arguments; expose the tools \
             over MCP and point the binary at them with --mcp-config",
        );
    }
    if request.temperature.is_some() {
        return unsupported("temperature", "the CLI has no temperature flag");
    }
    if request.max_tokens.is_some() {
        return unsupported("max_tokens", "the CLI has no output-token limit flag");
    }
    if request.additional_params.is_some() {
        return unsupported(
            "additional_params",
            "the CLI takes no provider-specific request body",
        );
    }
    Ok(())
}

/// Build the rejection for one unsupported setting.
fn unsupported<T>(setting: &str, reason: &str) -> Result<T, CompletionError> {
    Err(CompletionError::RequestError(
        format!("rig-claude-code cannot honor {setting}: {reason}").into(),
    ))
}

/// Assemble the system instructions.
///
/// Both sources are collected: the legacy `preamble` field, and any
/// [`Message::System`] in the history, which is where rig has put an agent's
/// preamble since 0.33.
fn render_system(request: &CompletionRequest) -> Option<String> {
    let mut parts: Vec<&str> = Vec::new();
    if let Some(preamble) = request.preamble.as_deref() {
        parts.push(preamble);
    }
    for message in request.chat_history.iter() {
        if let Message::System { content } = message {
            parts.push(content);
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("\n\n"))
    }
}

/// Flatten the request into the single prompt string the CLI accepts.
///
/// The CLI takes one prompt rather than a message list, so context documents
/// and prior turns are rendered as labelled sections ahead of the current
/// prompt. System messages are omitted here because [`render_system`] has
/// already carried them to `--system-prompt`.
fn render_prompt(request: &CompletionRequest) -> String {
    let mut out = String::new();

    if !request.documents.is_empty() {
        out.push_str("<context>\n");
        for document in &request.documents {
            out.push_str(&document.text);
            out.push('\n');
        }
        out.push_str("</context>\n\n");
    }

    // `OneOrMany` guarantees at least one message and the last one is the
    // prompt, so everything before it is prior context. The code still spells
    // that out with iterators rather than an index, so a future change to the
    // container cannot turn the assumption into a panic.
    let history: Vec<&Message> = request.chat_history.iter().collect();
    let earlier = history.len().saturating_sub(1);

    let transcript: Vec<String> = history
        .iter()
        .take(earlier)
        .filter_map(|message| transcript_line(message))
        .collect();
    if !transcript.is_empty() {
        out.push_str("<transcript>\n");
        for line in transcript {
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str("</transcript>\n\n");
    }

    if let Some(last) = history.last() {
        out.push_str(&message_text(last));
    }
    out
}

/// Render one prior turn as a labelled transcript line, or nothing for a
/// system message, which has already gone to `--system-prompt`.
fn transcript_line(message: &Message) -> Option<String> {
    match message {
        Message::System { .. } => None,
        Message::User { content } => Some(format!("user: {}", user_text(content))),
        Message::Assistant { content, .. } => {
            Some(format!("assistant: {}", assistant_text(content)))
        }
    }
}

/// The text of any message, whatever its role.
fn message_text(message: &Message) -> String {
    match message {
        Message::System { content } => content.clone(),
        Message::User { content } => user_text(content),
        Message::Assistant { content, .. } => assistant_text(content),
    }
}

/// Concatenate the text blocks of a user message.
///
/// Non-text content — images, audio, documents, tool results — is dropped,
/// because the CLI's prompt argument carries text only.
fn user_text(content: &OneOrMany<UserContent>) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Concatenate the text blocks of an assistant message.
fn assistant_text(content: &OneOrMany<AssistantContent>) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;
    use rig_core::completion::Document;

    /// A request carrying just `prompt`, with every optional setting unset.
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

    fn document(id: &str, text: &str) -> Document {
        Document {
            id: id.to_owned(),
            text: text.to_owned(),
            additional_props: std::collections::HashMap::new(),
        }
    }

    /// The value passed after `flag`, if the flag is present.
    fn value_after<'a>(spec: &'a CommandSpec, flag: &str) -> Option<&'a str> {
        let index = spec.args.iter().position(|arg| arg == flag)?;
        spec.args.get(index + 1).map(String::as_str)
    }

    #[test]
    fn runs_the_configured_binary() {
        let spec = build("/opt/claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert_eq!(spec.program, "/opt/claude");
    }

    #[test]
    fn asks_for_print_mode_and_json() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert!(spec.args.contains(&"-p".to_owned()));
        assert_eq!(value_after(&spec, "--output-format"), Some("json"));
    }

    #[test]
    fn uses_the_default_model() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert_eq!(value_after(&spec, "--model"), Some("haiku"));
    }

    #[test]
    fn lets_the_request_override_the_model() {
        let mut req = request("hi");
        req.model = Some("opus".to_owned());
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(value_after(&spec, "--model"), Some("opus"));
    }

    #[test]
    fn strips_the_invocation_down_to_the_model() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert_eq!(value_after(&spec, "--tools"), Some(""));
        assert_eq!(value_after(&spec, "--setting-sources"), Some(""));
        assert!(spec.args.contains(&"--strict-mcp-config".to_owned()));
        assert!(spec.args.contains(&"--disable-slash-commands".to_owned()));
    }

    #[test]
    fn never_passes_the_bare_flag() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert!(
            !spec.args.contains(&"--bare".to_owned()),
            "--bare forces API-key auth and would defeat subscription auth"
        );
    }

    #[test]
    fn puts_the_prompt_last() {
        let spec = build("claude", "haiku", &request("hello there"), Mode::Blocking).unwrap();
        assert_eq!(spec.args.last().map(String::as_str), Some("hello there"));
    }

    #[test]
    fn separates_the_prompt_from_the_flags() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        let separator = spec
            .args
            .iter()
            .position(|arg| arg == "--")
            .expect("the prompt must be introduced by an argument separator");
        assert_eq!(
            separator + 2,
            spec.args.len(),
            "the separator sits immediately before the prompt"
        );
    }

    #[test]
    fn a_prompt_that_looks_like_a_flag_stays_a_prompt() {
        // Verified against Claude Code 2.1.233: without the separator,
        // `claude -p '--version'` prints its version and never answers.
        for hostile in [
            "--version",
            "--dangerously-skip-permissions",
            "-p",
            "--settings /tmp/evil.json",
            "--help",
        ] {
            let spec = build("claude", "haiku", &request(hostile), Mode::Blocking).unwrap();
            assert_eq!(spec.args.last().map(String::as_str), Some(hostile));
            let separator = spec.args.iter().position(|arg| arg == "--").unwrap();
            assert_eq!(
                separator + 2,
                spec.args.len(),
                "`{hostile}` must sit after the separator, not be parsed as a flag"
            );
        }
    }

    #[test]
    fn streaming_mode_also_separates_the_prompt() {
        let spec = build("claude", "haiku", &request("--version"), Mode::Streaming).unwrap();
        let separator = spec.args.iter().position(|arg| arg == "--").unwrap();
        assert_eq!(separator + 2, spec.args.len());
    }

    #[test]
    fn omits_the_system_flag_when_there_is_nothing_to_say() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert!(!spec.args.contains(&"--system-prompt".to_owned()));
    }

    #[test]
    fn carries_the_preamble_to_the_system_flag() {
        let mut req = request("hi");
        req.preamble = Some("Be terse.".to_owned());
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(value_after(&spec, "--system-prompt"), Some("Be terse."));
    }

    #[test]
    fn carries_a_system_message_to_the_system_flag() {
        let mut req = request("hi");
        req.chat_history = OneOrMany::many(vec![
            Message::System {
                content: "Be terse.".to_owned(),
            },
            Message::user("hi"),
        ])
        .unwrap();
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(value_after(&spec, "--system-prompt"), Some("Be terse."));
    }

    #[test]
    fn joins_a_preamble_and_a_system_message() {
        let mut req = request("hi");
        req.preamble = Some("First.".to_owned());
        req.chat_history = OneOrMany::many(vec![
            Message::System {
                content: "Second.".to_owned(),
            },
            Message::user("hi"),
        ])
        .unwrap();
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(
            value_after(&spec, "--system-prompt"),
            Some("First.\n\nSecond.")
        );
    }

    #[test]
    fn keeps_a_system_message_out_of_the_prompt() {
        let mut req = request("hi");
        req.chat_history = OneOrMany::many(vec![
            Message::System {
                content: "Be terse.".to_owned(),
            },
            Message::user("first"),
            Message::assistant("second"),
            Message::user("third"),
        ])
        .unwrap();
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        let prompt = spec.args.last().unwrap();
        assert!(!prompt.contains("Be terse."), "{prompt}");
        assert!(prompt.contains("user: first"), "{prompt}");
        assert!(prompt.contains("assistant: second"), "{prompt}");
        assert!(prompt.ends_with("third"), "{prompt}");
    }

    #[test]
    fn omits_the_transcript_for_a_single_turn() {
        let spec = build("claude", "haiku", &request("only"), Mode::Blocking).unwrap();
        assert_eq!(spec.args.last().map(String::as_str), Some("only"));
    }

    #[test]
    fn renders_prior_turns_as_a_transcript() {
        let mut req = request("ignored");
        req.chat_history = OneOrMany::many(vec![
            Message::user("first"),
            Message::assistant("second"),
            Message::user("third"),
        ])
        .unwrap();
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        let prompt = spec.args.last().unwrap();
        assert_eq!(
            prompt,
            "<transcript>\nuser: first\nassistant: second\n</transcript>\n\nthird"
        );
    }

    #[test]
    fn renders_documents_as_context() {
        let mut req = request("question");
        req.documents = vec![document("d1", "alpha"), document("d2", "beta")];
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(
            spec.args.last().unwrap(),
            "<context>\nalpha\nbeta\n</context>\n\nquestion"
        );
    }

    #[test]
    fn orders_context_before_the_transcript() {
        let mut req = request("ignored");
        req.documents = vec![document("d1", "alpha")];
        req.chat_history =
            OneOrMany::many(vec![Message::user("first"), Message::user("second")]).unwrap();
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        let prompt = spec.args.last().unwrap();
        let context = prompt.find("<context>").unwrap();
        let transcript = prompt.find("<transcript>").unwrap();
        assert!(context < transcript, "{prompt}");
    }

    #[test]
    fn joins_multiple_text_blocks_in_one_message() {
        let mut req = request("ignored");
        req.chat_history = OneOrMany::one(Message::User {
            content: OneOrMany::many(vec![
                UserContent::text("line one"),
                UserContent::text("line two"),
            ])
            .unwrap(),
        });
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(spec.args.last().unwrap(), "line one\nline two");
    }

    #[test]
    fn drops_non_text_content() {
        let mut req = request("ignored");
        req.chat_history = OneOrMany::one(Message::User {
            content: OneOrMany::many(vec![
                UserContent::text("keep me"),
                UserContent::image_url("https://example.invalid/cat.png", None, None),
            ])
            .unwrap(),
        });
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(spec.args.last().unwrap(), "keep me");
    }

    #[test]
    fn renders_a_trailing_assistant_message_as_the_prompt() {
        // A prefill: the caller ends the history with a partial assistant
        // turn for the model to continue.
        let mut req = request("ignored");
        req.chat_history = OneOrMany::many(vec![
            Message::user("finish this sentence"),
            Message::assistant("the answer is"),
        ])
        .unwrap();
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(
            spec.args.last().unwrap(),
            "<transcript>\nuser: finish this sentence\n</transcript>\n\nthe answer is"
        );
    }

    #[test]
    fn drops_non_text_assistant_content() {
        let mut req = request("ignored");
        req.chat_history = OneOrMany::many(vec![
            Message::Assistant {
                id: None,
                content: OneOrMany::many(vec![
                    AssistantContent::text("keep me"),
                    AssistantContent::tool_call("call-1", "add", serde_json::json!({"a": 1})),
                ])
                .unwrap(),
            },
            Message::user("and now?"),
        ])
        .unwrap();
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        let prompt = spec.args.last().unwrap();
        assert!(prompt.contains("assistant: keep me"), "{prompt}");
        assert!(!prompt.contains("add"), "{prompt}");
    }

    #[test]
    fn renders_a_trailing_system_message_as_the_prompt() {
        let mut req = request("ignored");
        req.chat_history = OneOrMany::one(Message::System {
            content: "system as prompt".to_owned(),
        });
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        assert_eq!(spec.args.last().unwrap(), "system as prompt");
        assert_eq!(
            value_after(&spec, "--system-prompt"),
            Some("system as prompt")
        );
    }

    #[test]
    fn rejects_tool_definitions() {
        let mut req = request("hi");
        req.tools = vec![rig_core::completion::ToolDefinition {
            name: "add".to_owned(),
            description: "adds".to_owned(),
            parameters: serde_json::json!({}),
        }];
        let error = build("claude", "haiku", &req, Mode::Blocking)
            .unwrap_err()
            .to_string();
        assert!(error.contains("tool definitions"), "{error}");
        assert!(error.contains("--mcp-config"), "{error}");
    }

    #[test]
    fn rejects_temperature() {
        let mut req = request("hi");
        req.temperature = Some(0.5);
        let error = build("claude", "haiku", &req, Mode::Blocking)
            .unwrap_err()
            .to_string();
        assert!(error.contains("temperature"), "{error}");
    }

    #[test]
    fn rejects_max_tokens() {
        let mut req = request("hi");
        req.max_tokens = Some(256);
        let error = build("claude", "haiku", &req, Mode::Blocking)
            .unwrap_err()
            .to_string();
        assert!(error.contains("max_tokens"), "{error}");
    }

    #[test]
    fn passes_an_output_schema_to_the_json_schema_flag() {
        let mut req = request("hi");
        req.output_schema = Some(rig_core::schemars::schema_for!(String));
        let spec = build("claude", "haiku", &req, Mode::Blocking).unwrap();
        let sent = value_after(&spec, "--json-schema").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(sent).unwrap();
        assert_eq!(parsed.get("type"), Some(&serde_json::json!("string")));
    }

    #[test]
    fn omits_the_json_schema_flag_when_no_schema_is_set() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert!(!spec.args.contains(&"--json-schema".to_owned()));
    }

    #[test]
    fn blocking_mode_asks_for_a_single_envelope() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        assert_eq!(value_after(&spec, "--output-format"), Some("json"));
        assert!(!spec.args.contains(&"--verbose".to_owned()));
        assert!(!spec.args.contains(&"--include-partial-messages".to_owned()));
    }

    #[test]
    fn streaming_mode_asks_for_frames_and_partials() {
        let spec = build("claude", "haiku", &request("hi"), Mode::Streaming).unwrap();
        assert_eq!(value_after(&spec, "--output-format"), Some("stream-json"));
        assert!(
            spec.args.contains(&"--include-partial-messages".to_owned()),
            "without partials the stream carries whole messages, not deltas"
        );
        assert!(
            spec.args.contains(&"--verbose".to_owned()),
            "the CLI refuses to stream frames in print mode without --verbose"
        );
    }

    #[test]
    fn both_modes_agree_on_everything_but_the_output_format() {
        let blocking = build("claude", "haiku", &request("hi"), Mode::Blocking).unwrap();
        let streaming = build("claude", "haiku", &request("hi"), Mode::Streaming).unwrap();
        assert_eq!(blocking.program, streaming.program);
        assert_eq!(blocking.args.last(), streaming.args.last());
        for flag in LEAN_FLAGS {
            assert!(blocking.args.contains(&(*flag).to_owned()), "{flag}");
            assert!(streaming.args.contains(&(*flag).to_owned()), "{flag}");
        }
    }

    #[test]
    fn rejects_additional_params() {
        let mut req = request("hi");
        req.additional_params = Some(serde_json::json!({"top_k": 5}));
        let error = build("claude", "haiku", &req, Mode::Blocking)
            .unwrap_err()
            .to_string();
        assert!(error.contains("additional_params"), "{error}");
    }

    #[test]
    fn accepts_an_inert_tool_choice() {
        let mut req = request("hi");
        req.tool_choice = Some(rig_core::message::ToolChoice::None);
        assert!(
            build("claude", "haiku", &req, Mode::Blocking).is_ok(),
            "tool choice is inert when no tools are advertised"
        );
    }
}
