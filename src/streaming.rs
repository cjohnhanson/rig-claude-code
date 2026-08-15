//! Translation of the CLI's newline-delimited JSON frames into rig's raw
//! streaming events.
//!
//! With `--output-format stream-json --include-partial-messages`, the CLI
//! wraps Anthropic's own streaming protocol: each line is a JSON object, and
//! a line of `"type": "stream_event"` carries one protocol event under
//! `event`. Everything this crate needs lives in three of them —
//! `content_block_delta` for text and thinking, and the terminal `result`
//! line for usage — so unrecognized frames are skipped rather than rejected.
//! A newer CLI emitting new frame types therefore does not break a stream.

use rig_core::completion::CompletionError;
use rig_core::streaming::RawStreamingChoice;
use serde::Deserialize;

use crate::response::CliResult;

/// One line of the CLI's frame stream.
///
/// Only the frames this crate acts on are modelled. Any other `type` value
/// deserializes into [`Frame::Other`] and is skipped.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Frame {
    /// A wrapped Anthropic streaming protocol event.
    #[serde(rename = "stream_event")]
    StreamEvent {
        /// The wrapped event.
        event: StreamEvent,
    },
    /// The terminal envelope, identical to blocking mode's stdout.
    #[serde(rename = "result")]
    Result(CliResult),
    /// A frame this crate does not act on.
    #[serde(other)]
    Other,
}

/// The wrapped protocol events this crate acts on.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum StreamEvent {
    /// An incremental update to an open content block.
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta {
        /// The incremental payload.
        delta: Delta,
    },
    /// An event this crate does not act on.
    #[serde(other)]
    Other,
}

/// The payload of a `content_block_delta`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Delta {
    /// Assistant-visible text.
    #[serde(rename = "text_delta")]
    Text {
        /// The text fragment.
        text: String,
    },
    /// Extended-thinking text.
    #[serde(rename = "thinking_delta")]
    Thinking {
        /// The reasoning fragment.
        thinking: String,
    },
    /// A delta kind this crate does not act on, such as `signature_delta`.
    #[serde(other)]
    Other,
}

/// What one input line means to the stream driver.
pub(crate) enum Event {
    /// Yield this event to the consumer.
    Emit(RawStreamingChoice<CliResult>),
    /// Yield this event and then the terminal envelope's identity.
    Finish(Box<CliResult>),
    /// Fail the stream.
    Fail(CompletionError),
    /// Nothing to do.
    Skip,
}

/// Classify one line of the CLI's output.
///
/// A line that is not JSON at all is skipped rather than fatal: the CLI writes
/// diagnostics to stderr, but a stray non-JSON line on stdout should not
/// destroy an otherwise good stream. A malformed *terminal* envelope is a
/// different matter and does fail the stream, because without it there is no
/// usage and no completion.
pub(crate) fn classify(line: &str) -> Event {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Event::Skip;
    }

    let Ok(frame) = serde_json::from_str::<Frame>(trimmed) else {
        // Distinguish "not our frame" from "our frame, but broken". A line
        // that parses as JSON and calls itself a result but will not
        // deserialize is a protocol change worth reporting.
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed)
            && value.get("type").and_then(serde_json::Value::as_str) == Some("result")
        {
            return Event::Fail(CompletionError::ResponseError(format!(
                "unparseable terminal frame from claude: {trimmed}"
            )));
        }
        return Event::Skip;
    };

    match frame {
        Frame::StreamEvent { event } => match event {
            StreamEvent::ContentBlockDelta { delta } => match delta {
                Delta::Text { text } => Event::Emit(RawStreamingChoice::Message(text)),
                Delta::Thinking { thinking } => Event::Emit(RawStreamingChoice::ReasoningDelta {
                    id: None,
                    reasoning: thinking,
                }),
                Delta::Other => Event::Skip,
            },
            StreamEvent::Other => Event::Skip,
        },
        Frame::Result(result) => {
            if result.is_error {
                Event::Fail(CompletionError::ProviderError(format!(
                    "claude reported a failed turn ({}): {}",
                    result.subtype,
                    result.result.as_deref().unwrap_or("no detail")
                )))
            } else {
                Event::Finish(Box::new(result))
            }
        }
        Frame::Other => Event::Skip,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    /// The emitted text of a classified line, if it produced a text delta.
    fn text_of(line: &str) -> Option<String> {
        match classify(line) {
            Event::Emit(RawStreamingChoice::Message(text)) => Some(text),
            _ => None,
        }
    }

    /// The emitted reasoning of a classified line, with its block id.
    fn reasoning_of(line: &str) -> Option<(Option<String>, String)> {
        match classify(line) {
            Event::Emit(RawStreamingChoice::ReasoningDelta { id, reasoning }) => {
                Some((id, reasoning))
            }
            _ => None,
        }
    }

    /// Whether a classified line produced nothing.
    fn is_skipped(line: &str) -> bool {
        matches!(classify(line), Event::Skip)
    }

    /// The terminal envelope of a classified line, if it finished the stream.
    fn finished(line: &str) -> Option<CliResult> {
        match classify(line) {
            Event::Finish(result) => Some(*result),
            _ => None,
        }
    }

    /// The rendered error of a classified line, if it failed the stream.
    fn failure_of(line: &str) -> Option<String> {
        match classify(line) {
            Event::Fail(error) => Some(error.to_string()),
            _ => None,
        }
    }

    fn skipped(line: &str) {
        assert!(is_skipped(line), "expected a skip for: {line}");
    }

    fn failed(line: &str) -> String {
        failure_of(line).unwrap_or_else(|| panic!("expected a failure for: {line}"))
    }

    #[test]
    fn emits_text_deltas() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta",
                "index":1,"delta":{"type":"text_delta","text":"1 2 3"}}}"#;
        assert_eq!(text_of(line).as_deref(), Some("1 2 3"));
        assert_eq!(reasoning_of(line), None);
        assert_eq!(failure_of(line), None);
        assert!(finished(line).is_none());
    }

    #[test]
    fn emits_thinking_deltas_as_reasoning() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta",
                "index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}}"#;
        assert_eq!(reasoning_of(line), Some((None, "hmm".to_owned())));
        assert_eq!(text_of(line), None);
    }

    #[test]
    fn emits_an_empty_text_delta_verbatim() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta",
                "delta":{"type":"text_delta","text":""}}}"#;
        assert_eq!(text_of(line).as_deref(), Some(""));
    }

    #[test]
    fn finishes_on_the_terminal_envelope() {
        let line = r#"{"type":"result","is_error":false,"subtype":"success","result":"done",
                "session_id":"s-9","usage":{"input_tokens":4,"output_tokens":2}}"#;
        let result = finished(line).expect("a terminal envelope finishes the stream");
        assert_eq!(result.result.as_deref(), Some("done"));
        assert_eq!(result.session_id.as_deref(), Some("s-9"));
        assert_eq!(result.usage.input_tokens, 4);
        assert!(!is_skipped(line));
    }

    #[test]
    fn fails_on_a_failed_terminal_envelope() {
        let rendered = failed(
            r#"{"type":"result","is_error":true,"subtype":"error_max_turns","result":"gave up"}"#,
        );
        assert!(rendered.contains("error_max_turns"), "{rendered}");
        assert!(rendered.contains("gave up"), "{rendered}");
    }

    #[test]
    fn fails_on_a_terminal_frame_it_cannot_read() {
        let rendered = failed(r#"{"type":"result","usage":"not an object"}"#);
        assert!(
            rendered.contains("unparseable terminal frame"),
            "{rendered}"
        );
    }

    #[test]
    fn skips_signature_deltas() {
        skipped(
            r#"{"type":"stream_event","event":{"type":"content_block_delta",
                "delta":{"type":"signature_delta","signature":"abc"}}}"#,
        );
    }

    #[test]
    fn skips_protocol_events_it_does_not_model() {
        skipped(r#"{"type":"stream_event","event":{"type":"message_start","message":{}}}"#);
        skipped(r#"{"type":"stream_event","event":{"type":"content_block_stop","index":0}}"#);
        skipped(r#"{"type":"stream_event","event":{"type":"message_stop"}}"#);
    }

    #[test]
    fn skips_frames_it_does_not_model() {
        skipped(r#"{"type":"system","subtype":"init","session_id":"s-1"}"#);
        skipped(r#"{"type":"assistant","message":{"content":[]}}"#);
        skipped(r#"{"type":"rate_limit_event","rate_limit_info":{"status":"allowed"}}"#);
    }

    #[test]
    fn skips_a_frame_type_invented_after_this_crate() {
        skipped(r#"{"type":"some_future_frame","payload":{"anything":true}}"#);
    }

    #[test]
    fn skips_blank_and_whitespace_lines() {
        skipped("");
        skipped("   \t  ");
    }

    #[test]
    fn skips_a_non_json_line() {
        skipped("Warning: something happened");
    }

    #[test]
    fn skips_json_that_is_not_a_frame() {
        skipped("[1, 2, 3]");
        skipped(r#""a bare string""#);
    }
}
