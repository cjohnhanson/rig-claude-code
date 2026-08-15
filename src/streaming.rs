//! Reading and classifying the CLI's newline-delimited JSON frames.
//!
//! With `--output-format stream-json --include-partial-messages`, the CLI
//! wraps Anthropic's own streaming protocol: each line is a JSON object, and a
//! line of `"type": "stream_event"` carries one protocol event under `event`.
//! The crate acts on three of them: `content_block_delta` for text and
//! thinking, and the terminal `result` line for usage. It skips every other
//! frame rather than rejecting it. A newer CLI emitting
//! new frame types therefore does not break a stream.

use rig_core::completion::CompletionError;
use rig_core::streaming::RawStreamingChoice;
use serde::Deserialize;
use tokio::io::{AsyncRead, AsyncReadExt as _};

use crate::response::CliResponse;

/// The largest single frame this crate will buffer.
///
/// A child that emits an endless line with no newline would otherwise grow the
/// buffer until the process dies. Real frames are a few hundred bytes; a whole
/// assistant message is far below this.
pub(crate) const MAX_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// A newline splitter that reads bytes rather than `str`.
///
/// [`tokio::io::AsyncBufReadExt::lines`] fails the whole stream on invalid
/// UTF-8, which would discard a perfectly good terminal frame that happened to
/// follow one bad byte. Decoding each line lossily keeps the noise-tolerance
/// policy consistent with [`classify`].
pub(crate) struct Lines<R> {
    reader: R,
    buffer: Vec<u8>,
    /// Bytes read but not yet split into a line.
    pending: Vec<u8>,
    /// How much of `pending` has already been searched for a newline.
    ///
    /// Without this the search restarts at the front after every read, which
    /// is quadratic in the frame size: a 16 MiB frame arriving in 8 KiB reads
    /// rescans about 16 GiB. Measured at 44 seconds for one test before this
    /// offset existed.
    searched: usize,
    done: bool,
}

impl<R: AsyncRead + Unpin> Lines<R> {
    /// Give the reader back, so the caller can keep the pipe drained after it
    /// stops splitting lines.
    ///
    /// Anything already buffered but not yet returned as a line is discarded:
    /// the caller is done reading frames, and the point of taking the reader
    /// back is to stop the pipe filling.
    pub(crate) fn into_reader(self) -> R {
        self.reader
    }

    /// Wrap a reader.
    pub(crate) fn new(reader: R) -> Self {
        Self {
            reader,
            buffer: vec![0; 8192],
            pending: Vec::new(),
            searched: 0,
            done: false,
        }
    }

    /// The next line, without its terminator, lossily decoded.
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying reader fails, or when a single
    /// frame exceeds [`MAX_FRAME_BYTES`].
    pub(crate) async fn next_line(&mut self) -> std::io::Result<Option<String>> {
        loop {
            let unsearched = self.pending.get(self.searched..).unwrap_or_default();
            if let Some(offset) = unsearched.iter().position(|byte| *byte == b'\n') {
                let end = self.searched.saturating_add(offset);
                let mut line: Vec<u8> = self.pending.drain(..=end).collect();
                self.searched = 0;
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
            }
            self.searched = self.pending.len();

            if self.done {
                if self.pending.is_empty() {
                    return Ok(None);
                }
                // A final line with no terminator is still a line.
                let line = std::mem::take(&mut self.pending);
                self.searched = 0;
                return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
            }

            let read = self.reader.read(&mut self.buffer).await?;
            if read == 0 {
                self.done = true;
                continue;
            }
            if self.pending.len().saturating_add(read) > MAX_FRAME_BYTES {
                return Err(std::io::Error::other(format!(
                    "a single frame exceeded {MAX_FRAME_BYTES} bytes without a newline"
                )));
            }
            self.pending
                .extend_from_slice(self.buffer.get(..read).unwrap_or_default());
        }
    }
}

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
    Result(CliResponse),
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
    Emit(RawStreamingChoice<CliResponse>),
    /// The turn is over; this is its envelope.
    Finish(Box<CliResponse>),
    /// Fail the stream.
    Fail(CompletionError),
    /// Nothing to do.
    Skip,
}

/// Classify one line of the CLI's output.
///
/// A line that is not JSON at all is skipped rather than fatal: the CLI writes
/// diagnostics to stderr, but a stray non-JSON line on stdout should not
/// destroy an otherwise good stream. A malformed *terminal* frame is a
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
                "unparseable terminal frame from claude: {}",
                crate::response::quote(trimmed.as_bytes())
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
        Frame::Result(result) => match result.failure() {
            Some(failure) => Event::Fail(failure),
            None => Event::Finish(Box::new(result)),
        },
        Frame::Other => Event::Skip,
    }
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
    fn finished(line: &str) -> Option<CliResponse> {
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

    async fn collect_lines(input: &[u8]) -> std::io::Result<Vec<String>> {
        let mut lines = Lines::new(input);
        let mut out = Vec::new();
        while let Some(line) = lines.next_line().await? {
            out.push(line);
        }
        Ok(out)
    }

    #[tokio::test]
    async fn splits_on_newlines() {
        let lines = collect_lines(b"one\ntwo\nthree\n").await.unwrap();
        assert_eq!(lines, ["one", "two", "three"]);
    }

    #[tokio::test]
    async fn keeps_a_final_line_with_no_terminator() {
        let lines = collect_lines(b"one\ntruncated").await.unwrap();
        assert_eq!(lines, ["one", "truncated"]);
    }

    #[tokio::test]
    async fn strips_a_carriage_return() {
        let lines = collect_lines(b"one\r\ntwo\r\n").await.unwrap();
        assert_eq!(lines, ["one", "two"]);
    }

    #[tokio::test]
    async fn yields_nothing_for_empty_input() {
        assert!(collect_lines(b"").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn survives_invalid_utf8_and_keeps_reading() {
        // `lines()` would fail the whole stream here, discarding the good
        // terminal frame that follows one bad byte.
        let input = b"\xff\xfe broken\n{\"type\":\"result\",\"result\":\"ok\"}\n";
        let lines = collect_lines(input).await.unwrap();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("broken"), "{:?}", lines[0]);
        assert_eq!(finished(&lines[1]).unwrap().result.as_deref(), Some("ok"));
    }

    #[tokio::test]
    async fn reads_a_line_larger_than_the_read_buffer() {
        let big = "x".repeat(64 * 1024);
        let input = format!("{big}\nafter\n");
        let lines = collect_lines(input.as_bytes()).await.unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].len(), big.len());
        assert_eq!(lines[1], "after");
    }

    #[tokio::test]
    async fn splits_a_very_large_frame_in_bounded_time() {
        // The `searched` offset is what makes this linear. Without it the scan
        // restarts at the front after every read: one 16 MiB frame arriving in
        // 8 KiB reads rescans about 16 GiB, measured at 44 seconds.
        let big = "x".repeat(MAX_FRAME_BYTES - 1);
        let input = format!("{big}\n");
        let started = std::time::Instant::now();
        let lines = collect_lines(input.as_bytes()).await.unwrap();
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].len(), big.len());
        assert!(
            started.elapsed() < std::time::Duration::from_secs(5),
            "took {:?}; the newline scan is quadratic again",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn refuses_a_frame_with_no_newline_in_sight() {
        let endless = vec![b'x'; MAX_FRAME_BYTES + 1];
        let error = collect_lines(&endless).await.unwrap_err();
        assert!(error.to_string().contains("exceeded"), "{error}");
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
    fn decodes_escapes_in_a_delta() {
        let line = r#"{"type":"stream_event","event":{"type":"content_block_delta",
                "delta":{"type":"text_delta","text":"café \"q\"\\ and\nnewline"}}}"#;
        assert_eq!(
            text_of(line).as_deref(),
            Some("café \"q\"\\ and\nnewline"),
            "model output contains all four of these constantly"
        );
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
    fn fails_on_an_error_subtype_whatever_the_flag_says() {
        let rendered = failed(
            r#"{"type":"result","is_error":false,"subtype":"error_max_turns","result":"gave up"}"#,
        );
        assert!(rendered.contains("error_max_turns"), "{rendered}");
    }

    #[test]
    fn fails_on_a_terminal_frame_it_cannot_read() {
        let rendered = failed(r#"{"type":"result","usage":[1,2,3]}"#);
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
