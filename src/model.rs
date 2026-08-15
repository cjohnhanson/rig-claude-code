//! The completion model: the only part of the crate that performs IO.

use std::process::Stdio;

use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
};
use rig_core::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use tokio::io::{AsyncBufReadExt as _, AsyncReadExt as _, BufReader};

use crate::client::{ClaudeCodeClient, DEFAULT_BINARY};
use crate::response::{self, CliResult};
use crate::{request, streaming};

/// Environment variable Claude Code sets to mark a nested session.
///
/// It is removed from the child's environment: with it set, a `claude`
/// launched from inside a Claude Code session treats itself as nested.
const NESTED_SESSION_ENV: &str = "CLAUDECODE";

/// Unwrap a child's pipe handle.
///
/// `spawn` only leaves one of these empty when the corresponding
/// [`Stdio`] was not configured as a pipe, which the caller three lines
/// earlier guarantees it was. The guard stays because that guarantee lives in
/// a different statement from this one, and a future edit to the `Stdio`
/// configuration should produce a named error rather than a panic.
fn require_pipe<T>(pipe: Option<T>, name: &str) -> Result<T, CompletionError> {
    pipe.ok_or_else(|| {
        CompletionError::ProviderError(format!("the child process exposed no {name}"))
    })
}

/// A rig completion model that runs `claude -p` for each turn.
///
/// Build one through [`ClaudeCodeClient`], or directly with
/// [`ClaudeCodeModel::new`] when there is no client to hang it off.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaudeCodeModel {
    model: String,
    binary: String,
}

impl ClaudeCodeModel {
    /// A model handle for the given model alias or id, running the `claude`
    /// binary on `PATH`.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            binary: DEFAULT_BINARY.to_owned(),
        }
    }

    /// Run a specific binary instead of the one on `PATH`.
    #[must_use]
    pub fn with_binary(mut self, binary: impl Into<String>) -> Self {
        self.binary = binary.into();
        self
    }

    /// The model alias or id passed to `--model`.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The binary this model runs.
    #[must_use]
    pub fn binary(&self) -> &str {
        &self.binary
    }
}

impl CompletionModel for ClaudeCodeModel {
    type Response = CliResult;
    type StreamingResponse = CliResult;
    type Client = ClaudeCodeClient;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(model).with_binary(client.binary())
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let spec = request::build(&self.binary, &self.model, &request, request::Mode::Blocking)?;

        let output = tokio::process::Command::new(&spec.program)
            .args(&spec.args)
            .env_remove(NESTED_SESSION_ENV)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .map_err(|error| {
                CompletionError::ProviderError(format!("cannot run `{}`: {error}", spec.program))
            })?;

        if !output.status.success() {
            return Err(CompletionError::ProviderError(format!(
                "`{}` exited with {}: {}",
                spec.program,
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }

        response::parse(&output.stdout)?.into_completion_response()
    }

    /// Stream a turn, translating the CLI's JSON frames into raw streaming
    /// events as they arrive.
    ///
    /// Text deltas arrive as [`RawStreamingChoice::Message`] and extended
    /// thinking as [`RawStreamingChoice::ReasoningDelta`]. The terminal
    /// envelope supplies the session id and the final response, which is what
    /// populates usage on the aggregated result.
    ///
    /// # Errors
    ///
    /// Returns [`CompletionError::RequestError`] for a request the CLI cannot
    /// express, and [`CompletionError::ProviderError`] when the process cannot
    /// be started. Failures that only become visible mid-stream — a non-zero
    /// exit, a failed turn, a stream that ends without its terminal frame —
    /// surface as an error item within the stream.
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let spec = request::build(
            &self.binary,
            &self.model,
            &request,
            request::Mode::Streaming,
        )?;

        let mut child = tokio::process::Command::new(&spec.program)
            .args(&spec.args)
            .env_remove(NESTED_SESSION_ENV)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| {
                CompletionError::ProviderError(format!("cannot run `{}`: {error}", spec.program))
            })?;

        let stdout = require_pipe(child.stdout.take(), "stdout")?;
        let stderr = require_pipe(child.stderr.take(), "stderr")?;

        // Drain stderr concurrently. A child that fills the stderr pipe while
        // this end only reads stdout would block forever.
        let stderr_drain = async move {
            let mut buffer = String::new();
            let mut reader = BufReader::new(stderr);
            let _ = reader.read_to_string(&mut buffer).await;
            Ok::<String, ()>(buffer)
        };

        let program = spec.program.clone();
        let frames = async_stream::stream! {
            let mut lines = BufReader::new(stdout).lines();
            let mut saw_terminal_frame = false;

            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => match streaming::classify(&line) {
                        streaming::Event::Emit(choice) => yield Ok(choice),
                        streaming::Event::Finish(result) => {
                            saw_terminal_frame = true;
                            if let Some(id) = result.session_id.clone() {
                                yield Ok(RawStreamingChoice::MessageId(id));
                            }
                            yield Ok(RawStreamingChoice::FinalResponse(*result));
                        }
                        streaming::Event::Fail(error) => {
                            yield Err(error);
                            return;
                        }
                        streaming::Event::Skip => {}
                    },
                    Ok(None) => break,
                    Err(error) => {
                        yield Err(CompletionError::ResponseError(format!(
                            "reading output from `{program}`: {error}"
                        )));
                        return;
                    }
                }
            }

            let status = child.wait().await;
            let stderr_text = stderr_drain.await.unwrap_or_default();

            match status {
                Ok(status) if !status.success() => {
                    yield Err(CompletionError::ProviderError(format!(
                        "`{program}` exited with {status}: {}",
                        stderr_text.trim()
                    )));
                }
                Ok(_) if !saw_terminal_frame => {
                    yield Err(CompletionError::ResponseError(format!(
                        "`{program}` closed the stream without a terminal result frame"
                    )));
                }
                Ok(_) => {}
                Err(error) => {
                    yield Err(CompletionError::ProviderError(format!(
                        "waiting for `{program}`: {error}"
                    )));
                }
            }
        };

        Ok(StreamingCompletionResponse::stream(Box::pin(frames)))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use rig_core::OneOrMany;
    use rig_core::completion::Message;

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

    #[test]
    fn require_pipe_passes_a_present_handle_through() {
        assert_eq!(require_pipe(Some(7), "stdout").unwrap(), 7);
    }

    #[test]
    fn require_pipe_names_the_missing_handle() {
        let error = require_pipe::<()>(None, "stderr").unwrap_err().to_string();
        assert!(error.contains("exposed no stderr"), "{error}");
    }

    #[test]
    fn defaults_to_the_binary_on_path() {
        assert_eq!(ClaudeCodeModel::new("haiku").binary(), "claude");
    }

    #[test]
    fn takes_an_explicit_binary() {
        let model = ClaudeCodeModel::new("haiku").with_binary("/opt/claude");
        assert_eq!(model.binary(), "/opt/claude");
        assert_eq!(model.model(), "haiku");
    }

    #[tokio::test]
    async fn reports_a_missing_binary_as_a_provider_error() {
        let model = ClaudeCodeModel::new("haiku").with_binary("/nonexistent/claude-binary");
        let error = model.completion(request()).await.unwrap_err().to_string();
        assert!(error.contains("cannot run"), "{error}");
        assert!(error.contains("/nonexistent/claude-binary"), "{error}");
    }

    #[tokio::test]
    async fn rejects_an_unsupported_request_before_spawning() {
        // The binary does not exist, so reaching a spawn would produce the
        // "cannot run" error instead. Getting the request-level error proves
        // the guard runs first.
        let model = ClaudeCodeModel::new("haiku").with_binary("/nonexistent/claude-binary");
        let mut req = request();
        req.temperature = Some(0.1);
        let error = model.completion(req).await.unwrap_err().to_string();
        assert!(error.contains("temperature"), "{error}");
    }

    #[tokio::test]
    async fn streaming_reports_a_missing_binary_before_returning_a_stream() {
        let model = ClaudeCodeModel::new("haiku").with_binary("/nonexistent/claude-binary");
        // `StreamingCompletionResponse` is not `Debug`, so the success side
        // cannot be unwrapped away; match instead.
        match model.stream(request()).await {
            Err(error) => {
                let rendered = error.to_string();
                assert!(rendered.contains("cannot run"), "{rendered}");
                assert!(
                    rendered.contains("/nonexistent/claude-binary"),
                    "{rendered}"
                );
            }
            Ok(_) => panic!("a missing binary should fail before a stream exists"),
        }
    }

    #[tokio::test]
    async fn streaming_rejects_an_unsupported_request_before_spawning() {
        let model = ClaudeCodeModel::new("haiku").with_binary("/nonexistent/claude-binary");
        let mut req = request();
        req.max_tokens = Some(10);
        match model.stream(req).await {
            Err(error) => assert!(error.to_string().contains("max_tokens"), "{error}"),
            Ok(_) => panic!("an unsupported request should be refused"),
        }
    }
}
