//! The completion model: the only part of the crate that performs IO.

use std::io::Write as _;
use std::process::Stdio;
use std::time::Duration;

use rig_core::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
};
use rig_core::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use rig_core::{OneOrMany, ProviderResponseError};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;

use crate::bridge::{Bridge, RecordedCall};
use crate::client::{ClaudeCodeClient, DEFAULT_BINARY};
use crate::request::CommandSpec;
use crate::response::{self, CliResponse};
use crate::{request, streaming};

/// Environment variables a live Claude Code session exports to mark itself,
/// all of which are removed from the child's environment.
///
/// `CLAUDECODE` alone is not enough. A session also exports a messaging socket
/// and token, a session id, an entrypoint, and `CLAUDE_EFFORT`. That last one
/// would change the effort level of every turn according to who launched the
/// host process, and with it the cost and reproducibility of the invocation.
///
/// These are matched by exact name, not by prefix. A `CLAUDE_` prefix would
/// also strip `CLAUDE_CONFIG_DIR`, which names the `.claude` directory the CLI
/// reads, and so selects which account's credential pays for the turn. It is
/// the standard way to keep several logins on one machine. A prefix would also
/// strip `CLAUDE_CODE_USE_BEDROCK` and `CLAUDE_CODE_USE_VERTEX`, which route
/// the turn to a different backend. Those are the caller's to control, exactly as
/// `ANTHROPIC_API_KEY` is; removing them would silently switch accounts.
const SESSION_MARKERS: &[&str] = &[
    "CLAUDECODE",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
    "AI_AGENT",
];

/// Remove every inherited Claude Code session marker from `command`.
fn strip_session_markers(command: &mut Command) {
    for marker in SESSION_MARKERS {
        command.env_remove(marker);
    }
}

/// Flags this crate sets itself, which [`ClaudeCodeModel::with_args`] must not
/// duplicate.
///
/// Passing `--model` through the escape hatch would otherwise silently
/// override the model the rig request asked for, which is the same class of
/// quiet misconfiguration the request guards exist to prevent.
const OWNED_FLAGS: &[&str] = &[
    "-p",
    "--print",
    "--output-format",
    "--model",
    "--tools",
    "--strict-mcp-config",
    "--setting-sources",
    "--disable-slash-commands",
    "--json-schema",
    "--system-prompt",
    "--system-prompt-file",
    "--include-partial-messages",
    "--verbose",
];

/// How long to keep reading a pipe after the child has exited.
///
/// Long enough that a loaded machine still delivers the last of the output,
/// short enough that a grandchild holding the pipe open does not hold the turn
/// open with it. Standard output gets the longer grace because it carries the
/// answer; standard error only decorates an error message.
const STDOUT_GRACE: Duration = Duration::from_secs(5);

/// How long to keep reading stderr after the child has exited.
const STDERR_GRACE: Duration = Duration::from_millis(200);

/// The most stderr kept for an error message.
const STDERR_LIMIT: usize = 64 * 1024;

/// The most stdout accepted from a blocking turn.
const STDOUT_LIMIT: usize = 16 * 1024 * 1024;

/// Unwrap a child's pipe handle.
///
/// `spawn` only leaves one of these empty when the corresponding [`Stdio`] was
/// not configured as a pipe, which [`ClaudeCodeModel::spawn_child`]
/// guarantees it was. The guard stays because that guarantee lives in a
/// different function from this one, and a future edit to the `Stdio`
/// configuration should produce a named error rather than a panic.
fn require_pipe<T>(pipe: Option<T>, name: &str) -> Result<T, CompletionError> {
    pipe.ok_or_else(|| {
        CompletionError::ProviderError(format!("the child process exposed no {name}"))
    })
}

/// Write `text` to a fresh temporary file readable only by this user.
///
/// `NamedTempFile` creates the file 0600, which matters because the system
/// prompt holds whatever the caller put in a preamble.
fn write_private_file(text: &str) -> Result<tempfile::NamedTempFile, CompletionError> {
    let write = || -> std::io::Result<tempfile::NamedTempFile> {
        let mut file = tempfile::NamedTempFile::new()?;
        file.write_all(text.as_bytes())?;
        file.flush()?;
        Ok(file)
    };
    write().map_err(|error| {
        CompletionError::ProviderError(format!(
            "cannot create a file for the system prompt: {error}"
        ))
    })
}

/// Report a turn that outlived its deadline.
///
/// `ResponseError` rather than `ProviderError`, for the reason on
/// [`child_failed`]: nothing this crate yields into a stream may be a
/// `ProviderError`.
fn timed_out(program: &str, timeout: Duration) -> CompletionError {
    CompletionError::ResponseError(format!(
        "`{program}` did not finish within {timeout:?}; the turn was abandoned \
         and any child still running was killed"
    ))
}

/// Report a child that exited without success.
///
/// The child's stderr goes into a [`ProviderResponseError`] body, never into a
/// `ProviderError` message. rig's stream driver treats any `ProviderError`
/// whose text contains `aborted` as a cancellation and ends the stream
/// cleanly, with no error item. Node's own `AbortError` message is
/// "This operation was aborted". Quoting untrusted stderr inside a
/// `ProviderError` would let the CLI's most common failure text turn a failed
/// streaming turn into an empty success. `ProviderResponse` is not sniffed,
/// and it hands the caller the text as data.
fn child_failed(program: &str, status: std::process::ExitStatus, stderr: &[u8]) -> CompletionError {
    CompletionError::ProviderResponse(ProviderResponseError {
        status: None,
        body: format!(
            "`{program}` exited with {status}: {}",
            response::quote(stderr)
        ),
    })
}

/// A task that is cancelled when this guard is dropped.
///
/// Dropping a bare [`JoinHandle`] detaches its task rather than cancelling it,
/// so a cancelled turn would leave the feeder blocked in `write_all` and the
/// drains blocked in `read` for as long as anything held the pipes open.
pub(crate) struct AbortOnDrop(pub(crate) JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A running drain of one of the child's pipes.
///
/// The bytes accumulate in a shared buffer rather than in the task's return
/// value, so a caller that stops waiting can still read what arrived. That
/// matters because a pipe reaches end-of-file only when *every* write end
/// closes, including a grandchild that inherited it: an MCP server, an
/// uploader, a telemetry flush. Waiting for that would hold the turn open for
/// the grandchild's whole lifetime.
struct Drain {
    task: AbortOnDrop,
    buffer: Arc<Mutex<Vec<u8>>>,
    /// Whether the retention limit was reached and output discarded.
    truncated: Arc<AtomicBool>,
}

impl Drain {
    /// Start reading `reader`, keeping at most `limit` bytes.
    ///
    /// Reading continues past the limit and discards the surplus, so the pipe
    /// never fills and the child never blocks writing to it. Only the retained
    /// prefix costs memory.
    fn start<R>(mut reader: R, limit: usize) -> Self
    where
        R: tokio::io::AsyncRead + Unpin + Send + 'static,
    {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let truncated = Arc::new(AtomicBool::new(false));
        let sink = Arc::clone(&buffer);
        let overflowed = Arc::clone(&truncated);
        let task = tokio::spawn(async move {
            let mut chunk = vec![0; 8192];
            loop {
                match reader.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => {
                        let slice = chunk.get(..read).unwrap_or_default();
                        let Ok(mut kept) = sink.lock() else { break };
                        let room = limit.saturating_sub(kept.len());
                        if room < read {
                            overflowed.store(true, Ordering::Relaxed);
                        }
                        if room > 0 {
                            kept.extend_from_slice(slice.get(..room.min(read)).unwrap_or_default());
                        }
                    }
                }
            }
        });
        Self {
            task: AbortOnDrop(task),
            buffer,
            truncated,
        }
    }

    /// Whether output was discarded for exceeding the retention limit.
    fn truncated(&self) -> bool {
        self.truncated.load(Ordering::Relaxed)
    }

    /// Everything read so far, without waiting for the pipe to close.
    fn snapshot(&self) -> Vec<u8> {
        self.buffer
            .lock()
            .map(|bytes| bytes.clone())
            .unwrap_or_default()
    }

    /// Wait up to `grace` for the pipe to close, then take what there is.
    ///
    /// Whatever arrived is returned either way: a pipe still held open by a
    /// grandchild is not a reason to discard the bytes already read.
    async fn finish(self, grace: Duration) -> Vec<u8> {
        let Self {
            mut task,
            buffer,
            truncated: _,
        } = self;
        let _ = tokio::time::timeout(grace, &mut task.0).await;
        // Dropping `task` aborts it, which is right either way: the read is
        // either finished or being abandoned, and the bytes are in the shared
        // buffer regardless.
        drop(task);
        buffer.lock().map(|bytes| bytes.clone()).unwrap_or_default()
    }
}

/// A rig completion model that runs `claude -p` for each turn.
///
/// Build one through [`ClaudeCodeClient`], or directly with
/// [`ClaudeCodeModel::new`] when there is no client to hang it off.
///
/// # Process lifetime
///
/// Every turn is one child process. The child is killed when the future or the
/// stream driving it is dropped, so an abandoned turn does not leave a `claude`
/// running and spending the login's usage. Set
/// [`ClaudeCodeModel::with_timeout`] to bound a turn that never finishes on its
/// own; without one, a wedged child waits forever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaudeCodeModel {
    model: String,
    binary: String,
    extra_args: Vec<String>,
    current_dir: Option<String>,
    timeout: Option<Duration>,
}

impl ClaudeCodeModel {
    /// A model handle for the given model alias or id, running the `claude`
    /// binary on `PATH`.
    ///
    /// `model` is passed to the CLI's `--model`. It takes the aliases in
    /// [`crate::models`] as well as a full model id.
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            binary: DEFAULT_BINARY.to_owned(),
            extra_args: Vec::new(),
            current_dir: None,
            timeout: None,
        }
    }

    /// Run a specific binary instead of the one on `PATH`.
    ///
    /// The binary runs with this process's full privileges, so its path is
    /// trusted configuration. See the crate's trust model in the README.
    #[must_use]
    pub fn with_binary(mut self, binary: impl Into<String>) -> Self {
        self.binary = binary.into();
        self
    }

    /// Pass additional arguments to the CLI.
    ///
    /// This is the escape hatch for CLI capabilities this crate does not model
    /// such as `--add-dir`, `--max-budget-usd`, and `--fallback-model`. The
    /// arguments are appended after the ones the crate sets.
    ///
    /// An argument that collides with a flag the crate owns is rejected when
    /// the turn runs, rather than silently overriding it.
    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Give the CLI an MCP server configuration of your own.
    ///
    /// Equivalent to [`ClaudeCodeModel::with_args`] with `--mcp-config` and the
    /// path. The tools in it run inside the CLI: rig never sees them. rig
    /// tools registered on the agent work alongside them, through the
    /// crate's own per-turn server. That server is named `rig`, so a server
    /// of that name in your configuration is shadowed on any turn that
    /// carries rig tools.
    #[must_use]
    pub fn with_mcp_config(self, path: impl Into<String>) -> Self {
        self.with_args(["--mcp-config".to_owned(), path.into()])
    }

    /// Run the child in a specific working directory.
    ///
    /// The child otherwise inherits this process's directory, which decides
    /// what `--add-dir` and any MCP server resolve relative paths against.
    #[must_use]
    pub fn with_current_dir(mut self, dir: impl Into<String>) -> Self {
        self.current_dir = Some(dir.into());
        self
    }

    /// Kill the child and fail the turn if it runs longer than `timeout`.
    ///
    /// There is no default. A `claude` that wedges holds the caller forever
    /// unless a timeout is set here or the caller imposes its own.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
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

    /// The timeout applied to a turn, if any.
    #[must_use]
    pub fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    /// The working directory the child runs in, if one is set.
    #[must_use]
    pub fn current_dir(&self) -> Option<&str> {
        self.current_dir.as_deref()
    }

    /// The additional arguments passed to the CLI.
    #[must_use]
    pub fn extra_args(&self) -> &[String] {
        &self.extra_args
    }

    /// Start a per-turn MCP server when the request advertises tools.
    async fn start_bridge(
        &self,
        request: &CompletionRequest,
    ) -> Result<Option<Bridge>, CompletionError> {
        if !request::advertises_tools(request) {
            return Ok(None);
        }
        Bridge::start(&request.tools)
            .await
            .map(Some)
            .map_err(|error| {
                CompletionError::ProviderError(format!("cannot start the tool bridge: {error}"))
            })
    }

    /// Reject extra arguments that would override a flag the crate sets.
    fn check_extra_args(&self) -> Result<(), CompletionError> {
        for arg in &self.extra_args {
            let flag = arg.split('=').next().unwrap_or(arg);
            if OWNED_FLAGS.contains(&flag) {
                return Err(CompletionError::RequestError(
                    format!(
                        "`{flag}` is set by rig-claude-code; passing it through with_args \
                         would override the request rather than extend it"
                    )
                    .into(),
                ));
            }
        }
        Ok(())
    }

    /// Start the child described by `spec`.
    ///
    /// Returns the child, the task feeding it the prompt, and the private
    /// files the child reads (the system prompt, the tool bridge's
    /// configuration), which must outlive the child.
    ///
    /// The prompt is written on a separate task. Writing it inline before
    /// reading stdout is a deadlock: a prompt larger than the pipe buffer
    /// blocks this end, while the child blocks writing to a stdout or stderr
    /// pipe nobody is draining. That is the same deadlock the concurrent
    /// stderr drain avoids, entered from the other side, and it would bite
    /// exactly the large prompts stdin exists to carry.
    fn spawn_child(
        &self,
        spec: &CommandSpec,
        bridge: Option<&Bridge>,
    ) -> Result<(Child, AbortOnDrop, Vec<tempfile::NamedTempFile>), CompletionError> {
        self.check_extra_args()?;

        let mut private_files = Vec::new();

        // The system prompt goes in a private file rather than argv: the CLI
        // scans raw argv for `--settings=` ahead of its own option parsing, so
        // a system prompt beginning with that text executes as a flag.
        let system_file = spec
            .system_prompt
            .as_deref()
            .map(write_private_file)
            .transpose()?;

        // The bridge's configuration carries the turn's bearer token. In argv
        // it would be readable through `ps` by any local process, which is
        // exactly the process the token exists to keep out. The CLI reads
        // `--mcp-config` from a file path as well as inline JSON, and
        // forwards `headers` from either (verified against 2.1.233).
        let bridge_file = bridge
            .map(|bridge| write_private_file(&bridge.mcp_config()))
            .transpose()?;

        let mut command = Command::new(&self.binary);
        command
            .args(&spec.args)
            .args(&self.extra_args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        strip_session_markers(&mut command);
        if let Some(file) = system_file {
            command
                .arg("--system-prompt-file")
                .arg(file.path().as_os_str());
            private_files.push(file);
        }
        // Last on purpose. `--mcp-config` and `--allowedTools` take several
        // values, so the CLI would read any positional argument after them
        // as one more value. Nothing follows them here, and the prompt is on
        // stdin.
        if let Some(file) = bridge_file {
            command
                .arg("--mcp-config")
                .arg(file.path().as_os_str())
                .arg("--allowedTools")
                .arg(Bridge::allowed_tools());
            private_files.push(file);
        }
        if let Some(dir) = &self.current_dir {
            command.current_dir(dir);
        }

        let mut child = command.spawn().map_err(|error| {
            CompletionError::ProviderError(format!("cannot run `{}`: {error}", self.binary))
        })?;

        let mut stdin = require_pipe(child.stdin.take(), "stdin")?;
        let prompt = spec.stdin.clone();
        let feed = tokio::spawn(async move {
            // A child that exits before reading the whole prompt closes the
            // pipe. A write to a closed pipe is not a reason to abandon the
            // turn: the child's own output says what went wrong, and
            // reporting the write failure instead would hide it.
            let _ = stdin.write_all(prompt.as_bytes()).await;
            let _ = stdin.shutdown().await;
        });

        Ok((child, AbortOnDrop(feed), private_files))
    }
}

impl CompletionModel for ClaudeCodeModel {
    type Response = CliResponse;
    type StreamingResponse = CliResponse;
    type Client = ClaudeCodeClient;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        client.configure(Self::new(model))
    }

    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
        let spec = request::build(&self.model, &request, request::Mode::Blocking)?;
        let bridge = self.start_bridge(&request).await?;
        let (mut child, feed, _private_files) = self.spawn_child(&spec, bridge.as_ref())?;

        let stdout = require_pipe(child.stdout.take(), "stdout")?;
        let stderr = require_pipe(child.stderr.take(), "stderr")?;
        let stderr_drain = Drain::start(stderr, STDERR_LIMIT);
        let stdout_drain = Drain::start(stdout, STDOUT_LIMIT);

        let binary = self.binary.clone();
        let turn = async move {
            let status = child.wait().await.map_err(|error| {
                CompletionError::ProviderError(format!("waiting for `{binary}`: {error}"))
            })?;
            drop(feed);

            // The envelope is usually complete the moment the child exits.
            // Taking it here avoids waiting out the grace period when a
            // grandchild is holding the pipe open, which is the whole reason
            // the grace exists.
            if let Some(envelope) = response::find_envelope(&stdout_drain.snapshot()) {
                return settle(&binary, envelope, status, bridge.as_ref());
            }
            let truncated = stdout_drain.truncated();
            let stdout_bytes = stdout_drain.finish(STDOUT_GRACE).await;

            // The envelope comes first, whatever the exit status. The CLI
            // reports a usage limit, a rate limit, or an unrecognized model as
            // a well-formed envelope on stdout *and* exit 1, with stderr empty
            // or holding only an internal code. Checking the status first
            // would discard the only readable explanation.
            if let Some(envelope) = response::find_envelope(&stdout_bytes) {
                return settle(&binary, envelope, status, bridge.as_ref());
            }

            let stderr_text = stderr_drain.finish(STDERR_GRACE).await;
            if !status.success() {
                return Err(child_failed(&binary, status, &stderr_text));
            }
            if truncated {
                // Without this the half-read buffer fails to parse and the
                // error reads as a protocol break rather than a size limit.
                return Err(CompletionError::ResponseError(format!(
                    "claude produced more than {STDOUT_LIMIT} bytes of output, \
                     which was truncated before it could be parsed"
                )));
            }
            Err(CompletionError::ResponseError(format!(
                "unparseable claude output; received: {}",
                response::quote(&stdout_bytes)
            )))
        };

        // The deadline covers the whole turn, grace periods included, so
        // `with_timeout` means what it says. Dropping `child` on that path
        // kills it, because the spawn set `kill_on_drop`.
        match self.timeout {
            Some(limit) => tokio::time::timeout(limit, turn)
                .await
                .map_err(|_| timed_out(&self.binary, limit))?,
            None => turn.await,
        }
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
    /// be started. A failure that becomes visible mid-stream surfaces as an
    /// error item within the stream: a non-zero exit, a failed turn, a stream
    /// that ends without its terminal frame, or a turn that outruns the
    /// timeout.
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        let spec = request::build(&self.model, &request, request::Mode::Streaming)?;
        let bridge = self.start_bridge(&request).await?;
        let (mut child, feed, private_files) = self.spawn_child(&spec, bridge.as_ref())?;

        let stdout = require_pipe(child.stdout.take(), "stdout")?;
        let stderr = require_pipe(child.stderr.take(), "stderr")?;
        let stderr_drain = Drain::start(stderr, STDERR_LIMIT);

        let program = self.binary.clone();
        let timeout = self.timeout;
        let frames = async_stream::stream! {
            // Both are held for the whole stream: the private files must
            // outlive the child that reads them, and the feed task must not
            // be cancelled before the prompt is written.
            let _private_files = private_files;
            let _feed = feed;
            // Held for the whole stream so the CLI can reach the bridge until
            // the turn ends.
            let bridge = bridge;
            let mut stopped_early = false;
            // On a turn that advertises tools, text is held back until the
            // terminal frame says whether the CLI made any calls. rig folds
            // every yielded text delta into the committed history, and the
            // text a model writes after the placeholder is about the
            // placeholder ("Let me wait for the result..."), which the next
            // turn would then read as its own past self. The blocking path
            // discards that text; this path must too, and it can only decide
            // once the calls are known.
            let mut held: Vec<RawStreamingChoice<CliResponse>> = Vec::new();
            let holding = bridge.is_some();

            let mut lines = streaming::Lines::new(stdout);
            let mut saw_terminal_frame = false;
            // The child's exit is the other way this loop can end. Waiting for
            // end-of-file alone tracks whatever still holds the pipe open, such
            // as an MCP server or a telemetry flush. A CLI that dies without
            // its terminal frame would then never surface the failure.
            let mut exit: Option<std::io::Result<std::process::ExitStatus>> = None;
            let mut after_exit: Option<tokio::time::Instant> = None;
            // `Instant + Duration` panics on overflow, so `with_timeout(Duration::MAX)`
            // would panic here. That is a plausible way to spell "no timeout".
            // A deadline that cannot be represented is no deadline, which is
            // what the caller meant and what the blocking path already does.
            let deadline = timeout.and_then(|limit| tokio::time::Instant::now().checked_add(limit));

            loop {
                // Once the child is gone, only what is already in the pipe can
                // still arrive, so the read gets a short budget rather than the
                // turn's.
                let next_deadline = match after_exit {
                    Some(at) => Some(deadline.map_or(at, |turn| turn.min(at))),
                    None => deadline,
                };

                let read = if exit.is_some() {
                    read_next(&mut lines, next_deadline).await
                } else {
                    tokio::select! {
                        line = read_next(&mut lines, next_deadline) => line,
                        status = child.wait() => {
                            exit = Some(status);
                            after_exit = tokio::time::Instant::now().checked_add(STDOUT_GRACE);
                            continue;
                        }
                    }
                };

                let line = match read {
                    Err(Deadline) if exit.is_some() => {
                        // The child is gone. If the *turn's* deadline is what
                        // fired, say so; a timeout reported as "no terminal
                        // frame" sends the reader after the wrong problem.
                        // Otherwise the post-exit grace has run out, and
                        // whatever still holds the pipe open is not going to
                        // add to this turn.
                        let turn_expired = deadline
                            .is_some_and(|at| tokio::time::Instant::now() >= at);
                        if turn_expired {
                            yield Err(timed_out(&program, timeout.unwrap_or_default()));
                            return;
                        }
                        break;
                    }
                    Err(Deadline) => {
                        // Returning drops `child`, which kills it: the spawn
                        // set `kill_on_drop`.
                        yield Err(timed_out(&program, timeout.unwrap_or_default()));
                        return;
                    }
                    Ok(Err(error)) => {
                        yield Err(CompletionError::ResponseError(format!(
                            "reading output from `{program}`: {error}"
                        )));
                        return;
                    }
                    Ok(Ok(None)) => break,
                    Ok(Ok(Some(line))) => line,
                };

                match streaming::classify(&line) {
                    streaming::Event::Emit(choice) if holding => held.push(choice),
                    streaming::Event::Emit(choice) => yield Ok(choice),
                    streaming::Event::Finish(result) => {
                        saw_terminal_frame = true;
                        for choice in finish_items(bridge.as_ref(), std::mem::take(&mut held), *result) {
                            yield Ok(choice);
                        }
                        // The terminal frame ends the turn. Reading on to
                        // end-of-file would wait for every process holding the
                        // pipe open, including a grandchild the CLI left
                        // running, and a second terminal frame would tear the
                        // result: usage from the first, identity from the
                        // second.
                        stopped_early = true;
                        break;
                    }
                    streaming::Event::Fail(error) => {
                        yield Err(error);
                        return;
                    }
                    streaming::Event::Skip => {}
                }
            }

            // Stopping the line reader must not stop the pipe being read: a
            // child with more than a pipe-buffer of trailing output blocks in
            // `write` forever, and `child.wait()` never returns.
            let tail = stopped_early.then(|| Drain::start(lines.into_reader(), 0));

            // End of output with no terminal frame: release whatever was
            // held so the caller at least sees it. (With a terminal frame,
            // `finish_items` already drained it.)
            for choice in std::mem::take(&mut held) {
                yield Ok(choice);
            }

            let Some(status) = await_exit(exit, &mut child, deadline).await else {
                // Returning drops `child`, which kills it.
                yield Err(timed_out(&program, timeout.unwrap_or_default()));
                return;
            };
            let stderr_text = drain_within(deadline, tail, stderr_drain).await;

            if let Some(error) = conclude(&program, status, saw_terminal_frame, &stderr_text) {
                yield Err(error);
            }
        };

        Ok(StreamingCompletionResponse::stream(Box::pin(frames)))
    }
}

/// Reconcile a parsed envelope with the child's exit status.
///
/// A failed envelope explains itself and wins regardless of status. A
/// successful envelope beside a non-zero exit is not a success: the CLI wrote
/// something envelope-shaped and then failed, and the streaming path already
/// reports that as an error, so the blocking path must agree rather than
/// return the answer as if the turn were clean.
fn settle(
    binary: &str,
    envelope: CliResponse,
    status: std::process::ExitStatus,
    bridge: Option<&Bridge>,
) -> Result<CompletionResponse<CliResponse>, CompletionError> {
    if let Some(failure) = envelope.failure() {
        return Err(failure);
    }
    if !status.success() {
        return Err(child_failed(
            binary,
            status,
            b"a successful envelope was produced first",
        ));
    }
    let mut response = envelope.into_completion_response()?;
    // Recorded tool calls replace the turn's text. The model wrote that text
    // against placeholder results, and rig is about to run the real calls and
    // ask again; returning the text as well would hand rig an answer built on
    // nothing.
    if let Some(calls) = bridge
        .map(Bridge::take_calls)
        .filter(|calls| !calls.is_empty())
    {
        let content: Vec<_> = calls.into_iter().map(RecordedCall::into_content).collect();
        response.choice = OneOrMany::many(content)
            .map_err(|_| CompletionError::ResponseError("recorded calls vanished".to_owned()))?;
    }
    Ok(response)
}

/// Finish the stream's drains, keeping them inside the turn's deadline.
///
/// The grace periods sit inside the deadline too. Otherwise a turn whose child
/// exited on time could still overrun the stated bound by however long a
/// grandchild keeps the pipes open, up to the grace. The README promises the
/// bound covers the whole turn. Whatever stderr arrived is returned either
/// way; a deadline that cuts the drain short costs at most the tail of an
/// error message.
async fn drain_within(
    deadline: Option<tokio::time::Instant>,
    tail: Option<Drain>,
    stderr: Drain,
) -> Vec<u8> {
    let drains = async {
        if let Some(tail) = tail {
            let _ = tail.finish(STDERR_GRACE).await;
        }
        stderr.finish(STDERR_GRACE).await
    };
    match deadline {
        Some(at) => tokio::time::timeout_at(at, drains)
            .await
            .unwrap_or_default(),
        None => drains.await,
    }
}

/// What a finished stream amounts to, once the child's status is known.
///
/// `None` is a clean turn. Everything else is the error item the stream ends
/// with.
fn conclude(
    program: &str,
    status: std::io::Result<std::process::ExitStatus>,
    saw_terminal_frame: bool,
    stderr_text: &[u8],
) -> Option<CompletionError> {
    match status {
        Ok(status) if !status.success() => Some(child_failed(program, status, stderr_text)),
        Ok(_) if !saw_terminal_frame => Some(CompletionError::ResponseError(format!(
            "`{program}` closed the stream without a terminal result frame"
        ))),
        Ok(_) => None,
        // `ResponseError`, not `ProviderError`: see `child_failed`. The OS
        // error text is not ours to trust either.
        Err(error) => Some(CompletionError::ResponseError(format!(
            "waiting for `{program}`: {error}"
        ))),
    }
}

/// The child's exit status, waited for under the turn's deadline.
///
/// `exit` is already known when the child's exit ended the read loop. The
/// deadline covers this wait too. It used to stop applying once the loop
/// broke, so a child that emitted its result and then took its time exiting
/// held the caller past the stated limit and then reported success.
async fn await_exit(
    exit: Option<std::io::Result<std::process::ExitStatus>>,
    child: &mut Child,
    deadline: Option<tokio::time::Instant>,
) -> Option<std::io::Result<std::process::ExitStatus>> {
    match exit {
        Some(status) => Some(status),
        None => match deadline {
            Some(at) => tokio::time::timeout_at(at, child.wait()).await.ok(),
            None => Some(child.wait().await),
        },
    }
}

/// The items a stream yields when its terminal frame arrives.
///
/// The session id first. Then, if the CLI made tool calls, the calls and
/// nothing else: text the model wrote on that turn is dropped, as the
/// blocking path drops it, because it was written against the placeholder.
/// Otherwise the held text, in order. The final response last either way.
///
/// Draining the calls at the terminal frame is not a race. The bridge records
/// a call before it replies, the CLI waits for the reply before it asks the
/// model again, and the terminal frame follows the model's last message. So
/// every call of the turn is recorded before the terminal frame is written.
fn finish_items(
    bridge: Option<&Bridge>,
    held: Vec<RawStreamingChoice<CliResponse>>,
    result: CliResponse,
) -> Vec<RawStreamingChoice<CliResponse>> {
    let mut items = Vec::new();
    if let Some(id) = result.session_id.clone() {
        items.push(RawStreamingChoice::MessageId(id));
    }
    let calls = bridge.map(Bridge::take_calls).unwrap_or_default();
    if calls.is_empty() {
        items.extend(held);
    } else {
        items.extend(
            calls
                .into_iter()
                .map(|call| RawStreamingChoice::ToolCall(call.into_raw())),
        );
    }
    items.push(RawStreamingChoice::FinalResponse(result));
    items
}

/// The deadline passed before a line arrived.
struct Deadline;

/// Read the next line, honoring `deadline` when there is one.
async fn read_next<R>(
    lines: &mut streaming::Lines<R>,
    deadline: Option<tokio::time::Instant>,
) -> Result<std::io::Result<Option<String>>, Deadline>
where
    R: tokio::io::AsyncRead + Unpin,
{
    match deadline {
        Some(at) => tokio::time::timeout_at(at, lines.next_line())
            .await
            .map_err(|_| Deadline),
        None => Ok(lines.next_line().await),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn exit_with(code: i32) -> std::process::ExitStatus {
        use std::os::unix::process::ExitStatusExt as _;
        std::process::ExitStatus::from_raw(code << 8)
    }

    #[test]
    fn a_good_envelope_with_a_failed_exit_is_not_a_success() {
        let envelope = crate::response::CliResponse {
            result: Some("the answer".to_owned()),
            ..Default::default()
        };
        let error = settle("claude", envelope, exit_with(3), None).unwrap_err();
        assert!(error.to_string().contains("exited with"), "{error}");
    }

    #[test]
    fn a_failed_envelope_explains_itself_whatever_the_exit() {
        let envelope = crate::response::CliResponse {
            is_error: true,
            subtype: "error_max_turns".to_owned(),
            ..Default::default()
        };
        let error = settle("claude", envelope, exit_with(0), None).unwrap_err();
        assert!(error.to_string().contains("error_max_turns"), "{error}");
    }

    #[test]
    fn a_good_envelope_with_a_clean_exit_is_the_answer() {
        let envelope = crate::response::CliResponse {
            result: Some("the answer".to_owned()),
            ..Default::default()
        };
        assert!(settle("claude", envelope, exit_with(0), None).is_ok());
    }

    #[test]
    fn a_clean_exit_with_a_terminal_frame_concludes_nothing() {
        assert!(conclude("claude", Ok(exit_with(0)), true, b"").is_none());
    }

    #[test]
    fn a_clean_exit_without_a_terminal_frame_is_reported() {
        let error = conclude("claude", Ok(exit_with(0)), false, b"").unwrap();
        assert!(
            error
                .to_string()
                .contains("without a terminal result frame"),
            "{error}"
        );
    }

    #[test]
    fn a_failed_exit_quotes_stderr_whatever_the_frames_said() {
        let error = conclude("claude", Ok(exit_with(3)), true, b"it broke").unwrap();
        let rendered = error.to_string();
        assert!(rendered.contains("exited with"), "{rendered}");
        assert!(rendered.contains("it broke"), "{rendered}");
    }

    #[test]
    fn a_wait_failure_is_reported() {
        let failure = std::io::Error::other("no such process");
        let error = conclude("claude", Err(failure), true, b"").unwrap();
        assert!(
            error.to_string().contains("waiting for `claude`"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn a_drain_records_that_it_truncated() {
        let input = std::io::Cursor::new(vec![b'x'; 5000]);
        let drain = Drain::start(input, 100);
        let kept = {
            let flag = Arc::clone(&drain.truncated);
            let bytes = drain.finish(Duration::from_secs(5)).await;
            assert!(
                flag.load(Ordering::Relaxed),
                "5000 bytes into 100 is a truncation"
            );
            bytes
        };
        assert_eq!(kept.len(), 100);
    }

    #[tokio::test]
    async fn a_drain_under_budget_does_not_claim_truncation() {
        let input = std::io::Cursor::new(b"short".to_vec());
        let drain = Drain::start(input, 100);
        let flag = Arc::clone(&drain.truncated);
        drain.finish(Duration::from_secs(5)).await;
        assert!(!flag.load(Ordering::Relaxed));
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

    #[test]
    fn carries_no_timeout_by_default() {
        assert_eq!(ClaudeCodeModel::new("haiku").timeout(), None);
    }

    #[test]
    fn takes_a_timeout() {
        let model = ClaudeCodeModel::new("haiku").with_timeout(Duration::from_secs(3));
        assert_eq!(model.timeout(), Some(Duration::from_secs(3)));
    }

    #[test]
    fn accumulates_extra_arguments_in_order() {
        let model = ClaudeCodeModel::new("haiku")
            .with_args(["--add-dir", "/tmp"])
            .with_mcp_config("/etc/mcp.json");
        assert_eq!(
            model.extra_args,
            vec!["--add-dir", "/tmp", "--mcp-config", "/etc/mcp.json"]
        );
        assert!(model.check_extra_args().is_ok());
    }

    #[test]
    fn takes_a_working_directory() {
        let model = ClaudeCodeModel::new("haiku").with_current_dir("/srv");
        assert_eq!(model.current_dir.as_deref(), Some("/srv"));
    }

    #[test]
    fn rejects_extra_arguments_that_override_the_request() {
        for hostile in [
            vec!["--model", "opus"],
            vec!["--output-format", "text"],
            vec!["--system-prompt-file", "/tmp/other"],
            vec!["--tools", "Bash"],
        ] {
            let model = ClaudeCodeModel::new("haiku").with_args(hostile.clone());
            let error = model.check_extra_args().unwrap_err().to_string();
            assert!(
                error.contains("is set by rig-claude-code"),
                "{hostile:?} -> {error}"
            );
        }
    }

    #[test]
    fn rejects_an_owned_flag_in_its_equals_form() {
        let model = ClaudeCodeModel::new("haiku").with_args(["--model=opus"]);
        assert!(model.check_extra_args().is_err());
    }

    #[test]
    fn names_the_timeout_in_the_error() {
        let error = timed_out("claude", Duration::from_millis(250)).to_string();
        assert!(error.contains("250ms"), "{error}");
        assert!(error.contains("was killed"), "{error}");
    }

    #[tokio::test]
    async fn a_drain_keeps_only_its_budget() {
        let input = std::io::Cursor::new(vec![b'x'; 5000]);
        let kept = Drain::start(input, 100)
            .finish(Duration::from_secs(5))
            .await;
        assert_eq!(kept.len(), 100, "the surplus is drained and discarded");
    }

    #[tokio::test]
    async fn a_drain_keeps_everything_under_budget() {
        let input = std::io::Cursor::new(b"short".to_vec());
        let kept = Drain::start(input, 100)
            .finish(Duration::from_secs(5))
            .await;
        assert_eq!(kept, b"short");
    }
}
