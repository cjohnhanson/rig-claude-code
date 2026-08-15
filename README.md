# rig-claude-code

[![CI](https://github.com/cjohnhanson/rig-claude-code/actions/workflows/ci.yml/badge.svg)](https://github.com/cjohnhanson/rig-claude-code/actions/workflows/ci.yml)

A [rig](https://github.com/0xPlaygrounds/rig) completion model backed by the
Claude Code CLI.

**Unofficial.** Not affiliated with or endorsed by Anthropic or by
0xPlaygrounds, who maintain rig and its `rig-*` integration crates.

rig's Anthropic provider authenticates with an API key. This one does not. It
runs the local `claude` binary in print mode, so the credential is whatever
Claude Code is already logged in with. Anthropic's help centre states that
`claude -p` usage draws on a subscription's usage limits — see [Use Claude Code
with your Pro or Max plan][cc-plan] — so on a subscription login, no API
credits are spent. That was true of Claude Code 2.1.233 in August 2026; check
it yourself before relying on it.

[cc-plan]: https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan

**Reach for this** when you want rig's agent, RAG, and memory plumbing with
turns drawn from a Claude subscription rather than API credits, for a modest
number of sequential turns on a machine that has an interactive Claude Code
login. **Reach for rig's own Anthropic provider instead** for tool-calling
agents, sampling control, high concurrency, or any server without a logged-in
CLI.

**Prerequisite:** a `claude` binary on `PATH` that is already logged in.
Install Claude Code and run `claude` once interactively to log in.

```rust,no_run
use std::time::Duration;
use rig::completion::Prompt;
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, models};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `from_env` runs `claude --version` synchronously to confirm the binary
    // works. Call it during setup, not on a hot async path.
    let client = ClaudeCodeClient::from_env()?
        .with_timeout(Duration::from_secs(120));

    let agent = client
        .agent(models::HAIKU)
        .preamble("You are terse.")
        .build();

    println!("{}", agent.prompt("Say hello.").await?);
    Ok(())
}
```

The client implements the same traits as a built-in provider —
`ProviderClient`, `CompletionClient`, `VerifyClient` — so
`client.completion_model(..)` and `client.verify()` work directly, and
`client.agent(..)` works through rig's `AgentClientExt`, which the `rig`
crate's default `agent` feature brings in.

Every setting lives on the client and is inherited by each agent it builds:

```rust,no_run
use std::time::Duration;
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, models};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = ClaudeCodeClient::from_env()?
    .with_timeout(Duration::from_secs(120))
    .with_mcp_config("/etc/agent/mcp.json")
    .with_current_dir("/srv/workspace");

let agent = client.agent(models::HAIKU).build();
# let _ = agent;
# Ok(())
# }
```

`ClaudeCodeModel` carries the same `with_*` methods for building a model
directly. `rig_claude_code::Client` and `rig_claude_code::CompletionModel`
alias the two types under the names the rig ecosystem uses.

## Install

Not yet on crates.io. Until it is, depend on the repository:

```toml
[dependencies]
rig-claude-code = { git = "https://github.com/cjohnhanson/rig-claude-code" }
rig = "0.41"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

The streaming example below also needs `futures`; the structured-output
example needs `serde` and `schemars`.

`ClaudeCodeClient::from_env` reads `RIG_CLAUDE_CODE_BIN` to find the binary
somewhere other than `PATH`; `ClaudeCodeModel::with_binary` is the equivalent
when building a model directly. `from_env` and `verify()` confirm the binary
runs — they cannot confirm it is logged in, which only a real turn
establishes. A logged-out CLI fails the first turn with a refusal envelope
(see *Failed turns*). `from_env` blocks the calling thread while
`claude --version` runs; `ClaudeCodeClient::new` plus `version().await` is the
non-blocking equivalent.

A turn needs a tokio runtime with I/O and time enabled, which
`#[tokio::main]` and `Runtime::new()` both provide. Each turn starts a Node
process. Measured on one machine against 2.1.233, a one-word `haiku` turn took
2.3–2.7 s wall clock against about 1 s of model time: roughly 1.5 s of
per-turn process overhead, which is the largest practical difference from an
HTTP provider.

## Choosing a model

The string goes to the CLI's `--model`. `rig_claude_code::models` exports the
aliases `HAIKU`, `SONNET`, `OPUS`, and `FABLE`, each tracking the latest model
in its family. A full model id such as `claude-haiku-4-5-20251001` pins a turn
to one version. The CLI's own help advertises `fable`, `opus`, and `sonnet`;
`haiku` is accepted too — verified by a real turn — but being undocumented it
is the one most likely to change.

## Streaming

```rust,no_run
use futures::StreamExt as _;
use rig::agent::MultiTurnStreamItem;
use rig::prelude::*;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use rig_claude_code::{ClaudeCodeClient, models};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let agent = ClaudeCodeClient::from_env()?.agent(models::HAIKU).build();
let mut stream = agent.stream_prompt("Count to five.").await;

while let Some(item) = stream.next().await {
    if let MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(chunk)) =
        item?
    {
        print!("{}", chunk.text);
    }
}
# Ok(())
# }
```

`examples/shipping_forecast.rs` runs the blocking, multi-turn, and streaming
paths against a real binary.

## Structured output

The CLI has a `--json-schema` flag, so native structured output works and needs
no special mode. `OutputMode::Auto`, the default, resolves to `Native` for an
agent with no tools — and this transport carries no tools (see *What the
transport cannot do*), so it always does.

```rust,no_run
use rig::completion::TypedPrompt;
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, models};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u8,
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let agent = ClaudeCodeClient::from_env()?.agent(models::HAIKU).build();
let person: Person = agent.prompt_typed("Describe Ada Lovelace.").await?;
# let _ = (person.name, person.age);
# Ok(())
# }
```

## What the transport cannot do

The CLI takes a prompt, a system prompt, a model, and an output schema. It
takes no tool definitions and no sampling parameters. Every unsupported
*request setting* is rejected with an error rather than dropped, because a
silently ignored `max_tokens` is harder to diagnose than a refused request. The
cause is a `rig_claude_code::UnsupportedSetting`, so a caller can `downcast_ref`
and fall back to another provider instead of matching on message text.

| Setting | Why not |
| --- | --- |
| Tools | The CLI accepts no tool definitions as arguments. Its route to tools is MCP: `with_mcp_config` passes a server config, and those tools stay under the CLI's control. This also rules out `OutputMode::Tool` and rig's `ExtractorBuilder`, both of which use a synthetic output tool. |
| `temperature`, `max_tokens` | No flags exist. |
| `additional_params` | There is no request body to extend. |
| Non-`None` `tool_choice` | No tools are advertised, so any other choice asks for a call that cannot happen. |
| A last message with no text | The CLI needs a prompt, and a transcript with nothing after it is answered as if it were the question. |
| An output schema over 96 KiB | It is the one caller-sized value still passed as an argument, and Linux caps one at 128 KiB. |

**Message content is text.** Images, audio, video, document blocks, and tool
results cannot cross the prompt, so each is replaced by a visible placeholder —
`[an image was omitted: this transport sends text only]` — rather than dropped.
A model told that a picture was omitted behaves very differently from one shown
a message with a hole in it.

**History is flattened, not replayed.** The CLI takes one prompt, so prior
turns are rendered into it as a labelled transcript. Every marker — the section
tags, the `user`/`assistant` labels, each document's wrapper — carries a
per-request nonce, so message text cannot forge a turn or close a section
early. Two consequences worth knowing: a `Message::System` anywhere in the
history is hoisted into the system prompt, so an instruction injected after
turn three arrives as if it had been there from the start; and a trailing
`Message::System` is sent as the system prompt *and* as the prompt. Each turn
is a fresh process. The `session_id` the CLI reports is surfaced as rig's
`message_id` for observability only; feeding it back does not resume anything.

## Failed turns

A usage limit, a rate limit, a logged-out CLI, and an unrecognized model all
arrive as a well-formed envelope on stdout, often alongside exit 1. The
envelope is a failure when *either* its `is_error` flag is set *or* its
`subtype` begins with `error` — the two are not always set together, and a
usage-limit envelope has been seen with `is_error: true` and `subtype:
"success"`. The crate reads the envelope before the exit status so the
explanation is never lost, and returns it as
`CompletionError::ProviderResponse` with the whole envelope as the body. Branch
on the envelope rather than on message text — and check both fields, as the
crate does:

```rust,no_run
use rig::completion::CompletionError;

# fn handle(error: &CompletionError) {
if let Ok(Some(body)) = error.provider_response_json() {
    match body.get("subtype").and_then(|s| s.as_str()) {
        Some("error_max_turns") => { /* raise the budget and retry */ }
        Some(other) => eprintln!("claude refused: {other}"),
        None => {}
    }
}
# }
```

Through an agent the error is a `PromptError` wrapping the `CompletionError`;
match `PromptError::CompletionError(inner)` first, or walk `Error::source`.
The same applies to `UnsupportedSetting`, whose rustdoc shows the walk.

## Why the invocation looks the way it does

A plain `claude -p` loads the full Claude Code agent system prompt, the
project's `CLAUDE.md`, configured MCP servers, and skills. Measured against
Claude Code 2.1.233, a one-word prompt costs about 42,000 input tokens that
way. This crate passes `--tools ""`, `--strict-mcp-config`,
`--setting-sources ""`, and `--disable-slash-commands`, which brings the same
prompt to about 165.

`--bare` looks like it belongs in that list and does not: it forces
authentication through `ANTHROPIC_API_KEY` and never reads the subscription
credential, which would defeat the point.

Neither the prompt nor the system prompt is passed as an argument. The prompt
goes to standard input; the system prompt goes to a 0600 temporary file named
by `--system-prompt-file`. Three reasons, all verified against 2.1.233:

1. **Injection.** Any argv element beginning with `-` is parsed as an option,
   and `--flag=value` splits on the first `=`. A prompt of
   `--settings={"hooks":…"command":"touch /tmp/proof"…}` executed that command
   before any API call. The CLI also scans raw argv for `--settings=` ahead of
   its own parsing, so the payload fires from an option *value* too, and an
   end-of-options `--` does not stop it.
2. **Length.** Linux caps a single argument at 128 KiB. A flattened transcript
   passes that easily and fails with `E2BIG`, which reads like a missing
   binary — on Linux only, so it never shows up on a developer's Mac.
3. **Exposure.** Argv is world-readable through `/proc` and `ps`. For a RAG
   agent that would expose every retrieved document.

`CLAUDECODE` is removed from the child's environment, so a `claude` launched
from inside a Claude Code session does not treat itself as nested.

## Process lifetime

Every turn is one child process, killed when the future or stream driving it is
dropped — so an abandoned turn does not leave a `claude` running and spending
the login's usage. That kill reaches the child, not its descendants: an MCP
server the CLI started may outlive the turn. There is no default timeout; set
`with_timeout` on the client or model to bound a turn that never finishes on
its own. The bound covers the whole turn, including the short grace periods
spent draining pipes after the child exits.

A blocking turn retains at most 16 MiB of output and reports an overflow as a
size limit, not a parse failure. A streaming turn has no total cap; a single
frame over 16 MiB with no newline fails the stream with a read error.

Concurrent turns are concurrent processes. Each one starts a Node runtime, so
the practical ceiling is much lower than for an HTTP provider.

## Tools through MCP

`with_mcp_config` is the route to tools, and the invocation pairs it with
`--strict-mcp-config` so nothing else is loaded. A non-interactive `-p` run
only calls a tool the caller has pre-approved, so name them through
`with_args`:

```rust,no_run
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, models};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = ClaudeCodeClient::from_env()?
    .with_mcp_config("/etc/agent/mcp.json")
    .with_args(["--allowedTools", "mcp__search__query"]);
let agent = client.agent(models::SONNET).build();
# let _ = agent;
# Ok(())
# }
```

The crate's `--tools ""` disables the CLI's *built-in* tools only; MCP tools
are governed by the config and the allow-list. They stay under the CLI's
control — rig never sees them, never observes their calls, and registering a
rig tool on the agent still fails.

## Trust model

The binary runs with the host process's full privileges, so both `PATH` and
`RIG_CLAUDE_CODE_BIN` are **trusted configuration**, equivalent to naming code
to execute. Prefer an absolute path. Note that `from_env` runs the binary
immediately, so constructing a client executes whatever the variable names.

The child inherits the environment, minus the ten variables a live Claude Code
session exports to mark itself (`CLAUDECODE`, `CLAUDE_CODE_SESSION_ID`,
`CLAUDE_CODE_MESSAGING_TOKEN`, `CLAUDE_EFFORT`, and the rest — matched by exact
name, listed in `src/model.rs`). `CLAUDE_EFFORT` in particular would silently
change the effort level and cost of every turn depending on who launched the
host process. Everything else the CLI honors reaches the child, on purpose,
because it is the caller's to control — including the variables that *select
the credential*: `CLAUDE_CONFIG_DIR` (which account's login is used),
`CLAUDE_CODE_USE_BEDROCK` / `CLAUDE_CODE_USE_VERTEX` (a different backend),
`ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` (bill an API account rather than
the subscription), `ANTHROPIC_BASE_URL` (send every prompt elsewhere), and
`ANTHROPIC_MODEL`.

Supported on Unix. Windows is untested and not covered by CI.

## Testing

```console
cargo test          # unit, integration, and doc tests — spends no usage
cargo llvm-cov      # coverage
cargo clippy --all-targets
```

The integration suite drives a scripted stand-in for the `claude` binary, so it
asserts on the real spawned argument vector, what crossed standard input, the
child's environment, exit-status handling, and process lifetime — without
spending any usage. It reproduces the shapes a naive double cannot: output that
arrives in stages, a flood of standard error ahead of the frames, a grandchild
that outlives the child holding its pipes open, and a child abandoned
mid-turn. Those tests are Unix-only because the stand-in is a shell script.

`cargo run --example shipping_forecast` exercises a real binary and does spend
usage.

## Compatibility

Built against rig 0.41 and Claude Code 2.1.233. rig ships breaking changes on
most minor releases, so treat the version pin as load-bearing. The MSRV of
1.88 is set by this crate's own use of let chains on edition 2024; rig-core
declares no MSRV of its own, so its floor can move independently. One trait
detail: `ProviderClient::Error` is this crate's `ClientError` rather than rig's
`ProviderClientError`, so generic code bounded on the latter will not take this
client. The
CLI's flags and output shape are likewise a moving target; the response types
keep unknown fields rather than rejecting them, and a `null` or fractional
value where a number is expected is tolerated rather than fatal.

One flag deserves naming: `--system-prompt-file`, which carries the injection
defence, is not in `claude --help`'s option list in 2.1.233 — it appears only
inside the `--bare` description. It works, but losing it would take the crate
out rather than degrade it.

## Terms of service

This crate shells out to the `claude` binary and parses its documented output.
It performs no authentication bypass and no circumvention of Anthropic's terms.
Anyone using it is expected to follow both the letter and the spirit of
Anthropic's terms of service and usage policies, including the rules on what
may authenticate with a Claude subscription — in particular, Anthropic does not
permit third-party developers to offer their users access through someone
else's subscription credentials.

## License

MIT.
