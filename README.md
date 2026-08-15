# rig-claude-code

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

```rust
use rig::completion::Prompt;
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, models};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeCodeClient::from_env()?;

    let agent = client
        .agent(models::HAIKU)
        .preamble("You are terse.")
        .build();

    println!("{}", agent.prompt("Say hello.").await?);
    Ok(())
}
```

The client implements the ordinary provider traits, so `client.agent(..)`,
`client.completion_model(..)`, and `client.verify()` compose exactly as they do
for a built-in provider.

## Install

```toml
[dependencies]
rig-claude-code = "0.1"
rig = "0.41"
```

It needs a `claude` binary on `PATH` that is already logged in.
`ClaudeCodeClient::from_env` reads `RIG_CLAUDE_CODE_BIN` to find it elsewhere;
`ClaudeCodeModel::with_binary` is the equivalent when building a model
directly.

## Choosing a model

The string goes to the CLI's `--model`. `rig_claude_code::models` exports the
aliases `HAIKU`, `SONNET`, `OPUS`, and `FABLE`, each tracking the latest model
in its family. A full model id such as `claude-haiku-4-5-20251001` pins a turn
to one version.

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
no special mode — `OutputMode::Auto`, the default, already resolves to
`Native` for an agent with no tools, which is always the case here.

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
| Tools | The CLI accepts no tool definitions as arguments. Its route to tools is MCP: `ClaudeCodeModel::with_mcp_config` passes a server config, and those tools stay under the CLI's control. This also rules out `OutputMode::Tool` and rig's `ExtractorBuilder`, both of which use a synthetic `submit` tool. |
| `temperature`, `max_tokens` | No flags exist. |
| `additional_params` | There is no request body to extend. |
| Non-`None` `tool_choice` | No tools are advertised, so any other choice asks for a call that cannot happen. |

**Message content is text.** Images, audio, video, document blocks, and tool
results cannot cross the prompt, so each is replaced by a visible placeholder —
`[an image was omitted: this transport sends text only]` — rather than dropped.
A model told that a picture was omitted behaves very differently from one shown
a message with a hole in it.

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
the login's usage. There is no default timeout: set
`ClaudeCodeModel::with_timeout` to bound a turn that never finishes on its own.

Concurrent turns are concurrent processes. Each one starts a Node runtime, so
the practical ceiling is much lower than for an HTTP provider.

## Trust model

The binary runs with the host process's full privileges, so both `PATH` and
`RIG_CLAUDE_CODE_BIN` are **trusted configuration**, equivalent to naming code
to execute. Prefer an absolute path. Note that `from_env` runs the binary
immediately, so constructing a client executes whatever the variable names.

The child inherits the environment except `CLAUDECODE`. Two variables in the
parent environment can quietly change what the crate does: `ANTHROPIC_API_KEY`
makes the CLI bill an API account rather than the subscription, and
`ANTHROPIC_BASE_URL` sends every prompt to a different endpoint.

Supported on Unix. Windows is untested and not covered by CI.

## Testing

```console
cargo test          # unit, integration, and doc tests
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
most minor releases, so treat the version pin as load-bearing. The CLI's flags
and output shape are likewise a moving target; the response types keep unknown
fields rather than rejecting them, and a `null` or fractional value where a
number is expected is tolerated rather than fatal.

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
