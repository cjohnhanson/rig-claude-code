# rig-claude-code

A [rig](https://github.com/0xPlaygrounds/rig) completion model backed by the
Claude Code CLI.

rig's Anthropic provider authenticates with an API key. This one does not. It
runs the local `claude` binary in print mode, so the credential is whatever
Claude Code is already logged in with. For a subscription login, Anthropic's
documentation states that `claude -p` usage draws on the subscription's usage
limits, so no API credits are spent.

```rust
use rig::completion::Prompt;
use rig::prelude::*;
use rig_claude_code::ClaudeCodeClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeCodeClient::from_env()?;

    let agent = client
        .agent("haiku")
        .preamble("You are terse.")
        .build();

    println!("{}", agent.prompt("Say hello.").await?);
    Ok(())
}
```

The client implements the ordinary provider traits, so `client.agent(..)`
composes exactly as it does for a built-in provider.

## Install

```toml
[dependencies]
rig-claude-code = "0.1"
rig = "0.41"
```

It needs a `claude` binary on `PATH` that is already logged in. Set
`RIG_CLAUDE_CODE_BIN` to point somewhere else.

## What works

Preambles and system messages, context documents, multi-turn history,
conversation memory, hooks, token accounting, native structured output through
the CLI's `--json-schema`, and streaming with token-level text and reasoning
deltas.

## What the transport cannot do

The CLI takes a prompt, a system prompt, a model, and an output schema. It
takes no tool definitions and no sampling parameters. Every unsupported
setting is **rejected with an error** rather than dropped, because a silently
ignored `max_tokens` is harder to diagnose than a refused request.

| Setting | Why not |
| --- | --- |
| Tools | The CLI accepts no tool definitions as arguments. Its route to tools is MCP. This also rules out `OutputMode::Tool` and the extractor, which use a synthetic `submit` tool — use `OutputMode::Native` on a plain agent instead. |
| `temperature` | No flag exists. |
| `max_tokens` | No flag exists. |
| `additional_params` | There is no request body to extend. |

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

`CLAUDECODE` is removed from the child's environment, so a `claude` launched
from inside a Claude Code session does not treat itself as nested.

## Testing

```
cargo test          # unit, integration, and doc tests
cargo llvm-cov      # coverage
cargo clippy --all-targets
```

The integration suite drives a scripted stand-in for the `claude` binary, so
it asserts on the real spawned argument vector, the child's environment, and
exit-status handling without spending any usage. Those tests are Unix-only
because the stand-in is a shell script; the library is not.

`cargo run --example shipping_forecast` exercises a real binary and does spend
usage.

## Compatibility

Built against rig 0.41 and Claude Code 2.1.233. rig ships breaking changes on
most minor releases, so treat the version pin as load-bearing.

## Terms of service

This crate shells out to the `claude` binary and parses its documented output.
It performs no authentication bypass and no circumvention of Anthropic's
terms. Anyone using it is expected to follow both the letter and the spirit of
Anthropic's terms of service and usage policies, including the rules on what
may authenticate with a Claude subscription.

## License

MIT.
