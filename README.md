# rig-claude-code

[![CI](https://github.com/cjohnhanson/rig-claude-code/actions/workflows/ci.yml/badge.svg)](https://github.com/cjohnhanson/rig-claude-code/actions/workflows/ci.yml)

A [rig](https://github.com/0xPlaygrounds/rig) model provider that runs the
Claude Code CLI. Turns draw on a Claude subscription, not on API credits.

**Unofficial.** This crate is not affiliated with Anthropic or with
0xPlaygrounds, who maintain rig.

## Contents

1. [Who this is for](#who-this-is-for)
2. [Install and run the first turn](#install-and-run-the-first-turn)
3. [Configure the client](#configure-the-client)
4. [Stream a response](#stream-a-response)
5. [Get structured output](#get-structured-output)
6. [Give the model tools](#give-the-model-tools)
7. [Handle a failed turn](#handle-a-failed-turn)
8. [Reference: what the transport does not do](#reference-what-the-transport-does-not-do)
9. [Reference: environment and trust](#reference-environment-and-trust)
10. [Explanation: why the invocation looks the way it does](#explanation-why-the-invocation-looks-the-way-it-does)
11. [Test](#test)
12. [Compatibility](#compatibility)
13. [Terms of service and license](#terms-of-service-and-license)

## Who this is for

Use this crate when you want rig's agent, RAG, and memory features, and you
want each turn billed to a Claude subscription. It suits a small number of
sequential turns on a machine that has an interactive Claude Code login.

Use rig's own Anthropic provider instead when you need sampling parameters,
high concurrency, streamed structured output, or a server with no logged-in
CLI.

Anthropic's help centre states that `claude -p` usage draws on a
subscription's usage limits ([Use Claude Code with your Pro or Max
plan][cc-plan]). That statement held for Claude Code 2.1.233 in August 2026.
Check it before you rely on it.

[cc-plan]: https://support.claude.com/en/articles/11145838-use-claude-code-with-your-pro-or-max-plan

## Install and run the first turn

1. Install Claude Code. Run `claude` once and log in. The crate needs a
   `claude` binary on `PATH` that is already logged in.

2. Add the dependencies. The crate is not on crates.io yet.

   ```toml
   [dependencies]
   rig-claude-code = { git = "https://github.com/cjohnhanson/rig-claude-code" }
   rig = "0.41"
   tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
   ```

3. Build a client, build an agent, and prompt it.

   ```rust,no_run
   use std::time::Duration;
   use rig::completion::Prompt;
   use rig::prelude::*;
   use rig_claude_code::{ClaudeCodeClient, models};

   #[tokio::main]
   async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

`from_env` runs `claude --version` on the calling thread to confirm the binary
works. Call it during setup. `ClaudeCodeClient::new` followed by
`version().await` does the same check without holding the thread. Neither call confirms
the CLI is logged in. Only a real turn confirms that; a logged-out CLI fails
the first turn with a refusal envelope (see [Handle a failed
turn](#handle-a-failed-turn)).

Each turn starts a Node process. On one machine against Claude Code 2.1.233,
a one-word `haiku` turn took 2.3 to 2.7 seconds wall clock against about one
second of model time. Expect about 1.5 seconds of process overhead per turn.

The client implements `ProviderClient`, `CompletionClient`, and
`VerifyClient`. `client.agent(..)` comes from rig's `AgentClientExt`, which the
`rig` crate's default `agent` feature provides.

## Configure the client

Every setting lives on the client. Each agent the client builds inherits them.

```rust,no_run
use std::time::Duration;
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, models};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = ClaudeCodeClient::from_env()?
    .with_timeout(Duration::from_secs(120))
    .with_current_dir("/srv/workspace")
    .with_args(["--max-budget-usd", "0.50"]);

let agent = client.agent(models::SONNET).build();
# let _ = agent;
# Ok(())
# }
```

| Setting | Effect |
| --- | --- |
| `with_timeout` | Kills the child and fails the turn after this long. There is no default; without one, a wedged child waits forever. |
| `with_current_dir` | Runs the child in this directory. |
| `with_args` | Appends arguments to the CLI invocation. An argument that collides with a flag the crate sets is refused. |
| `with_mcp_config` | Passes an MCP server configuration. See [Give the model tools](#give-the-model-tools). |
| `with_binary` (on `ClaudeCodeModel`) | Runs a specific binary instead of the one on `PATH`. |

`RIG_CLAUDE_CODE_BIN` names the binary for `from_env` when it is not on
`PATH`. `ClaudeCodeModel` carries the same `with_*` methods for building a
model directly. `rig_claude_code::Client` and `rig_claude_code::CompletionModel`
are aliases for the two types, under the names the rig ecosystem uses.

`models` exports the aliases `HAIKU`, `SONNET`, `OPUS`, and `FABLE`. Each one
tracks the latest model in its family. Pass a full model id, such as
`claude-haiku-4-5-20251001`, to pin a turn to one version. The CLI's help
lists `fable`, `opus`, and `sonnet`. `haiku` is accepted but not listed, so it
is the alias most likely to change.

## Stream a response

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

This example needs the `futures` crate. `examples/shipping_forecast.rs` runs
`prompt`, `chat`, and `stream_prompt` against a real binary.

## Get structured output

Structured output works through `prompt` and `prompt_typed`, which return
the whole answer at once. It does not work through `stream_prompt`; see the
end of this section. Set no output mode: the default `OutputMode::Auto`
resolves to native structured output when the agent has no tools. An agent
with tools uses `OutputMode::Tool`, which the crate does not support (see the
reference table).

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

This example needs the `serde` and `schemars` crates. `cargo run --example
typed_output` runs it against a real binary.

Do not stream a schema-bearing agent. The CLI enforces a schema in a second
internal turn, so a streamed turn yields the model's first, unconstrained
answer as text. The enforced JSON arrives only in the terminal frame.

## Give the model tools

Register rig tools on the agent as you would with any provider. rig runs
them; the CLI is only the model.

```rust,no_run
use rig::completion::Prompt;
use rig::prelude::*;
use rig::tool::{Tool, ToolContext};
use rig_claude_code::{ClaudeCodeClient, models};
use serde::Deserialize;

#[derive(Deserialize)]
struct Args { sku: String }

#[derive(Debug, thiserror::Error)]
#[error("lookup failed")]
struct LookupError;

struct LookupPrice;

impl Tool for LookupPrice {
    const NAME: &'static str = "lookup_price";
    type Args = Args;
    type Output = String;
    type Error = LookupError;
    fn description(&self) -> String { "The price of a product, by SKU.".into() }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"]})
    }
    async fn call(&self, _: &mut ToolContext, args: Args) -> Result<String, LookupError> {
        Ok(format!("price of {}: $14.20", args.sku))
    }
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let agent = ClaudeCodeClient::from_env()?
    .agent(models::HAIKU)
    .tool(LookupPrice)
    .default_max_turns(4)
    .build();
println!("{}", agent.prompt("How much is SKU A-113?").await?);
# Ok(())
# }
```

This example needs the `serde` and `thiserror` crates. `cargo run --example
tool_loop` runs it against a real binary.

Set `default_max_turns` to at least two. rig's default is one model turn, and
a tool call needs one turn to ask and one to answer with the result in hand.

### How the loop works

The CLI is an agent harness of its own. When its model emits a tool call, the
harness looks the tool up, and if it finds none it tells the model the tool
does not exist. So the crate cannot describe tools in the prompt and read the
calls back from the text: the model reaches for a real tool call, the harness
rejects it, and the model reports the tool as broken.

Instead, for each turn that carries rig tools, the crate binds a loopback
HTTP listener, serves MCP on it with rig's tool definitions, and points the
CLI at it. The server executes nothing. It records each call the CLI makes
and answers with an acknowledgement. The turn's response then carries the
recorded calls, and rig runs its own tool implementations, appends the
results, and asks again. Tool calls and results appear in the next turn's
prompt in full, so the model sees what it asked and what came back.

The consequences for a caller:

- Tools run in your process, with rig's hooks and permissions, exactly as with
  any other provider.
- Each turn that carries tools opens a loopback port for its duration. Every
  local process can reach that port, so the crate mints a random bearer token
  per turn, hands it to the CLI in a 0600 file it reads as its MCP
  configuration, and answers `401` to any request without it. The token is
  not in the child's argument vector, so `ps` does not show it. A recorded
  call therefore came from a process that could read this user's files, in
  practice this turn's CLI.
- The model's text on a turn that made calls is discarded, since it was
  written before any result existed. rig's runner asks again.
- `tool_choice` `Auto` and `None` are honored. `Required` and `Specific` are
  refused: the CLI's harness decides whether the model calls a tool.
- A tool's `parameters` must be a JSON Schema object with `"type": "object"`
  at the top level. The turn fails with a `RequestError` that names the tool
  otherwise. The CLI drops a tool whose schema it expects the API to reject,
  such as `{}` or one with a top-level `oneOf`, and reports nothing the crate
  can see; the model then answers as if the tool did not exist. The check
  turns that silent failure into a loud one before any usage is spent.

`with_mcp_config` still passes an MCP server of your own to the CLI, and its
tools run inside the CLI as before. The two mechanisms coexist. The crate's
own server is named `rig`; a server of that name in your configuration is
shadowed on any turn that carries rig tools.

## Handle a failed turn

A usage limit, a rate limit, a logged-out CLI, and an unrecognized model each
produce an envelope on the CLI's standard output. The crate treats an envelope
as a failure when its `is_error` flag is set, or when its `subtype` begins with
`error`. The two fields do not always agree; a usage-limit envelope has been
seen with `is_error: true` and `subtype: "success"`.

The crate returns the whole envelope as `CompletionError::ProviderResponse`.
Branch on the envelope, not on message text:

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

Through an agent, the error arrives as a `PromptError` that wraps the
`CompletionError`. Match `PromptError::CompletionError(inner)` first, or walk
`Error::source`.

A refused request setting arrives as `CompletionError::RequestError` with an
`UnsupportedSetting` as its source. The rustdoc for `UnsupportedSetting` shows
the source walk.

## Reference: what the transport does not do

The CLI takes a prompt, a system prompt, a model, and an output schema. It
takes no sampling parameters. Tools reach it through a per-turn MCP server
(see [Give the model tools](#give-the-model-tools)). The crate refuses each
unsupported request setting with an error. It never drops one silently.

| Setting | Why the crate refuses it |
| --- | --- |
| `temperature`, `max_tokens` | The CLI has no such flags. |
| `additional_params` | The CLI has no request body to extend. |
| `tool_choice` of `Required` or `Specific` | The CLI's harness decides whether the model calls a tool. `Auto` and `None` are honored. |
| `OutputMode::Tool` and `ExtractorBuilder` | Both need a forced call to a synthetic output tool, which is `tool_choice: Required`. Use `prompt_typed` on an agent without tools instead. |
| A last message with no text | The CLI needs a prompt. A transcript with nothing after it is answered as if it were the question. |
| An output schema over 96 KiB | The schema is passed as one argument, and Linux caps an argument at 128 KiB. |

Message content is text. The crate replaces an image, audio, video, document
block, or tool result with a visible placeholder such as `[an image was
omitted: this transport sends text only]`, so the model is told what was left
out.

The CLI takes one prompt, so the crate renders prior turns into it as a
labelled transcript rather than replaying them. Every marker in the transcript
carries a per-request nonce, and message text therefore cannot forge a turn or
close a section early. A `Message::System` anywhere in the history moves into
the system prompt. A trailing `Message::System` is sent both as the system
prompt and as the prompt.

Each turn is a fresh process. The `session_id` the CLI reports appears as
rig's `message_id` for observability only; feeding it back resumes nothing.

A `prompt` turn keeps at most 16 MiB of output and reports an overflow as a
size limit. A `stream_prompt` turn has no total cap, but one frame over 16 MiB
with no newline fails the stream with a read error.

Dropping the future or stream that drives a turn kills the child, so an
abandoned turn does not keep spending usage. The kill reaches the child and
not its descendants; an MCP server the CLI started can outlive the turn.
Concurrent turns are concurrent processes.

The crate is tested on Unix. Windows is untested and not covered by CI.

## Reference: environment and trust

The binary runs with the host process's full privileges. `PATH` and
`RIG_CLAUDE_CODE_BIN` are trusted configuration; either one names code to
execute. Prefer an absolute path. `from_env` runs the binary at once, so
constructing a client executes whatever the variable names.

The child inherits the environment except for the ten variables a live Claude
Code session exports to mark itself: `CLAUDECODE`, `CLAUDE_CODE_SESSION_ID`,
`CLAUDE_CODE_CHILD_SESSION`, `CLAUDE_CODE_ENTRYPOINT`, `CLAUDE_CODE_EXECPATH`,
`CLAUDE_CODE_MESSAGING_SOCKET`, `CLAUDE_CODE_MESSAGING_TOKEN`, `CLAUDE_PID`,
`CLAUDE_EFFORT`, and `AI_AGENT`. The crate removes these by exact name.
`CLAUDE_EFFORT` would otherwise change the effort level and cost of every turn
according to who launched the host process.

Every other variable the CLI honors reaches the child. Several of them select
the credential or the endpoint, and the crate leaves them alone on purpose:

| Variable | Effect on the turn |
| --- | --- |
| `CLAUDE_CONFIG_DIR` | Selects which account's login is used. |
| `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX` | Route the turn to a different backend. |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` | Bill an API account instead of the subscription. |
| `ANTHROPIC_BASE_URL` | Send every prompt to a different endpoint. |
| `ANTHROPIC_MODEL` | Override the model. |

## Explanation: why the invocation looks the way it does

### Why these flags

A plain `claude -p` loads the full Claude Code agent system prompt, the
project's `CLAUDE.md`, configured MCP servers, and skills. Against Claude Code
2.1.233, a one-word prompt cost about 42,000 input tokens that way. The crate
passes `--tools ""`, `--strict-mcp-config`, `--setting-sources ""`, and
`--disable-slash-commands`, and the same prompt then costs about 165 tokens.
`--bare` is not in that list. It forces authentication through
`ANTHROPIC_API_KEY` and never reads the subscription credential.

### Why the prompt is on standard input and the system prompt is in a file

Neither is passed as an argument, for three reasons, each verified against
2.1.233.

1. Injection. The CLI parses any argument that begins with `-` as an option,
   and it splits `--flag=value` on the first `=`. A prompt of
   `--settings={"hooks":…"command":"touch /tmp/proof"…}` executed that
   command before any API call. The CLI also scans raw argv for `--settings=`
   before its own option parsing, so the payload fires from an option value
   too, and an end-of-options `--` does not stop it.
2. Length. Linux caps one argument at 128 KiB. A flattened transcript exceeds
   that and fails with `E2BIG`, an error that resembles a missing binary.
   macOS allows more, so the failure appears only on Linux.
3. Exposure. Arguments are readable through `/proc` and `ps`. For a RAG
   agent, that would expose every retrieved document.

The system prompt file is created with mode 0600 and passed as
`--system-prompt-file`. That flag is not in the CLI's option list in 2.1.233;
it appears only inside the description of `--bare`. It works, and the crate
depends on it.

### Why the schema loses its `$schema` key

schemars stamps every schema with a pointer to the 2020-12 metaschema. The
CLI's validator cannot resolve that URI and rejects the schema. The crate
removes that one key and nothing else.

### Why the envelope is read before the exit status

The CLI reports a usage limit, a rate limit, or an unrecognized model as an
envelope on standard output together with exit code 1, and its standard error
is often empty. The crate reads the envelope first so the explanation is never
lost. A successful envelope beside a failed exit is still a failure.

## Test

```console
cargo test                 # spends no usage
cargo clippy --all-targets
cargo llvm-cov             # coverage
```

The integration suite drives a scripted stand-in for the `claude` binary. It
asserts on the real spawned argument vector, on what crossed standard input,
on the child's environment, on exit-status handling, and on process lifetime.
The stand-in reproduces shapes a naive double cannot: output that arrives in
stages, a flood of standard error before any frames, a grandchild that holds
pipes open after the child exits, and a child abandoned mid-turn. Those tests
are Unix-only because the stand-in is a shell script.

`cargo run --example shipping_forecast` and `cargo run --example
typed_output` run against a real binary and spend usage.

## Compatibility

Built against rig 0.41 and Claude Code 2.1.233. rig ships breaking changes on
most minor releases, so pin it. The CLI's flags and output shape also change;
the response types keep unknown fields, and they accept a `null` or fractional
value where a number is expected.

The MSRV is 1.88, set by this crate's use of let chains on edition 2024.
rig-core declares no MSRV of its own.

`ProviderClient::Error` is this crate's `ClientError`, not rig's
`ProviderClientError`. Generic code bounded on the latter will not accept this
client.

## Terms of service and license

This crate runs the `claude` binary and parses its documented output. It
performs no authentication bypass. Follow Anthropic's terms of service and
usage policies, including the rules on what may authenticate with a Claude
subscription. Anthropic does not permit a third-party developer to offer
users access through the developer's own subscription credentials.

MIT.
