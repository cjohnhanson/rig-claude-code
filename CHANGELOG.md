# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Not yet published to crates.io. Everything below describes the initial
surface rather than a change from a previous version; it becomes `0.1.0` when
it ships.

### Added

The public surface, in full:

- `ClaudeCodeClient` (alias `Client`) and `ClaudeCodeModel` (alias
  `CompletionModel`): a rig completion model that runs the local `claude` CLI
  in print mode, so the credential is the one Claude Code is already logged in
  with rather than an API key. The client implements `ProviderClient`,
  `CompletionClient`, and `VerifyClient`. `ClaudeCodeClient::version` runs
  `claude --version`.
- `ClientError`, `UnsupportedSetting`, `InvalidToolSchema`, `CliResponse`, `CliUsage`,
  `OutputTokenDetails`, the `models` module, `BINARY_ENV`, `DEFAULT_BINARY`.
- Whole-answer completion through `prompt`, and streaming through
  `stream_prompt`. Streaming reads the CLI's newline-delimited JSON frames
  and emits token-level text and reasoning deltas.
- Native structured output through the CLI's `--json-schema`, for `prompt`
  and `prompt_typed`. schemars 1.x's `$schema` pointer to the 2020-12
  metaschema is stripped first: Claude Code 2.1.233 cannot resolve it and
  rejects the schema outright, so without that every `prompt_typed` turn
  failed. A streamed turn yields the model's pre-enforcement prose; the
  enforced JSON is in the terminal frame.
- rig tools, executed by rig. For each turn that carries tools the crate
  serves them to the CLI over a per-turn loopback MCP server. It records the
  calls the CLI makes and returns them as `AssistantContent::ToolCall` for
  rig's runner to execute. Tool calls and results render in full into the
  next turn's prompt. The crate allows the CLI the whole server with one
  rule, not one rule per tool. The CLI rewrites a tool name such as
  `lookup.price` to `lookup_price` in its own name for the tool, so a
  per-tool rule missed it and the model reported a missing permission. A
  tool whose `parameters` do not fit the MCP tool shape is refused with a
  `RequestError` whose cause is a downcastable `InvalidToolSchema` that
  names the tool. The shape is `"type": "object"` at the top level,
  `properties` an object if present, `required` an array of strings if
  present, no top-level `anyOf`, `oneOf`, or `allOf`, and property keys in
  `[A-Za-z0-9_.-]{1,64}`. The CLI handles each of these in silence. One tool
  outside the shape makes it load none of the tools it was given. A
  combinator or a bad key makes it rewrite or skip the tool, by remote flag.
- `UnsupportedSetting`: a public, downcastable cause for every request setting
  the transport cannot express: `temperature`, `max_tokens`,
  `additional_params`, a `tool_choice` of `Required` or `Specific`, a last
  message with no text, and an output schema over 96 KiB. Each is refused
  rather than dropped.
- `models`: the `HAIKU`, `SONNET`, `OPUS`, and `FABLE` aliases.
- `with_timeout`, `with_args`, `with_mcp_config`, and `with_current_dir` on
  both `ClaudeCodeClient` and `ClaudeCodeModel`. Settings on the client are
  inherited by every model and agent it builds. Extra arguments that collide
  with a flag the crate sets are refused rather than silently overriding it.
- `Client` and `CompletionModel`, the names the rig ecosystem uses for a
  provider's types, as aliases.
- A failed turn (a usage limit, a rate limit, an unrecognized model) is
  returned as `CompletionError::ProviderResponse` with the CLI's whole
  envelope, so a caller can branch on its `subtype`.

### Security

- Neither the prompt nor the system prompt travels in the argument vector.
  Verified against Claude Code 2.1.233: a prompt of
  `--settings={"hooks":…"command":"touch /tmp/proof"…}` executed that command
  as the host process's user, before any API call, and the CLI scans raw argv
  for `--settings=` ahead of its own option parsing, so the payload fired from
  an option *value* too. The prompt goes to standard input and the system
  prompt to a 0600 temporary file. That also lifts Linux's 128 KiB
  single-argument limit and keeps prompts and retrieved documents out of `ps`.
- The child is killed when the future or stream driving it is dropped, so an
  abandoned turn does not leave a `claude` running and spending the login's
  usage.
- Error messages quote a bounded prefix of the child's output. A `prompt`
  turn's pipes are read with a byte cap and an overflow is reported as a size
  limit rather than a parse failure; a `stream_prompt` turn caps each frame at
  16 MiB and fails the stream past that.
- A child that fails is reported as `CompletionError::ProviderResponse`, with
  its stderr as the body, never as `ProviderError`. rig's stream driver treats
  any `ProviderError` whose text contains "aborted" as a cancellation and ends
  the stream cleanly with no error item, and Node's own `AbortError` message
  is "This operation was aborted", so the CLI's most common failure text
  would have turned a failed streaming turn into an empty success.
- Every marker in the flattened transcript carries a per-request nonce keyed
  by a per-process salt: the section tags, the role labels, and each
  document's wrapper. Message content can neither close a section early nor
  forge a turn.
- The ten variables a live Claude Code session exports to mark itself are
  stripped from the child's environment, by exact name. Among them are a
  messaging token and `CLAUDE_EFFORT`, which would silently change the effort
  and cost of every turn. Variables that select the credential, such as
  `CLAUDE_CONFIG_DIR` and `ANTHROPIC_API_KEY`, are left alone on purpose.
- The per-turn tool bridge requires a bearer token. The bridge listens on
  loopback, which every local process can reach, and any of them can
  discover a listening port. Without a token any local process could post a
  `tools/call`. The crate would record it, return it as a `ToolCall`, and
  let rig execute it with the stranger's arguments. So each turn mints 256
  random bits from the OS and answers `401` to any request that does not
  present them. The CLI reads them as an `Authorization` header from a 0600
  file that holds its MCP configuration. The configuration goes through a
  file and not inline in `--mcp-config` because `ps` shows argv to the same
  local processes the token keeps out.

[Unreleased]: https://github.com/cjohnhanson/rig-claude-code
