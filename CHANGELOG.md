# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First release. Everything below describes the initial surface rather than a
change from a previous version.

### Added

- `ClaudeCodeClient` and `ClaudeCodeModel`: a rig completion model that runs
  the local `claude` CLI in print mode, so the credential is the one Claude
  Code is already logged in with rather than an API key. The client implements
  `ProviderClient`, `CompletionClient`, and `VerifyClient`, so it composes like
  any built-in rig provider.
- Blocking completion and streaming. Streaming reads the CLI's newline-
  delimited JSON frames and emits token-level text and reasoning deltas.
- Native structured output through the CLI's `--json-schema`.
- `UnsupportedSetting`: a public, downcastable cause for every request setting
  the transport cannot express — tool definitions, `temperature`, `max_tokens`,
  `additional_params`, and any `tool_choice` other than `None`. Each is
  rejected rather than silently dropped.
- `models`: the `HAIKU`, `SONNET`, `OPUS`, and `FABLE` aliases.
- `with_timeout`, `with_args`, `with_mcp_config`, and `with_current_dir` on
  both `ClaudeCodeClient` and `ClaudeCodeModel`. Settings on the client are
  inherited by every model and agent it builds. Extra arguments that collide
  with a flag the crate sets are refused rather than silently overriding it.
- `Client` and `CompletionModel`, the names the rig ecosystem uses for a
  provider's types, as aliases.
- A failed turn — usage limit, rate limit, unrecognized model — is returned
  as `CompletionError::ProviderResponse` carrying the CLI's whole envelope,
  so a caller can branch on its `subtype`.
- Refusal, up front, of a request whose last message renders to no text and
  of an output schema over 96 KiB — the one caller-sized value still passed
  as a command-line argument.

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
- Error messages quote a bounded prefix of the child's output, and both pipes
  are read with a byte cap, so a broken child cannot exhaust memory or fill a
  log line. Output past the cap is reported as a size limit, not a parse
  failure.
- Every marker in the flattened transcript — the section tags, the role
  labels, each document's wrapper — carries a per-request nonce keyed by a
  per-process salt, so message content can neither close a section early nor
  forge a turn.
- Every Claude Code session marker (`CLAUDECODE`, `CLAUDE_*`, `AI_AGENT`) is
  stripped from the child's environment. A live session exports a messaging
  token and `CLAUDE_EFFORT` among others; the latter would silently change
  the effort and cost of every turn.

[Unreleased]: https://github.com/cjohnhanson/rig-claude-code
