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
- `ClaudeCodeModel::with_timeout`, `with_args`, `with_mcp_config`,
  `with_current_dir`, and `with_binary`. Extra arguments that collide with a
  flag the crate sets are refused rather than silently overriding it.

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
  log line.

[Unreleased]: https://github.com/cjohnhanson/rig-claude-code
