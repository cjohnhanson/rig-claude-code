# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ClaudeCodeClient` and `ClaudeCodeModel`: a rig completion model that runs
  the local `claude` CLI in print mode, so the credential is the one Claude
  Code is already logged in with rather than an API key.
- Blocking completion and streaming. Streaming reads the CLI's newline-
  delimited JSON frames and emits text and reasoning deltas.
- Structured output through the CLI's `--json-schema` flag.
- Rejection, with a named error, of every request setting the CLI cannot
  express: tool definitions, `temperature`, `max_tokens`, and
  `additional_params`.

[Unreleased]: https://github.com/cjohnhanson/rig-claude-code
