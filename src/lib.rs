//! A [rig](https://github.com/0xPlaygrounds/rig) completion model backed by
//! the Claude Code CLI.
//!
//! rig's Anthropic provider authenticates with an API key. This crate does
//! not: it runs the local `claude` binary in print mode, so the credential is
//! whatever Claude Code is already logged in with. For a subscription login,
//! Anthropic's documentation states that `claude -p` usage draws on the
//! subscription's usage limits, so no API credits are spent.
//!
//! The crate implements the ordinary provider contracts, so it composes like
//! any built-in provider:
//!
//! ```no_run
//! use rig::completion::Prompt;
//! use rig::prelude::*;
//! use rig_claude_code::ClaudeCodeClient;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ClaudeCodeClient::from_env()?;
//! let agent = client
//!     .agent("haiku")
//!     .preamble("You are terse.")
//!     .build();
//!
//! println!("{}", agent.prompt("Say hello.").await?);
//! # Ok(())
//! # }
//! ```
//!
//! # What works
//!
//! Preambles and system messages, context documents, multi-turn history,
//! conversation memory, hooks, token accounting, native structured output
//! through the CLI's `--json-schema`, and streaming with token-level text and
//! reasoning deltas.
//!
//! # What the transport cannot do
//!
//! The CLI takes a prompt, a system prompt, a model, and an output schema. It
//! takes no tool definitions and no sampling parameters. Each unsupported
//! setting is **rejected with an error** rather than dropped, because a
//! silently ignored `max_tokens` is harder to diagnose than a refused request:
//!
//! - **Tools.** Registering rig tools fails. The CLI's route to tools is MCP;
//!   an agent that needs them wants a different transport, or an MCP server
//!   and `--mcp-config`. This also rules out `OutputMode::Tool`, and so the
//!   [`ExtractorBuilder`](https://docs.rs/rig-core/latest/rig/) path, which
//!   uses a synthetic `submit` tool. Use `OutputMode::Native` on a plain
//!   agent for structured output instead.
//! - **`temperature` and `max_tokens`.** No CLI flags exist.
//! - **`additional_params`.** There is no request body to extend.
//!
//! # Cost of the default invocation
//!
//! A plain `claude -p` loads the full Claude Code agent system prompt, the
//! project's `CLAUDE.md`, configured MCP servers, and skills. Measured against
//! Claude Code 2.1.233, a one-word prompt costs about 42,000 input tokens that
//! way. This crate passes the flags that strip all of it, which brings the
//! same prompt to about 165.

#![forbid(unsafe_code)]

mod client;
mod model;
mod request;
mod response;
mod streaming;

pub use client::{BINARY_ENV, ClaudeCodeClient, ClientError, DEFAULT_BINARY};
pub use model::ClaudeCodeModel;
pub use response::{CliResult, CliUsage, OutputTokenDetails};
