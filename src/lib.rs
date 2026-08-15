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
//! takes no tool definitions and no sampling parameters. Every unsupported
//! *request setting* is rejected with an error rather than dropped, because a
//! silently ignored `max_tokens` is harder to diagnose than a refused request.
//! The cause is an [`UnsupportedSetting`], so a caller can `downcast_ref` and
//! fall back to another provider rather than match on message text.
//!
//! - **Tools.** Registering rig tools fails. The CLI's route to tools is MCP,
//!   reachable through [`ClaudeCodeModel::with_mcp_config`]. That also rules
//!   out `OutputMode::Tool` and rig's extractor, both of which use a synthetic
//!   `submit` tool — use a plain agent, whose default `OutputMode::Auto`
//!   already resolves to native structured output when no tools are present.
//! - **`temperature` and `max_tokens`.** No CLI flags exist.
//! - **`additional_params`.** There is no request body to extend.
//! - **A `tool_choice` other than `None`.** No tools are advertised.
//!
//! Message content is text. An image, audio clip, video, document block, or
//! tool result is replaced by a visible placeholder rather than dropped
//! silently, so the model knows something was left out.
//!
//! # Cost, safety, and process lifetime
//!
//! A plain `claude -p` loads the full Claude Code agent system prompt, the
//! project's `CLAUDE.md`, configured MCP servers, and skills — about 42,000
//! input tokens for a one-word prompt, against about 165 with the flags this
//! crate passes. Neither the prompt nor the system prompt travels in argv; see
//! the `request` module's documentation for why, and the README for the trust
//! model around `PATH` and `RIG_CLAUDE_CODE_BIN`.
//!
//! Every turn is one child process, killed when the future or stream driving
//! it is dropped. There is no default timeout; see
//! [`ClaudeCodeModel::with_timeout`].

#![forbid(unsafe_code)]

mod client;
mod model;
pub mod models;
mod request;
mod response;
mod streaming;

/// Compiles the README's examples as doctests, so they cannot rot.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme {}

pub use client::{BINARY_ENV, ClaudeCodeClient, ClientError, DEFAULT_BINARY};
pub use model::ClaudeCodeModel;
pub use request::UnsupportedSetting;
pub use response::{CliResponse, CliUsage, OutputTokenDetails};
