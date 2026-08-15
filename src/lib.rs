//! A [rig](https://github.com/0xPlaygrounds/rig) completion model backed by
//! the Claude Code CLI.
//!
//! rig's Anthropic provider authenticates with an API key. This crate does
//! not: it runs the local `claude` binary in print mode, so the credential is
//! whatever Claude Code is already logged in with. Anthropic's help centre
//! states that `claude -p` usage draws on a subscription's usage limits, so on
//! a subscription login no API credits are spent — provided nothing in the
//! environment overrides that. `ANTHROPIC_API_KEY` in the parent environment
//! makes the CLI bill an API account instead, and this crate does not remove
//! it. True of Claude Code 2.1.233 in August 2026; check before relying on it.
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
//! token accounting, native structured output through the CLI's
//! `--json-schema`, and streaming with token-level text and reasoning deltas.
//! rig-level features that need nothing from a provider — conversation memory,
//! agent hooks — work here as they do anywhere. (Those are rig's hooks; the
//! CLI's own hooks are deliberately not loaded.)
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
//! - **A request whose last message carries no text.** The CLI needs a
//!   prompt, and a transcript with nothing after it is answered as if it were
//!   the question.
//! - **An output schema over 96 KiB.** It is the one caller-sized value still
//!   passed as an argument, and Linux caps one at 128 KiB.
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
//! it is dropped — the child, not its descendants. There is no default
//! timeout; see [`ClaudeCodeClient::with_timeout`]. A turn needs a tokio
//! runtime with I/O and time enabled.
//!
//! A failed turn — usage limit, rate limit, unrecognized model — is returned
//! as [`rig_core::completion::CompletionError::ProviderResponse`] with the
//! CLI's whole envelope as the body, so a caller can branch on its `subtype`
//! through `provider_response_json` rather than on message text.

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

/// The rig ecosystem's conventional name for a provider's client.
///
/// Companion crates such as `rig-fastembed` expose `Client`, so that is what a
/// rig user reaches for first. [`ClaudeCodeClient`] is the same type under the
/// name that reads better when imported on its own.
pub use client::ClaudeCodeClient as Client;

/// The rig ecosystem's conventional name for a provider's completion model.
pub use model::ClaudeCodeModel as CompletionModel;
pub use request::UnsupportedSetting;
pub use response::{CliResponse, CliUsage, OutputTokenDetails};
