//! A [rig](https://github.com/0xPlaygrounds/rig) model provider that runs the
//! Claude Code CLI. Turns draw on a Claude subscription, not on API credits.
//!
//! rig's Anthropic provider authenticates with an API key. This crate runs
//! the local `claude` binary in print mode instead, so the credential is the
//! one Claude Code is already logged in with. Anthropic's help centre states
//! that `claude -p` usage draws on a subscription's usage limits. That held
//! for Claude Code 2.1.233 in August 2026; check it before you rely on it.
//! `ANTHROPIC_API_KEY` in the parent environment makes the CLI bill an API
//! account instead, and this crate does not remove it.
//!
//! The client implements rig's provider traits, so it composes like a
//! built-in provider:
//!
//! ```no_run
//! use rig::completion::Prompt;
//! use rig::prelude::*;
//! use rig_claude_code::{ClaudeCodeClient, models};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = ClaudeCodeClient::from_env()?;
//! let agent = client
//!     .agent(models::HAIKU)
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
//! token accounting, streaming with token-level text and reasoning deltas,
//! and native structured output through the CLI's `--json-schema` for
//! `prompt` and `prompt_typed`, though not for `stream_prompt`. rig features
//! that need nothing from a provider, such as
//! conversation memory and agent hooks, work unchanged. Those are rig's hooks;
//! the crate does not load the CLI's own hooks.
//!
//! # Tools
//!
//! Register rig tools on the agent as with any provider; rig runs them and
//! the CLI is only the model. For each turn that carries tools the crate
//! serves them to the CLI over a per-turn loopback MCP server, records the
//! calls the CLI makes, and returns them as `AssistantContent::ToolCall` for
//! rig's runner to execute. Set `default_max_turns` to at least two.
//!
//! # What the transport does not do
//!
//! The CLI takes a prompt, a system prompt, a model, and an output schema. It
//! takes no sampling parameters. The crate refuses each unsupported request
//! setting with an error, and never drops one silently. The error's source
//! is an [`UnsupportedSetting`], so a caller can branch on the cause and fall
//! back to another provider.
//!
//! - `temperature` and `max_tokens`. The CLI has no such flags.
//! - `additional_params`. The CLI has no request body to extend.
//! - A `tool_choice` of `Required` or `Specific`. The CLI's harness decides
//!   whether the model calls a tool. This also rules out `OutputMode::Tool`
//!   and rig's extractor, which force a call to a synthetic output tool.
//! - A last message with no text. The CLI needs a prompt.
//! - An output schema over 96 KiB. The schema is passed as one argument, and
//!   Linux caps an argument at 128 KiB.
//!
//! Message content is text. The crate replaces an image, audio clip, video,
//! document block, or tool result with a visible placeholder, so the model is
//! told what was left out.
//!
//! # Process lifetime and errors
//!
//! Each turn is one child process. Dropping the future or stream that drives
//! a turn kills the child, not its descendants. There is no default timeout;
//! see [`ClaudeCodeClient::with_timeout`]. A turn needs a tokio runtime with
//! I/O and time enabled.
//!
//! A failed turn returns
//! [`rig_core::completion::CompletionError::ProviderResponse`] with the CLI's
//! whole envelope as the body. Branch on the envelope's `subtype` through
//! `provider_response_json`, not on message text.
//!
//! The README covers configuration, the trust model, and why the invocation
//! is shaped as it is.

#![forbid(unsafe_code)]

mod bridge;
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
