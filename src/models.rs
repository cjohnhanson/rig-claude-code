//! Model aliases the CLI accepts for `--model`.
//!
//! Each alias tracks the latest model in its family, so an agent built with
//! one moves forward as Anthropic ships new versions. Pass a full model id
//! instead when a turn must stay pinned.
//!
//! Claude Code 2.1.233's `--help` advertises `fable`, `opus`, and `sonnet`.
//! `haiku` is not listed there but is accepted, verified by a real turn that
//! resolved it to `claude-haiku-4-5-20251001`. Because it is undocumented it
//! is the alias most likely to change; pin the full id where that matters.
//!
//! ```
//! use rig_claude_code::{ClaudeCodeModel, models};
//!
//! let tracking = ClaudeCodeModel::new(models::HAIKU);
//! let pinned = ClaudeCodeModel::new("claude-haiku-4-5-20251001");
//! # let _ = (tracking, pinned);
//! ```

/// The Haiku family: the fastest and cheapest of the four, well suited to
/// tests and short turns. Accepted by the CLI but not advertised in its
/// `--help`; see the module docs.
pub const HAIKU: &str = "haiku";

/// The Sonnet family: the balanced choice.
pub const SONNET: &str = "sonnet";

/// The Opus family: the most capable.
pub const OPUS: &str = "opus";

/// The Fable family: the newest, and the smallest of the current generation.
/// Advertised by the CLI alongside `opus` and `sonnet`.
pub const FABLE: &str = "fable";
