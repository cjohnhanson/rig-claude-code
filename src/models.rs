//! Model aliases the CLI accepts for `--model`.
//!
//! Each alias tracks the latest model in its family, so an agent built with
//! one moves forward as Anthropic ships new versions. Pass a full model id
//! instead when a turn must stay pinned:
//!
//! ```
//! use rig_claude_code::{ClaudeCodeModel, models};
//!
//! let tracking = ClaudeCodeModel::new(models::HAIKU);
//! let pinned = ClaudeCodeModel::new("claude-haiku-4-5-20251001");
//! # let _ = (tracking, pinned);
//! ```

/// The fastest and cheapest family.
pub const HAIKU: &str = "haiku";

/// The balanced family.
pub const SONNET: &str = "sonnet";

/// The most capable family.
pub const OPUS: &str = "opus";

/// The lightweight family.
pub const FABLE: &str = "fable";
