//! The provider client.

use std::process::Stdio;

use rig_core::client::ProviderClient;
use rig_core::client::completion::CompletionClient;
use rig_core::client::verify::{VerifyClient, VerifyError};

use crate::model::ClaudeCodeModel;
use crate::response::quote;

/// Environment variable naming the `claude` binary when it is not on `PATH`.
///
/// Read by [`ClaudeCodeClient::from_env`] only. [`ClaudeCodeModel::new`]
/// does not consult it; use [`ClaudeCodeModel::with_binary`] there.
pub const BINARY_ENV: &str = "RIG_CLAUDE_CODE_BIN";

/// The binary name used when nothing else names one.
pub const DEFAULT_BINARY: &str = "claude";

/// Failure constructing or checking a [`ClaudeCodeClient`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// The named binary could not be started at all.
    #[error("cannot run the claude binary `{binary}`: {source}")]
    BinaryNotRunnable {
        /// The binary that was tried.
        binary: String,
        /// Why starting it failed.
        source: std::io::Error,
    },

    /// The binary started and reported failure.
    ///
    /// A path that names some other executable produces this rather than
    /// passing as a working `claude`.
    #[error("the claude binary `{binary}` exited with {status}: {stderr}")]
    BinaryFailed {
        /// The binary that was tried.
        binary: String,
        /// Its exit status.
        status: std::process::ExitStatus,
        /// A bounded quote of what it wrote to stderr.
        stderr: String,
    },
}

/// A provider client for the local `claude` CLI.
///
/// It implements the same traits as any built-in rig provider, so construction
/// reads the same way:
///
/// ```no_run
/// use rig::prelude::*;
/// use rig_claude_code::ClaudeCodeClient;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let client = ClaudeCodeClient::from_env()?;
/// let agent = client.agent("haiku").preamble("Be terse.").build();
/// # let _ = agent;
/// # Ok(())
/// # }
/// ```
///
/// There is no API key. The credential is whatever the `claude` CLI is already
/// logged in with, which for a subscription login means the turn draws on the
/// subscription's usage limits rather than API credits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaudeCodeClient {
    binary: String,
    timeout: Option<std::time::Duration>,
    extra_args: Vec<String>,
    current_dir: Option<String>,
}

impl ClaudeCodeClient {
    /// Use a specific binary path, without consulting the environment and
    /// without checking that it runs.
    ///
    /// The binary runs with this process's full privileges, so its path is
    /// trusted configuration.
    #[must_use]
    pub fn new(binary: impl Into<String>) -> Self {
        Self {
            binary: binary.into(),
            timeout: None,
            extra_args: Vec::new(),
            current_dir: None,
        }
    }

    /// Bound every turn this client's models run.
    ///
    /// Carried onto each [`ClaudeCodeModel`] the client builds, so
    /// `client.agent(..)` inherits it. Without a timeout here or on the model,
    /// a wedged child holds the caller forever.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Pass additional CLI arguments on every turn.
    ///
    /// See [`ClaudeCodeModel::with_args`] for the rules; an argument that
    /// collides with a flag the crate sets is refused when the turn runs.
    #[must_use]
    pub fn with_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Give the CLI an MCP server configuration on every turn.
    #[must_use]
    pub fn with_mcp_config(self, path: impl Into<String>) -> Self {
        self.with_args(["--mcp-config".to_owned(), path.into()])
    }

    /// Run every child in a specific working directory.
    #[must_use]
    pub fn with_current_dir(mut self, dir: impl Into<String>) -> Self {
        self.current_dir = Some(dir.into());
        self
    }

    /// Apply this client's settings to a model it built.
    pub(crate) fn configure(&self, model: ClaudeCodeModel) -> ClaudeCodeModel {
        let model = model
            .with_binary(&self.binary)
            .with_args(self.extra_args.clone());
        let model = match self.timeout {
            Some(timeout) => model.with_timeout(timeout),
            None => model,
        };
        match &self.current_dir {
            Some(dir) => model.with_current_dir(dir),
            None => model,
        }
    }

    /// The binary this client runs.
    #[must_use]
    pub fn binary(&self) -> &str {
        &self.binary
    }

    /// Run the binary with `--version` and return what it prints.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::BinaryNotRunnable`] when the binary cannot be
    /// started, and [`ClientError::BinaryFailed`] when it starts and exits
    /// non-zero — which is what a path naming some other executable does.
    pub async fn version(&self) -> Result<String, ClientError> {
        let output = tokio::process::Command::new(&self.binary)
            .arg("--version")
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|source| ClientError::BinaryNotRunnable {
                binary: self.binary.clone(),
                source,
            })?;

        if !output.status.success() {
            return Err(ClientError::BinaryFailed {
                binary: self.binary.clone(),
                status: output.status,
                stderr: quote(&output.stderr),
            });
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }
}

impl ProviderClient for ClaudeCodeClient {
    /// The path to a `claude` binary.
    type Input = String;
    type Error = ClientError;

    /// Resolve the binary from [`BINARY_ENV`], falling back to `PATH`, and
    /// confirm it runs successfully.
    ///
    /// Unlike an API-key provider there is no secret to read, so this is what
    /// the method checks instead. Without it, a missing or wrong binary
    /// surfaces one prompt later as a spawn failure inside a completion, far
    /// from the configuration mistake that caused it.
    ///
    /// This runs the binary synchronously, which parks the calling thread for
    /// as long as `claude --version` takes. The trait signature is sync, so
    /// there is no way around it; call it during setup rather than on a hot
    /// async path, or use [`ClaudeCodeClient::new`] and
    /// [`ClaudeCodeClient::version`] to do the same check without blocking.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::BinaryNotRunnable`] when the resolved binary
    /// cannot be started, and [`ClientError::BinaryFailed`] when it starts and
    /// exits non-zero.
    fn from_env() -> Result<Self, Self::Error> {
        let binary = std::env::var(BINARY_ENV).unwrap_or_else(|_| DEFAULT_BINARY.to_owned());
        let output = std::process::Command::new(&binary)
            .arg("--version")
            .stdin(Stdio::null())
            .output()
            .map_err(|source| ClientError::BinaryNotRunnable {
                binary: binary.clone(),
                source,
            })?;

        if !output.status.success() {
            return Err(ClientError::BinaryFailed {
                binary,
                status: output.status,
                stderr: quote(&output.stderr),
            });
        }

        Ok(Self::new(binary))
    }

    /// Build a client for the binary at `input`, without checking that it runs.
    ///
    /// # Errors
    ///
    /// Never fails; the signature is fixed by the trait.
    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Ok(Self::new(input))
    }
}

impl CompletionClient for ClaudeCodeClient {
    type CompletionModel = ClaudeCodeModel;
}

impl VerifyClient for ClaudeCodeClient {
    /// Check that the CLI is present and working.
    ///
    /// This is the trait generic rig code reaches for, so a caller that
    /// verifies every provider before use works here too. It confirms the
    /// binary runs; it does not confirm the CLI is logged in, which only a
    /// real turn can establish.
    ///
    /// # Errors
    ///
    /// Returns [`VerifyError::ProviderError`] when the binary is missing or
    /// exits non-zero.
    async fn verify(&self) -> Result<(), VerifyError> {
        self.version()
            .await
            .map(|_| ())
            .map_err(|error| VerifyError::ProviderError(error.to_string()))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_binary_it_was_given() {
        assert_eq!(ClaudeCodeClient::new("/opt/claude").binary(), "/opt/claude");
    }

    #[test]
    fn from_val_never_checks_the_binary() {
        let client = ClaudeCodeClient::from_val("/nowhere/claude".to_owned()).unwrap();
        assert_eq!(client.binary(), "/nowhere/claude");
    }

    #[test]
    fn builds_a_model_carrying_the_clients_binary() {
        let client = ClaudeCodeClient::new("/opt/claude");
        let model = client.completion_model("haiku");
        assert_eq!(model.binary(), "/opt/claude");
        assert_eq!(model.model(), "haiku");
    }

    #[test]
    fn builds_a_model_carrying_every_client_setting() {
        // `client.agent(..)` is the documented construction route, and it goes
        // through `completion_model`. Settings that stopped here would be
        // unreachable from the only path the README shows.
        let client = ClaudeCodeClient::new("/opt/claude")
            .with_timeout(std::time::Duration::from_secs(9))
            .with_mcp_config("/etc/mcp.json")
            .with_current_dir("/srv");
        let model = client.completion_model("haiku");

        assert_eq!(model.timeout(), Some(std::time::Duration::from_secs(9)));
        assert_eq!(model.current_dir(), Some("/srv"));
        assert_eq!(
            model.extra_args(),
            ["--mcp-config".to_owned(), "/etc/mcp.json".to_owned()]
        );
    }
}
