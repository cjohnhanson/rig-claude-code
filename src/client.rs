//! The provider client.

use std::process::Stdio;

use rig_core::client::ProviderClient;
use rig_core::client::completion::CompletionClient;

use crate::model::ClaudeCodeModel;

/// Environment variable naming the `claude` binary when it is not on `PATH`.
pub const BINARY_ENV: &str = "RIG_CLAUDE_CODE_BIN";

/// The binary name used when nothing else names one.
pub const DEFAULT_BINARY: &str = "claude";

/// Failure constructing a [`ClaudeCodeClient`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ClientError {
    /// The named binary could not be run.
    #[error("cannot run the claude binary `{binary}`: {source}")]
    BinaryNotRunnable {
        /// The binary that was tried.
        binary: String,
        /// Why running it failed.
        source: std::io::Error,
    },
}

/// A provider client for the local `claude` CLI.
///
/// It implements the same traits as any built-in rig provider, so
/// construction reads the same way:
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
/// There is no API key. The credential is whatever the `claude` CLI is
/// already logged in with, which for a subscription login means the turn draws
/// on the subscription's usage limits rather than API credits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaudeCodeClient {
    binary: String,
}

impl ClaudeCodeClient {
    /// Use a specific binary path, without consulting the environment and
    /// without checking that it runs.
    #[must_use]
    pub fn new(binary: impl Into<String>) -> Self {
        Self {
            binary: binary.into(),
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
    /// executed.
    pub async fn version(&self) -> Result<String, ClientError> {
        let output = tokio::process::Command::new(&self.binary)
            .arg("--version")
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|source| ClientError::BinaryNotRunnable {
                binary: self.binary.clone(),
                source,
            })?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }
}

impl ProviderClient for ClaudeCodeClient {
    /// The path to a `claude` binary.
    type Input = String;
    type Error = ClientError;

    /// Resolve the binary from [`BINARY_ENV`], falling back to `PATH`, and
    /// confirm it runs.
    ///
    /// Unlike an API-key provider there is no secret to read. The check that
    /// earns this method its name is that the binary exists at all. Without
    /// it, a missing binary surfaces one prompt later as a spawn failure
    /// inside a completion, far from the configuration mistake that caused it.
    ///
    /// # Errors
    ///
    /// Returns [`ClientError::BinaryNotRunnable`] when the resolved binary
    /// cannot be executed.
    fn from_env() -> Result<Self, Self::Error> {
        let binary = std::env::var(BINARY_ENV).unwrap_or_else(|_| DEFAULT_BINARY.to_owned());
        std::process::Command::new(&binary)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|source| ClientError::BinaryNotRunnable {
                binary: binary.clone(),
                source,
            })?;
        Ok(Self { binary })
    }

    /// Build a client for the binary at `input`, without checking that it runs.
    ///
    /// # Errors
    ///
    /// Never fails; the signature is fixed by the trait.
    fn from_val(input: Self::Input) -> Result<Self, Self::Error> {
        Ok(Self { binary: input })
    }
}

impl CompletionClient for ClaudeCodeClient {
    type CompletionModel = ClaudeCodeModel;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_binary_it_was_given() {
        let client = ClaudeCodeClient::new("/opt/claude");
        assert_eq!(client.binary(), "/opt/claude");
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
    fn reports_an_unrunnable_binary_by_name() {
        let error = ClientError::BinaryNotRunnable {
            binary: "/nowhere/claude".to_owned(),
            source: std::io::Error::from(std::io::ErrorKind::NotFound),
        };
        assert!(error.to_string().contains("/nowhere/claude"), "{error}");
    }
}
