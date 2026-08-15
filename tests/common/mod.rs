//! A fake `claude` binary the tests drive.
//!
//! Each instance owns a temporary directory holding the script and the files
//! that parameterise it, so tests stay independent and can run in parallel.
//! The script records what it was called with, which lets a test assert on the
//! real spawned argument vector and environment rather than on a value the
//! library merely intended to pass.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::PathBuf;

/// Written by the fake at each invocation.
const RECORD_ARGV: &str = "argv";
/// Records whether `CLAUDECODE` was present in the child's environment.
const RECORD_NESTED: &str = "nested";

/// A scripted stand-in for the `claude` binary.
pub struct FakeClaude {
    dir: tempfile::TempDir,
}

impl FakeClaude {
    /// A fake that prints `stdout` and exits zero.
    pub fn printing(stdout: &str) -> Self {
        Self::build(stdout, "", 0)
    }

    /// A fake that prints `stderr` and exits with `code`.
    pub fn failing(stderr: &str, code: i32) -> Self {
        Self::build("", stderr, code)
    }

    fn build(stdout: &str, stderr: &str, code: i32) -> Self {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path();

        fs::write(path.join("stdout"), stdout).expect("write stdout fixture");
        fs::write(path.join("stderr"), stderr).expect("write stderr fixture");
        fs::write(path.join("exit_code"), code.to_string()).expect("write exit code");

        // `$0` is the script path, so the script finds its own fixtures
        // without needing an environment variable, which keeps parallel tests
        // from interfering with each other.
        let script = r#"#!/bin/sh
here=$(dirname "$0")
printf '%s\0' "$@" > "$here/argv"
if [ -n "${CLAUDECODE+set}" ]; then echo present > "$here/nested"; else echo absent > "$here/nested"; fi
cat "$here/stdout"
cat "$here/stderr" >&2
exit "$(cat "$here/exit_code")"
"#;
        let binary = path.join("claude");
        let mut file = fs::File::create(&binary).expect("create fake binary");
        file.write_all(script.as_bytes()).expect("write script");
        drop(file);
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755))
            .expect("make the fake executable");

        Self { dir }
    }

    /// Path to the fake binary.
    pub fn path(&self) -> String {
        self.dir.path().join("claude").display().to_string()
    }

    /// The arguments the fake was last called with.
    ///
    /// # Panics
    ///
    /// Panics if the fake was never invoked.
    pub fn argv(&self) -> Vec<String> {
        let recorded = fs::read_to_string(self.dir.path().join(RECORD_ARGV))
            .expect("the fake was never invoked");
        // The script separates arguments with NUL rather than a newline:
        // arguments routinely contain newlines (any multi-turn prompt does),
        // and a newline separator would split one argument into several.
        // Every argument is followed by a NUL, so the trailing empty element
        // is an artifact of the split rather than an empty argument.
        let mut args: Vec<String> = recorded.split('\0').map(str::to_owned).collect();
        args.pop();
        args
    }

    /// The value passed immediately after `flag`, if the flag was present.
    ///
    /// # Panics
    ///
    /// Panics if the fake was never invoked.
    pub fn value_after(&self, flag: &str) -> Option<String> {
        let args = self.argv();
        let index = args.iter().position(|arg| arg == flag)?;
        args.get(index + 1).cloned()
    }

    /// Whether `CLAUDECODE` was set in the child's environment.
    ///
    /// # Panics
    ///
    /// Panics if the fake was never invoked.
    pub fn saw_nested_marker(&self) -> bool {
        let recorded = fs::read_to_string(self.dir.path().join(RECORD_NESTED))
            .expect("the fake was never invoked");
        recorded.trim() == "present"
    }

    /// A path inside the temporary directory that holds no binary.
    pub fn missing_path(&self) -> PathBuf {
        self.dir.path().join("no-such-binary")
    }
}
