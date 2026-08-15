//! A scripted stand-in for the `claude` binary.
//!
//! Each instance owns a temporary directory holding the script and the files
//! that parameterise it, so tests stay independent and can run in parallel.
//! The script records what it was called with — arguments, standard input,
//! the environment marker, the working directory, and how many times it ran —
//! which lets a test assert on what really crossed the process boundary
//! rather than on what the library intended to send.
//!
//! The builder also reproduces the awkward shapes a real CLI produces and a
//! naive double cannot: output that arrives in stages, a flood of standard
//! error ahead of the frames, a grandchild that outlives the child and holds
//! its pipes open, and a child that runs long enough to be abandoned.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use std::fs;
use std::io::Write as _;
use std::os::unix::fs::PermissionsExt as _;
use std::path::PathBuf;
use std::time::Duration;

/// Marker in a stdout fixture where the fake pauses mid-output.
pub const PAUSE: &str = "%%PAUSE%%";

/// A scripted stand-in for the `claude` binary.
pub struct FakeClaude {
    dir: tempfile::TempDir,
}

/// Builds a [`FakeClaude`].
pub struct FakeClaudeBuilder {
    stdout: String,
    stderr: String,
    exit_code: i32,
    delay_before: Option<f32>,
    delay_mid: Option<f32>,
    stderr_first: bool,
    orphan_seconds: Option<f32>,
    sentinel_after: Option<f32>,
}

impl FakeClaude {
    /// A fake that prints `stdout` and exits zero.
    pub fn printing(stdout: &str) -> Self {
        Self::builder().stdout(stdout).build()
    }

    /// A fake that prints `stderr` and exits with `code`.
    pub fn failing(stderr: &str, code: i32) -> Self {
        Self::builder().stderr(stderr).exit_code(code).build()
    }

    /// Start describing a fake.
    pub fn builder() -> FakeClaudeBuilder {
        FakeClaudeBuilder {
            stdout: String::new(),
            stderr: String::new(),
            exit_code: 0,
            delay_before: None,
            delay_mid: None,
            stderr_first: false,
            orphan_seconds: None,
            sentinel_after: None,
        }
    }

    /// Path to the fake binary.
    pub fn path(&self) -> String {
        self.dir.path().join("claude").display().to_string()
    }

    /// A path inside the temporary directory that holds no binary.
    pub fn missing_path(&self) -> PathBuf {
        self.dir.path().join("no-such-binary")
    }

    fn read(&self, name: &str) -> Option<String> {
        fs::read_to_string(self.dir.path().join(name)).ok()
    }

    /// The arguments of the last invocation.
    ///
    /// # Panics
    ///
    /// Panics if the fake was never invoked.
    pub fn argv(&self) -> Vec<String> {
        let recorded = self.read("argv").expect("the fake was never invoked");
        // Arguments are separated by NUL rather than a newline: arguments
        // routinely contain newlines, and a newline separator would split one
        // argument into several. Every argument is followed by a NUL, so the
        // trailing empty element is an artifact of the split.
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

    /// What the library wrote to the child's standard input.
    ///
    /// # Panics
    ///
    /// Panics if the fake was never invoked.
    pub fn stdin(&self) -> String {
        self.read("stdin").expect("the fake was never invoked")
    }

    /// The contents of the file named by `--system-prompt-file`, if any.
    ///
    /// Captured by the script at invocation time: the caller deletes the file
    /// as soon as the turn ends, so reading it afterwards finds nothing.
    pub fn system_prompt(&self) -> Option<String> {
        self.read("system_prompt")
    }

    /// The permission bits of the file named by `--system-prompt-file`.
    ///
    /// Captured by the script at invocation time, for the same reason as
    /// [`FakeClaude::system_prompt`].
    pub fn system_prompt_mode(&self) -> Option<u32> {
        let recorded = self.read("system_prompt_mode")?;
        u32::from_str_radix(recorded.trim(), 8).ok()
    }

    /// How many times the fake has been invoked.
    pub fn spawn_count(&self) -> usize {
        self.read("spawns").map_or(0, |text| text.trim().len())
    }

    /// Whether `CLAUDECODE` was set in the child's environment.
    ///
    /// # Panics
    ///
    /// Panics if the fake was never invoked.
    pub fn saw_nested_marker(&self) -> bool {
        self.read("nested")
            .expect("the fake was never invoked")
            .trim()
            == "present"
    }

    /// Whether the child's delayed sentinel has appeared.
    pub fn sentinel_exists(&self) -> bool {
        self.dir.path().join("sentinel").exists()
    }

    /// The working directory the child ran in.
    pub fn working_dir(&self) -> Option<String> {
        self.read("cwd").map(|text| text.trim().to_owned())
    }
}

impl FakeClaudeBuilder {
    /// What the fake writes to standard output.
    #[must_use]
    pub fn stdout(mut self, stdout: &str) -> Self {
        stdout.clone_into(&mut self.stdout);
        self
    }

    /// What the fake writes to standard error.
    #[must_use]
    pub fn stderr(mut self, stderr: &str) -> Self {
        stderr.clone_into(&mut self.stderr);
        self
    }

    /// The fake's exit status.
    #[must_use]
    pub fn exit_code(mut self, code: i32) -> Self {
        self.exit_code = code;
        self
    }

    /// Wait before writing anything.
    #[must_use]
    pub fn delay_before(mut self, delay: Duration) -> Self {
        self.delay_before = Some(delay.as_secs_f32());
        self
    }

    /// Wait where the stdout fixture contains [`PAUSE`].
    #[must_use]
    pub fn delay_mid(mut self, delay: Duration) -> Self {
        self.delay_mid = Some(delay.as_secs_f32());
        self
    }

    /// Write standard error first, before any of standard output.
    ///
    /// This is the shape that deadlocks a reader which drains stdout to the
    /// end before touching stderr.
    #[must_use]
    pub fn stderr_first(mut self) -> Self {
        self.stderr_first = true;
        self
    }

    /// Leave a backgrounded grandchild holding the inherited pipes open.
    #[must_use]
    pub fn orphan_for(mut self, lifetime: Duration) -> Self {
        self.orphan_seconds = Some(lifetime.as_secs_f32());
        self
    }

    /// Sleep, then touch a sentinel file, so a test can prove the child was
    /// or was not killed.
    #[must_use]
    pub fn sentinel_after(mut self, delay: Duration) -> Self {
        self.sentinel_after = Some(delay.as_secs_f32());
        self
    }

    /// Write the fake to a temporary directory.
    pub fn build(self) -> FakeClaude {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path();

        let stdout = if self.delay_mid.is_some() {
            self.stdout.clone()
        } else {
            // An unused marker would otherwise reach the stream as content.
            self.stdout.replace(PAUSE, "")
        };
        fs::write(path.join("stdout"), &stdout).expect("write stdout fixture");
        fs::write(path.join("stderr"), &self.stderr).expect("write stderr fixture");

        let sleep_before = self
            .delay_before
            .map(|seconds| format!("sleep {seconds}\n"))
            .unwrap_or_default();

        // A pause inside stdout is expressed by splitting the fixture on the
        // marker and sleeping between the halves.
        let emit_stdout = match self.delay_mid {
            Some(seconds) => format!(
                "awk 'BEGIN{{RS=\"{PAUSE}\"}}{{printf \"%s\", $0; fflush(); \
                 if (RT != \"\") system(\"sleep {seconds}\")}}' \"$here/stdout\"\n"
            ),
            None => "cat \"$here/stdout\"\n".to_owned(),
        };

        let orphan = self
            .orphan_seconds
            .map(|seconds| format!("( sleep {seconds} ) &\n"))
            .unwrap_or_default();

        let sentinel = self
            .sentinel_after
            .map(|seconds| format!("sleep {seconds}\ntouch \"$here/sentinel\"\n"))
            .unwrap_or_default();

        let (first, second) = if self.stderr_first {
            ("cat \"$here/stderr\" >&2\n".to_owned(), emit_stdout)
        } else {
            (emit_stdout, "cat \"$here/stderr\" >&2\n".to_owned())
        };

        // `$0` is the script path, so the script finds its own fixtures
        // without an environment variable, which keeps parallel tests from
        // interfering with each other.
        let script = format!(
            r#"#!/bin/sh
here=$(dirname "$0")
printf '%s\0' "$@" > "$here/argv"
cat > "$here/stdin"
printf 'x' >> "$here/spawns"
pwd > "$here/cwd"
# Copy the system prompt while the child is alive: the caller deletes the
# temporary file as soon as the turn ends.
for a in "$@"; do
  if [ -n "$next_is_sp" ]; then
    cp "$a" "$here/system_prompt" 2>/dev/null
    {{ stat -f '%Lp' "$a" 2>/dev/null || stat -c '%a' "$a" 2>/dev/null; }} > "$here/system_prompt_mode"
    next_is_sp=
  fi
  if [ "$a" = "--system-prompt-file" ]; then next_is_sp=1; fi
done
if [ -n "${{CLAUDECODE+set}}" ]; then echo present > "$here/nested"; else echo absent > "$here/nested"; fi
{orphan}{sleep_before}{first}{second}{sentinel}exit {code}
"#,
            code = self.exit_code
        );

        let binary = path.join("claude");
        let mut file = fs::File::create(&binary).expect("create fake binary");
        file.write_all(script.as_bytes()).expect("write script");
        drop(file);
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755))
            .expect("make the fake executable");

        FakeClaude { dir }
    }
}
