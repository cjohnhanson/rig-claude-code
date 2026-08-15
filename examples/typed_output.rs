//! Structured output against a real `claude` binary.
//!
//! Run with `cargo run --example typed_output`. Needs a logged-in `claude` on
//! `PATH` and spends that login's usage. This is the path the README's
//! structured-output section documents; it is exercised here rather than only
//! against the scripted stand-in because the CLI's schema validator is what
//! decides whether the schema is accepted at all.

use rig::completion::TypedPrompt;
use rig::prelude::*;
use rig_claude_code::{ClaudeCodeClient, models};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)] // read through Debug only
struct Person {
    name: String,
    age: u8,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeCodeClient::from_env()?;
    let agent = client.agent(models::HAIKU).build();

    let person: Person = agent
        .prompt_typed("Describe Ada Lovelace: her name and the age at which she died.")
        .await?;
    println!("{person:?}");

    Ok(())
}
