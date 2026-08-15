//! A rig tool executed by rig, with the Claude Code CLI as the model.
//!
//! Run with `cargo run --example tool_loop`. Needs a logged-in `claude` on
//! `PATH` and spends that login's usage. The tool is deliberately one the
//! model cannot answer without: a made-up price lookup.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::tool::{Tool, ToolContext};
use rig_claude_code::{ClaudeCodeClient, models};
use serde::Deserialize;

#[derive(Deserialize)]
struct LookupArgs {
    sku: String,
}

#[derive(Debug, thiserror::Error)]
#[error("lookup failed")]
struct LookupError;

struct LookupPrice;

impl Tool for LookupPrice {
    const NAME: &'static str = "lookup_price";
    type Args = LookupArgs;
    type Output = String;
    type Error = LookupError;

    fn description(&self) -> String {
        "The current price of a product, by SKU. Prices are not knowable any other way.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "sku": { "type": "string" } },
            "required": ["sku"]
        })
    }

    async fn call(
        &self,
        _: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        eprintln!("[rig ran lookup_price for {}]", args.sku);
        Ok(format!("${:.2}", 14.20))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeCodeClient::from_env()?;
    let agent = client
        .agent(models::HAIKU)
        .preamble("Answer in one sentence.")
        .tool(LookupPrice)
        .default_max_turns(4)
        .build();

    let answer = agent.prompt("How much is SKU A-113?").await?;
    println!("{answer}");
    Ok(())
}
