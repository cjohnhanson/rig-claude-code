//! Drives a real `claude` binary: one blocking turn, one multi-turn
//! conversation, and one streamed turn.
//!
//! Run it with `cargo run --example shipping_forecast`. It needs a `claude`
//! on `PATH` that is already logged in, and it spends that login's usage.

use futures::StreamExt as _;
use rig::agent::MultiTurnStreamItem;
use rig::completion::{Chat, Message, Prompt};
use rig::prelude::*;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use rig_claude_code::ClaudeCodeClient;
use std::io::Write as _;

const PREAMBLE: &str = "You write shipping forecasts in the style of the BBC. \
     Answer in at most two sentences. No preamble, no caveats. \
     Invent plausible conditions; never refuse for want of live data.";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = ClaudeCodeClient::from_env()?;
    println!("claude {}\n", client.version().await?);

    let agent = client
        .agent("haiku")
        .name("shipping-forecast")
        .preamble(PREAMBLE)
        .build();

    println!("— one shot —");
    println!("{}\n", agent.prompt("Report on Dogger.").await?);

    println!("— multi-turn —");
    let mut history: Vec<Message> = Vec::new();
    println!("{}", agent.chat("Report on Fastnet.", &mut history).await?);
    println!(
        "{}\n",
        agent
            .chat("Is that better or worse than Dogger?", &mut history)
            .await?
    );
    println!("history holds {} messages\n", history.len());

    println!("— streamed —");
    let mut stream = agent.stream_prompt("Report on Rockall.").await;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::StreamAssistantItem(content) = item? {
            match content {
                StreamedAssistantContent::Text(chunk) => {
                    print!("{}", chunk.text);
                    std::io::stdout().flush()?;
                }
                StreamedAssistantContent::ReasoningDelta { .. } => print!("."),
                _ => {}
            }
        }
    }
    println!();

    Ok(())
}
