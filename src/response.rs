//! The JSON envelope that `claude -p --output-format json` writes to stdout,
//! and its translation into rig's canonical response types.

use rig_core::OneOrMany;
use rig_core::completion::{
    AssistantContent, CompletionError, CompletionResponse, GetTokenUsage, Usage,
};
use serde::{Deserialize, Serialize};

/// The result envelope emitted by `claude -p --output-format json`.
///
/// Unknown fields are captured in [`CliResult::extra`] rather than rejected, so
/// a newer CLI adding a field does not break deserialization.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CliResult {
    /// Whether the CLI reported the turn as failed.
    #[serde(default)]
    pub is_error: bool,
    /// Result discriminator, such as `success` or `error_during_execution`.
    #[serde(default)]
    pub subtype: String,
    /// The assistant's final text.
    #[serde(default)]
    pub result: Option<String>,
    /// The CLI session identifier for this turn.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Why the model stopped, when the CLI reports it.
    #[serde(default)]
    pub stop_reason: Option<String>,
    /// Cost the CLI attributes to this turn, in US dollars.
    ///
    /// Reported for observability. On a subscription-authenticated run this is
    /// an equivalent-cost figure, not an amount billed to an API account.
    #[serde(default)]
    pub total_cost_usd: f64,
    /// Token counts for this turn.
    #[serde(default)]
    pub usage: CliUsage,
    /// Fields this crate does not model.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Token counts reported by the CLI.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CliUsage {
    /// Input tokens that were not served from cache.
    #[serde(default)]
    pub input_tokens: u64,
    /// Output tokens produced, including reasoning tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Input tokens read from the provider-managed cache.
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    /// Input tokens written to the provider-managed cache.
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    /// Breakdown of the output token count.
    #[serde(default)]
    pub output_tokens_details: OutputTokenDetails,
}

/// The part of the output token count spent on internal reasoning.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputTokenDetails {
    /// Tokens spent on extended thinking.
    #[serde(default)]
    pub thinking_tokens: u64,
}

impl CliUsage {
    /// Translate CLI token counts into rig's [`Usage`].
    ///
    /// `total_tokens` is derived, because the CLI reports the two halves and
    /// not a total.
    fn to_rig_usage(&self) -> Usage {
        Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            total_tokens: self.input_tokens.saturating_add(self.output_tokens),
            cached_input_tokens: self.cache_read_input_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: self.output_tokens_details.thinking_tokens,
        }
    }
}

impl CliResult {
    /// Convert a successful envelope into rig's canonical response.
    ///
    /// # Errors
    ///
    /// Returns [`CompletionError::ProviderError`] when the envelope reports a
    /// failed turn, and [`CompletionError::ResponseError`] when a successful
    /// turn carries no result text.
    pub(crate) fn into_completion_response(
        self,
    ) -> Result<CompletionResponse<Self>, CompletionError> {
        if self.is_error {
            return Err(CompletionError::ProviderError(format!(
                "claude reported a failed turn ({}): {}",
                self.subtype,
                self.result.as_deref().unwrap_or("no detail")
            )));
        }

        let Some(text) = self.result.clone() else {
            return Err(CompletionError::ResponseError(
                "claude returned a successful envelope with no result text".to_owned(),
            ));
        };

        let usage = self.usage.to_rig_usage();
        let message_id = self.session_id.clone();

        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::text(text)),
            usage,
            raw_response: self,
            message_id,
        })
    }
}

impl GetTokenUsage for CliResult {
    /// Report this turn's usage.
    ///
    /// rig needs this on the streaming path, where the terminal envelope is
    /// the only place usage appears.
    fn token_usage(&self) -> Usage {
        self.usage.to_rig_usage()
    }
}

/// Deserialize the CLI's stdout into an envelope.
///
/// # Errors
///
/// Returns [`CompletionError::ResponseError`] when stdout is not the JSON
/// envelope, quoting what was received so the failure is diagnosable.
pub(crate) fn parse(stdout: &[u8]) -> Result<CliResult, CompletionError> {
    serde_json::from_slice(stdout).map_err(|error| {
        CompletionError::ResponseError(format!(
            "unparseable claude output: {error}; received: {}",
            String::from_utf8_lossy(stdout).trim()
        ))
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::indexing_slicing)]
mod tests {
    use super::*;

    fn success_envelope() -> &'static str {
        r#"{"is_error":false,"subtype":"success","result":"pong","session_id":"s-1",
            "stop_reason":"end_turn","total_cost_usd":0.001,
            "usage":{"input_tokens":10,"output_tokens":7,"cache_read_input_tokens":3,
                     "cache_creation_input_tokens":2,
                     "output_tokens_details":{"thinking_tokens":4}}}"#
    }

    #[test]
    fn parses_a_success_envelope() {
        let parsed = parse(success_envelope().as_bytes()).unwrap();
        assert!(!parsed.is_error);
        assert_eq!(parsed.result.as_deref(), Some("pong"));
        assert_eq!(parsed.session_id.as_deref(), Some("s-1"));
        assert_eq!(parsed.stop_reason.as_deref(), Some("end_turn"));
        assert!((parsed.total_cost_usd - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn keeps_unmodelled_fields() {
        let parsed = parse(br#"{"result":"hi","brand_new_field":42}"#).unwrap();
        assert_eq!(
            parsed.extra.get("brand_new_field"),
            Some(&serde_json::json!(42))
        );
    }

    #[test]
    fn tolerates_an_envelope_with_only_a_result() {
        let parsed = parse(br#"{"result":"hi"}"#).unwrap();
        assert!(!parsed.is_error);
        assert_eq!(parsed.subtype, "");
        assert_eq!(parsed.usage.input_tokens, 0);
    }

    #[test]
    fn reports_unparseable_output_with_the_payload() {
        let error = parse(b"not json at all").unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("unparseable claude output"), "{rendered}");
        assert!(rendered.contains("not json at all"), "{rendered}");
    }

    #[test]
    fn maps_usage_onto_rig_usage() {
        let parsed = parse(success_envelope().as_bytes()).unwrap();
        let response = parsed.into_completion_response().unwrap();
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 7);
        assert_eq!(response.usage.total_tokens, 17);
        assert_eq!(response.usage.cached_input_tokens, 3);
        assert_eq!(response.usage.cache_creation_input_tokens, 2);
        assert_eq!(response.usage.reasoning_tokens, 4);
        assert_eq!(response.usage.tool_use_prompt_tokens, 0);
    }

    #[test]
    fn reports_usage_through_the_streaming_trait() {
        let parsed = parse(success_envelope().as_bytes()).unwrap();
        let usage = parsed.token_usage();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 7);
        assert_eq!(usage.total_tokens, 17);
        assert_eq!(usage.reasoning_tokens, 4);
    }

    #[test]
    fn derives_a_saturating_total() {
        let usage = CliUsage {
            input_tokens: u64::MAX,
            output_tokens: 5,
            ..CliUsage::default()
        };
        assert_eq!(usage.to_rig_usage().total_tokens, u64::MAX);
    }

    #[test]
    fn carries_the_session_id_as_the_message_id() {
        let parsed = parse(success_envelope().as_bytes()).unwrap();
        let response = parsed.into_completion_response().unwrap();
        assert_eq!(response.message_id.as_deref(), Some("s-1"));
    }

    #[test]
    fn keeps_the_envelope_as_the_raw_response() {
        let parsed = parse(success_envelope().as_bytes()).unwrap();
        let response = parsed.into_completion_response().unwrap();
        assert_eq!(response.raw_response.subtype, "success");
    }

    /// The text of an assistant content block, if it is text.
    fn block_text(block: &AssistantContent) -> Option<&str> {
        match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }
    }

    #[test]
    fn yields_one_text_content_block() {
        let parsed = parse(success_envelope().as_bytes()).unwrap();
        let response = parsed.into_completion_response().unwrap();
        let blocks: Vec<&AssistantContent> = response.choice.iter().collect();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks.first().and_then(|b| block_text(b)), Some("pong"));
    }

    #[test]
    fn block_text_declines_a_non_text_block() {
        let call = AssistantContent::tool_call("id", "name", serde_json::json!({}));
        assert_eq!(block_text(&call), None);
    }

    #[test]
    fn rejects_a_failed_turn() {
        let parsed =
            parse(br#"{"is_error":true,"subtype":"error_during_execution","result":"boom"}"#)
                .unwrap();
        let error = parsed.into_completion_response().unwrap_err();
        let rendered = error.to_string();
        assert!(rendered.contains("error_during_execution"), "{rendered}");
        assert!(rendered.contains("boom"), "{rendered}");
    }

    #[test]
    fn describes_a_failed_turn_that_carries_no_detail() {
        let parsed = parse(br#"{"is_error":true,"subtype":"error_max_turns"}"#).unwrap();
        let error = parsed.into_completion_response().unwrap_err();
        assert!(error.to_string().contains("no detail"), "{error}");
    }

    #[test]
    fn rejects_a_success_envelope_with_no_text() {
        let parsed = parse(br#"{"is_error":false,"subtype":"success"}"#).unwrap();
        let error = parsed.into_completion_response().unwrap_err();
        assert!(error.to_string().contains("no result text"), "{error}");
    }

    #[test]
    fn accepts_empty_result_text() {
        let parsed = parse(br#"{"result":""}"#).unwrap();
        let response = parsed.into_completion_response().unwrap();
        assert_eq!(block_text(&response.choice.first()), Some(""));
    }
}
