//! The JSON envelope that `claude -p --output-format json` writes to stdout,
//! and its translation into rig's canonical response types.

use rig_core::OneOrMany;
use rig_core::ProviderResponseError;
use rig_core::completion::{
    AssistantContent, CompletionError, CompletionResponse, GetTokenUsage, Usage,
};
use std::fmt::Write as _;

use serde::{Deserialize, Deserializer, Serialize};

/// How much of the child's output an error message may quote.
///
/// These strings reach application logs. Without a cap, one broken run can put
/// megabytes into a single log line.
pub(crate) const QUOTE_LIMIT: usize = 2_000;

/// Quote untrusted output for an error message, bounded.
pub(crate) fn quote(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    let mut out: String = trimmed.chars().take(QUOTE_LIMIT).collect();
    if out.len() < trimmed.len() {
        let _ = write!(out, " … (truncated, {} bytes total)", trimmed.len());
    }
    out
}

/// Deserialize a field that may be present, absent, or explicitly `null`.
///
/// `#[serde(default)]` alone covers only the absent case. An explicit `null`
/// still fails, which turns one new `"usage": null` from a future CLI into a
/// lost turn reported as unparseable output — the exact break this module's
/// `extra` catch-all exists to prevent.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Deserialize a token count that may arrive as an integer, a float, or null.
///
/// The CLI reports whole numbers today. A float would otherwise fail the whole
/// envelope, so it is truncated instead: an approximate count is worth more
/// than a lost turn.
fn lenient_count<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    let Some(value) = Option::<serde_json::Value>::deserialize(deserializer)? else {
        return Ok(0);
    };
    Ok(match value {
        serde_json::Value::Number(number) => number.as_u64().or_else(|| {
            number
                .as_f64()
                .filter(|float| float.is_finite() && *float >= 0.0)
                .map(|float| {
                    // Truncating a fractional token count is the documented
                    // behavior: an approximate count beats a lost turn. The
                    // filter above rules out the negative and non-finite cases.
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "an approximate count beats a lost turn"
                    )]
                    let truncated = float as u64;
                    truncated
                })
        }),
        _ => None,
    }
    .unwrap_or(0))
}

/// The result envelope emitted by `claude -p --output-format json`.
///
/// Unknown fields are captured in [`CliResponse::extra`] rather than rejected,
/// so a newer CLI adding a field does not break deserialization.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CliResponse {
    /// Whether the CLI reported the turn as failed.
    #[serde(default, deserialize_with = "null_as_default")]
    pub is_error: bool,
    /// Result discriminator, such as `success` or `error_during_execution`.
    #[serde(default, deserialize_with = "null_as_default")]
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
    #[serde(default, deserialize_with = "null_as_default")]
    pub total_cost_usd: f64,
    /// Token counts for this turn.
    #[serde(default, deserialize_with = "null_as_default")]
    pub usage: CliUsage,
    /// Fields this crate does not model.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Token counts reported by the CLI.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CliUsage {
    /// Input tokens that were not served from cache.
    #[serde(default, deserialize_with = "lenient_count")]
    pub input_tokens: u64,
    /// Output tokens produced, including reasoning tokens.
    #[serde(default, deserialize_with = "lenient_count")]
    pub output_tokens: u64,
    /// Input tokens read from the provider-managed cache.
    #[serde(default, deserialize_with = "lenient_count")]
    pub cache_read_input_tokens: u64,
    /// Input tokens written to the provider-managed cache.
    #[serde(default, deserialize_with = "lenient_count")]
    pub cache_creation_input_tokens: u64,
    /// Breakdown of the output token count.
    #[serde(default, deserialize_with = "null_as_default")]
    pub output_tokens_details: OutputTokenDetails,
    /// Usage fields this crate does not model, such as `service_tier` and
    /// `server_tool_use`.
    ///
    /// Kept for the same reason as [`CliResponse::extra`]: a future usage
    /// field should be recoverable rather than discarded.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// The part of the output token count spent on internal reasoning.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutputTokenDetails {
    /// Tokens spent on extended thinking.
    #[serde(default, deserialize_with = "lenient_count")]
    pub thinking_tokens: u64,
    /// Detail fields this crate does not model.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
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

impl GetTokenUsage for CliResponse {
    /// Report this turn's usage.
    ///
    /// rig needs this on the streaming path, where the terminal envelope is
    /// the only place usage appears.
    fn token_usage(&self) -> Usage {
        self.usage.to_rig_usage()
    }
}

impl CliResponse {
    /// Whether this envelope describes a failed turn.
    ///
    /// Either signal is enough. `is_error` is the CLI's own flag; the
    /// `subtype` is its classifier, and the failure subtypes all begin with
    /// `error`. Requiring both would miss an envelope where the two disagree —
    /// `is_error: true` with `subtype: "success"` is a shape the CLI's own
    /// tooling produces for a usage limit — and an envelope where the flag is
    /// simply absent, which `#[serde(default)]` reads as `false`.
    fn failed(&self) -> bool {
        self.is_error || self.subtype.starts_with("error")
    }

    /// The error this envelope describes, if it describes one.
    ///
    /// The whole envelope travels as the error's body rather than being
    /// flattened into a message. A usage limit, a rate limit, and an
    /// unrecognized model all arrive this way, and a caller that wants to back
    /// off on one and fail hard on another needs to branch on `subtype` —
    /// which a formatted string does not allow. rig exposes the body through
    /// `provider_response_json`:
    ///
    /// ```no_run
    /// # use rig_core::completion::CompletionError;
    /// # fn example(error: &CompletionError) {
    /// if let Ok(Some(body)) = error.provider_response_json() {
    ///     match body.get("subtype").and_then(|s| s.as_str()) {
    ///         Some("error_max_turns") => { /* retry with a larger budget */ }
    ///         Some(other) => eprintln!("claude refused: {other}"),
    ///         None => {}
    ///     }
    /// }
    /// # }
    /// ```
    pub(crate) fn failure(&self) -> Option<CompletionError> {
        self.failed().then(|| {
            // Serializing an envelope that was itself deserialized from JSON
            // cannot fail: every field is a string, bool, number, option, or
            // JSON map. `unwrap_or_default` keeps the crate free of `expect`
            // without inventing a fallback body for a case that cannot occur.
            let body = serde_json::to_string(self).unwrap_or_default();
            CompletionError::ProviderResponse(ProviderResponseError { status: None, body })
        })
    }

    /// Convert a successful envelope into rig's canonical response.
    ///
    /// # Errors
    ///
    /// Returns [`CompletionError::ProviderResponse`] when the envelope reports
    /// a failed turn, and [`CompletionError::ResponseError`] when a successful
    /// turn carries no result text.
    pub(crate) fn into_completion_response(
        self,
    ) -> Result<CompletionResponse<Self>, CompletionError> {
        if let Some(failure) = self.failure() {
            return Err(failure);
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

/// Find the envelope in the CLI's stdout.
///
/// The whole buffer is tried first, which is the ordinary case. Failing that,
/// each line is tried from the last backwards, so a deprecation warning or a
/// stray diagnostic printed ahead of the JSON does not lose the turn. The
/// streaming path is deliberately tolerant of noise on stdout; without this
/// the blocking path would not be, and one warning line from a future CLI
/// would take out every non-streaming call.
pub(crate) fn find_envelope(stdout: &[u8]) -> Option<CliResponse> {
    if let Ok(parsed) = serde_json::from_slice::<CliResponse>(stdout) {
        return Some(parsed);
    }
    let text = String::from_utf8_lossy(stdout);
    text.lines()
        .rev()
        .filter(|line| !line.trim().is_empty())
        .find_map(|line| envelope_line(line.trim()))
}

/// Parse one line, but only if it looks like the result envelope.
///
/// Every field of [`CliResponse`] is defaulted and unknown fields are kept, so
/// *any* JSON object deserializes into one. Without a discriminator, the scan
/// that exists to survive a warning line printed **before** the envelope turns
/// a diagnostic printed **after** it into a lost turn. The streaming
/// classifier requires `"type":"result"`; this requires the same, or, for the
/// blocking envelope which carries no `type`, at least one field only a real
/// envelope has.
fn envelope_line(line: &str) -> Option<CliResponse> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let object = value.as_object()?;
    let looks_like_a_result = match object.get("type").and_then(serde_json::Value::as_str) {
        Some(kind) => kind == "result",
        None => ["result", "subtype", "session_id", "is_error"]
            .iter()
            .any(|field| object.contains_key(*field)),
    };
    looks_like_a_result.then(|| serde_json::from_value(value).ok())?
}

/// Deserialize the CLI's stdout into an envelope.
///
/// # Errors
///
/// Returns [`CompletionError::ResponseError`] when stdout holds no envelope,
/// quoting a bounded prefix of what was received so the failure is
/// diagnosable.
#[cfg_attr(not(test), expect(dead_code, reason = "used by the response tests"))]
pub(crate) fn parse(stdout: &[u8]) -> Result<CliResponse, CompletionError> {
    find_envelope(stdout).ok_or_else(|| {
        CompletionError::ResponseError(format!(
            "unparseable claude output; received: {}",
            quote(stdout)
        ))
    })
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    fn success_envelope() -> &'static str {
        r#"{"is_error":false,"subtype":"success","result":"pong","session_id":"s-1",
            "stop_reason":"end_turn","total_cost_usd":0.001,
            "usage":{"input_tokens":10,"output_tokens":7,"cache_read_input_tokens":3,
                     "cache_creation_input_tokens":2,
                     "output_tokens_details":{"thinking_tokens":4}}}"#
    }

    /// The text of an assistant content block, if it is text.
    fn block_text(block: &AssistantContent) -> Option<&str> {
        match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        }
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
    fn keeps_unmodelled_usage_fields() {
        let parsed =
            parse(br#"{"result":"hi","usage":{"input_tokens":1,"service_tier":"standard"}}"#)
                .unwrap();
        assert_eq!(
            parsed.usage.extra.get("service_tier"),
            Some(&serde_json::json!("standard"))
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
    fn tolerates_null_valued_fields() {
        // `#[serde(default)]` covers a missing field, never an explicit null.
        // A future CLI emitting `"usage": null` on a refusal would otherwise
        // lose the turn.
        for body in [
            r#"{"result":"hi","usage":null}"#,
            r#"{"result":"hi","subtype":null}"#,
            r#"{"result":"hi","is_error":null}"#,
            r#"{"result":"hi","total_cost_usd":null}"#,
            r#"{"result":"hi","usage":{"input_tokens":null}}"#,
            r#"{"result":"hi","usage":{"output_tokens_details":null}}"#,
        ] {
            let parsed = parse(body.as_bytes())
                .unwrap_or_else(|error| panic!("{body} should parse, got {error}"));
            assert_eq!(parsed.result.as_deref(), Some("hi"));
        }
    }

    #[test]
    fn tolerates_a_float_token_count() {
        let parsed = parse(br#"{"result":"hi","usage":{"input_tokens":10.7}}"#).unwrap();
        assert_eq!(parsed.usage.input_tokens, 10);
    }

    #[test]
    fn treats_a_nonsensical_token_count_as_zero() {
        for body in [
            r#"{"result":"hi","usage":{"input_tokens":"many"}}"#,
            r#"{"result":"hi","usage":{"input_tokens":-4}}"#,
        ] {
            let parsed = parse(body.as_bytes()).unwrap();
            assert_eq!(parsed.usage.input_tokens, 0, "{body}");
        }
    }

    #[test]
    fn finds_an_envelope_after_a_warning_line() {
        // The streaming path ignores noise on stdout. Without this the
        // blocking path would not, and one deprecation warning from a future
        // CLI would take out every non-streaming call.
        let stdout = b"Warning: something deprecated\n{\"result\":\"hi\"}\n";
        assert_eq!(parse(stdout).unwrap().result.as_deref(), Some("hi"));
    }

    #[test]
    fn finds_the_last_envelope_when_lines_follow_it() {
        let stdout = b"{\"result\":\"first\"}\n{\"result\":\"second\"}\n";
        assert_eq!(parse(stdout).unwrap().result.as_deref(), Some("second"));
    }

    #[test]
    fn ignores_a_json_diagnostic_printed_after_the_envelope() {
        // Every field is defaulted and unknown fields are kept, so any JSON
        // object deserializes into an envelope. Without a discriminator this
        // telemetry line wins the backwards scan and the answer is lost.
        let stdout = b"{\"result\":\"the answer\"}\n{\"type\":\"telemetry\",\"flushed\":true}\n";
        assert_eq!(parse(stdout).unwrap().result.as_deref(), Some("the answer"));
    }

    #[test]
    fn ignores_a_plain_json_object_printed_after_the_envelope() {
        let stdout = b"{\"result\":\"the answer\"}\n{\"flushed\":true}\n";
        assert_eq!(parse(stdout).unwrap().result.as_deref(), Some("the answer"));
    }

    #[test]
    fn accepts_a_typed_result_line() {
        let stdout = b"noise\n{\"type\":\"result\",\"result\":\"ok\"}\n";
        assert_eq!(parse(stdout).unwrap().result.as_deref(), Some("ok"));
    }

    #[test]
    fn reports_unparseable_output_with_a_bounded_quote() {
        let error = parse(b"not json at all").unwrap_err().to_string();
        assert!(error.contains("unparseable claude output"), "{error}");
        assert!(error.contains("not json at all"), "{error}");
    }

    #[test]
    fn truncates_a_huge_quote() {
        let huge = "x".repeat(QUOTE_LIMIT * 3);
        let quoted = quote(huge.as_bytes());
        assert!(quoted.len() < huge.len(), "the quote must be bounded");
        assert!(quoted.contains("truncated"), "{quoted}");
        assert!(quoted.contains(&(QUOTE_LIMIT * 3).to_string()), "{quoted}");
    }

    #[test]
    fn quotes_short_output_whole() {
        assert_eq!(quote(b"  brief  "), "brief");
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
    fn decodes_escapes_and_non_ascii_verbatim() {
        let parsed = parse(r#"{"result":"café \"quoted\"\\ and\nnewline"}"#.as_bytes()).unwrap();
        let response = parsed.into_completion_response().unwrap();
        assert_eq!(
            block_text(&response.choice.first()),
            Some("café \"quoted\"\\ and\nnewline")
        );
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
    fn treats_an_error_subtype_as_a_failure_whatever_the_flag_says() {
        // Trusting the flag alone returns "I gave up" as a successful
        // assistant message.
        let parsed =
            parse(br#"{"is_error":false,"subtype":"error_max_turns","result":"I gave up"}"#)
                .unwrap();
        let error = parsed.into_completion_response().unwrap_err();
        assert!(error.to_string().contains("error_max_turns"), "{error}");
    }

    #[test]
    fn a_failure_carries_the_whole_envelope_for_the_caller_to_branch_on() {
        // A rate limit and an unrecognized model are both failed turns. A
        // caller that backs off on one and fails hard on the other needs the
        // subtype as data, not inside a sentence.
        let parsed =
            parse(br#"{"is_error":true,"subtype":"error_max_turns","result":"gave up"}"#).unwrap();
        let error = parsed.into_completion_response().unwrap_err();

        let body: serde_json::Value = error
            .provider_response_json()
            .expect("the body is valid JSON")
            .expect("the failure carries a provider response");
        assert_eq!(
            body.get("subtype").and_then(serde_json::Value::as_str),
            Some("error_max_turns")
        );
        assert_eq!(
            body.get("result").and_then(serde_json::Value::as_str),
            Some("gave up")
        );
    }

    #[test]
    fn a_failure_with_no_detail_still_carries_its_subtype() {
        let parsed = parse(br#"{"is_error":true,"subtype":"error_max_turns"}"#).unwrap();
        let error = parsed.into_completion_response().unwrap_err();
        assert!(error.to_string().contains("error_max_turns"), "{error}");
    }

    #[test]
    fn a_failure_with_no_subtype_still_carries_its_detail() {
        let parsed = parse(br#"{"is_error":true,"result":"boom"}"#).unwrap();
        let error = parsed.into_completion_response().unwrap_err();
        assert!(error.to_string().contains("boom"), "{error}");
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

    #[test]
    fn a_successful_envelope_reports_no_failure() {
        assert!(
            parse(success_envelope().as_bytes())
                .unwrap()
                .failure()
                .is_none()
        );
    }
}
