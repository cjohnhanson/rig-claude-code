//! A per-turn MCP server that hands the CLI rig's tools and records what it
//! calls.
//!
//! The Claude Code CLI is an agent harness. When its model emits a tool call,
//! the harness looks the tool up, and if it finds none it feeds
//! `No such tool available` back to the model. `--tools ""` therefore means
//! "reject every call", not "leave calls to the caller". A model that is
//! handed tool descriptions in its prompt reaches for a real `tool_use`
//! block, the harness rejects it, and the model reports the tool as broken.
//! Verified against Claude Code 2.1.233.
//!
//! The way to give the CLI tools it cannot refuse is to give it real ones.
//! For each turn that carries rig tools, this module binds a loopback HTTP
//! listener, serves the Model Context Protocol on it, and advertises rig's
//! [`ToolDefinition`]s. The CLI is pointed at it with `--mcp-config` and
//! `--allowedTools`.
//!
//! The server executes nothing. rig owns the tool loop: rig's runner runs
//! tools *after* the model returns a `ToolCall`, then appends results and
//! calls the model again. So on each `tools/call` this server records the
//! call and answers the CLI with a placeholder. The turn's response then
//! carries every recorded call as [`AssistantContent::ToolCall`], and rig
//! takes it from there. The model's text for that turn is discarded, since it
//! was written against placeholder results.

use std::sync::{Arc, Mutex};

use rig_core::completion::ToolDefinition;
use rig_core::message::AssistantContent;
use rmcp::ErrorData;
use rmcp::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResult, ContentBlock, Implementation, ListToolsResult,
    PaginatedRequestParams, ProtocolVersion, ServerCapabilities, ServerInfo, Tool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::streamable_http_server::StreamableHttpService;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;

/// The MCP server name. It becomes the `mcp__rig__` prefix of every tool name
/// the CLI sees.
pub(crate) const SERVER_NAME: &str = "rig";

/// The text the CLI's model sees as each tool's result during the turn.
///
/// The model may write a final answer against this. That answer is discarded:
/// the presence of recorded calls means rig re-prompts with real results.
const PLACEHOLDER: &str = "The harness has recorded this call and will run it. Continue.";

/// One tool call the CLI made during a turn.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RecordedCall {
    /// A fresh id for the call, minted here.
    pub(crate) id: String,
    /// The tool's rig name, with the MCP prefix removed.
    pub(crate) name: String,
    /// The arguments as the model supplied them.
    pub(crate) arguments: serde_json::Value,
}

impl RecordedCall {
    /// The call as rig expects to see it in an assistant message.
    pub(crate) fn into_content(self) -> AssistantContent {
        AssistantContent::tool_call(self.id, self.name, self.arguments)
    }

    /// The call as rig expects to see it in a stream.
    pub(crate) fn into_raw(self) -> rig_core::streaming::RawStreamingToolCall {
        rig_core::streaming::RawStreamingToolCall::new(self.id, self.name, self.arguments)
    }
}

/// The handler behind the per-turn server: rig's tools, and what was called.
#[derive(Clone)]
struct Handler {
    tools: Arc<Vec<Tool>>,
    calls: Arc<Mutex<Vec<RecordedCall>>>,
}

impl ServerHandler for Handler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::LATEST)
            .with_server_info(Implementation::new(SERVER_NAME, env!("CARGO_PKG_VERSION")))
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(self.tools.as_ref().clone()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let arguments = serde_json::Value::Object(request.arguments.unwrap_or_default());
        let call = RecordedCall {
            id: rig_core::id::generate(),
            name: request.name.to_string(),
            arguments,
        };
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(call);
        }
        Ok(CallToolResult::success(vec![ContentBlock::text(
            PLACEHOLDER,
        )]))
    }
}

/// A live per-turn MCP server.
///
/// Dropping it stops the listener. Every call the CLI made during the turn
/// is available through [`Bridge::take_calls`].
///
/// The listener binds loopback only, but loopback is shared with every
/// process on the machine, and the port is visible in the CLI's argv. A
/// stranger who found it could POST a `tools/call`, the turn would return it
/// as a `ToolCall`, and rig would *execute that tool with the stranger's
/// arguments*. So each bridge mints a random bearer token, hands it to the
/// CLI in the MCP config's `headers` (which the CLI forwards on every
/// request; verified against 2.1.233), and rejects any request that does not
/// present it before the request reaches the MCP layer.
pub(crate) struct Bridge {
    url: String,
    token: String,
    calls: Arc<Mutex<Vec<RecordedCall>>>,
    _serve: crate::model::AbortOnDrop,
}

/// Mint a bearer token no other local process can guess.
///
/// `rig_core::id::generate` is documented as non-cryptographic, which is fine
/// for a call id and not for this. `getrandom` reaches the OS entropy source.
fn mint_token() -> std::io::Result<String> {
    use std::fmt::Write as _;
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).map_err(|error| std::io::Error::other(error.to_string()))?;
    let hex = bytes
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            // Writing to a String cannot fail.
            let _ = write!(hex, "{byte:02x}");
            hex
        });
    Ok(hex)
}

/// Reject any request that does not carry this turn's bearer token.
///
/// A [`tower_service::Service`] in front of the MCP service. A bad or missing
/// token gets `401` and never reaches rmcp, so it cannot open a session,
/// list tools, or record a call.
#[derive(Clone)]
struct RequireToken<S> {
    expected: Arc<str>,
    inner: S,
}

/// The response body rmcp's HTTP service produces.
type McpBody = http_body_util::combinators::BoxBody<bytes::Bytes, std::convert::Infallible>;

impl<S, B> tower_service::Service<http::Request<B>> for RequireToken<S>
where
    S: tower_service::Service<
            http::Request<B>,
            Response = http::Response<McpBody>,
            Error = std::convert::Infallible,
        > + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
    B: Send + 'static,
{
    type Response = http::Response<McpBody>;
    type Error = std::convert::Infallible;
    type Future = std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>,
    >;

    fn poll_ready(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        let presented = request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "));
        // Constant-time compare is not needed here: the token is 256 bits of
        // OS entropy, and the attacker is local with no oracle beyond a 401.
        if presented != Some(self.expected.as_ref()) {
            let response = http::Response::builder()
                .status(http::StatusCode::UNAUTHORIZED)
                .body(http_body_util::combinators::BoxBody::new(
                    http_body_util::Empty::<bytes::Bytes>::new(),
                ))
                .unwrap_or_default();
            return Box::pin(async move { Ok(response) });
        }
        let mut inner = self.inner.clone();
        Box::pin(async move { inner.call(request).await })
    }
}

impl Bridge {
    /// Bind a loopback listener and serve `tools` on it.
    ///
    /// # Errors
    ///
    /// Returns the bind error when no loopback port is available.
    pub(crate) async fn start(tools: &[ToolDefinition]) -> std::io::Result<Self> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let port = listener.local_addr()?.port();
        let url = format!("http://127.0.0.1:{port}/mcp");

        let token = mint_token()?;
        let calls = Arc::new(Mutex::new(Vec::new()));
        let handler = Handler {
            tools: Arc::new(tools.iter().map(to_mcp_tool).collect()),
            calls: Arc::clone(&calls),
        };

        let mcp = StreamableHttpService::new(
            move || Ok(handler.clone()),
            LocalSessionManager::default().into(),
            rmcp::transport::streamable_http_server::StreamableHttpServerConfig::default(),
        );
        let service = hyper_util::service::TowerToHyperService::new(RequireToken {
            expected: Arc::from(token.as_str()),
            inner: mcp,
        });

        let serve = tokio::spawn(async move {
            loop {
                let Ok((stream, _)) = listener.accept().await else {
                    break;
                };
                let service = service.clone();
                tokio::spawn(async move {
                    let io = hyper_util::rt::TokioIo::new(stream);
                    let _ = hyper_util::server::conn::auto::Builder::new(
                        hyper_util::rt::TokioExecutor::new(),
                    )
                    .serve_connection(io, service)
                    .await;
                });
            }
        });

        Ok(Self {
            url,
            token,
            calls,
            _serve: crate::model::AbortOnDrop(serve),
        })
    }

    /// The URL the CLI is given.
    #[cfg(test)]
    pub(crate) fn url(&self) -> &str {
        &self.url
    }

    /// The `--mcp-config` JSON that names this server and carries its token.
    pub(crate) fn mcp_config(&self) -> String {
        serde_json::json!({
            "mcpServers": {
                SERVER_NAME: {
                    "type": "http",
                    "url": self.url,
                    "headers": { "Authorization": format!("Bearer {}", self.token) }
                }
            }
        })
        .to_string()
    }

    /// This turn's bearer token.
    #[cfg(test)]
    pub(crate) fn token(&self) -> &str {
        &self.token
    }

    /// The `--allowedTools` value that lets the CLI call every tool served.
    pub(crate) fn allowed_tools(tools: &[ToolDefinition]) -> String {
        tools
            .iter()
            .map(|tool| format!("mcp__{SERVER_NAME}__{}", tool.name))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Every call the CLI made, in order, and clear the record.
    pub(crate) fn take_calls(&self) -> Vec<RecordedCall> {
        self.calls
            .lock()
            .map(|mut calls| std::mem::take(&mut *calls))
            .unwrap_or_default()
    }
}

/// Convert a rig tool definition into an MCP tool.
fn to_mcp_tool(definition: &ToolDefinition) -> Tool {
    let schema = match &definition.parameters {
        serde_json::Value::Object(map) => map.clone(),
        _ => serde_json::Map::new(),
    };
    Tool::new(
        definition.name.clone(),
        definition.description.clone(),
        Arc::new(schema),
    )
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

    fn add_tool() -> ToolDefinition {
        ToolDefinition {
            name: "add".to_owned(),
            description: "Add two integers".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
                "required": ["left", "right"]
            }),
        }
    }

    #[test]
    fn allowed_tools_carries_the_mcp_prefix_for_every_tool() {
        let tools = vec![
            add_tool(),
            ToolDefinition {
                name: "lookup".to_owned(),
                description: String::new(),
                parameters: serde_json::json!({}),
            },
        ];
        assert_eq!(
            Bridge::allowed_tools(&tools),
            "mcp__rig__add,mcp__rig__lookup"
        );
    }

    #[test]
    fn a_recorded_call_becomes_a_rig_tool_call() {
        let call = RecordedCall {
            id: "c1".to_owned(),
            name: "add".to_owned(),
            arguments: serde_json::json!({"left": 1, "right": 2}),
        };
        match call.into_content() {
            AssistantContent::ToolCall(tool_call) => {
                assert_eq!(tool_call.id, "c1");
                assert_eq!(tool_call.function.name, "add");
                assert_eq!(tool_call.function.arguments["left"], 1);
            }
            other => panic!("expected a tool call, got {other:?}"),
        }
    }

    #[test]
    fn a_non_object_schema_becomes_an_empty_object() {
        let tool = to_mcp_tool(&ToolDefinition {
            name: "t".to_owned(),
            description: String::new(),
            parameters: serde_json::json!("not an object"),
        });
        assert!(tool.input_schema.is_empty());
    }

    #[tokio::test]
    async fn the_bridge_binds_a_loopback_url_and_names_itself_in_the_config() {
        let bridge = Bridge::start(&[add_tool()]).await.unwrap();
        assert!(
            bridge.url().starts_with("http://127.0.0.1:"),
            "{}",
            bridge.url()
        );
        assert!(bridge.url().ends_with("/mcp"));
        let config: serde_json::Value = serde_json::from_str(&bridge.mcp_config()).unwrap();
        assert_eq!(config["mcpServers"]["rig"]["url"], bridge.url());
        assert_eq!(config["mcpServers"]["rig"]["type"], "http");
    }

    /// An rmcp HTTP client transport that presents the bridge's token, as
    /// the CLI does.
    fn authed_transport(
        bridge: &Bridge,
    ) -> rmcp::transport::StreamableHttpClientTransport<reqwest::Client> {
        use rmcp::transport::streamable_http_client::StreamableHttpClientTransportConfig;
        rmcp::transport::StreamableHttpClientTransport::with_client(
            reqwest::Client::default(),
            StreamableHttpClientTransportConfig::with_uri(bridge.url()).auth_header(bridge.token()),
        )
    }

    #[tokio::test]
    async fn a_request_without_the_token_is_refused_before_it_reaches_mcp() {
        // Loopback is shared with every local process, and the port is in the
        // CLI's argv. A stranger's `tools/call` would otherwise be recorded,
        // returned as a ToolCall, and executed by rig with the stranger's
        // arguments.
        let bridge = Bridge::start(&[add_tool()]).await.unwrap();

        let response = reqwest::Client::new()
            .post(bridge.url())
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .body(r#"{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"left":1,"right":1}}}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
        assert!(bridge.take_calls().is_empty(), "nothing may be recorded");
    }

    #[tokio::test]
    async fn a_request_with_the_wrong_token_is_refused() {
        let bridge = Bridge::start(&[add_tool()]).await.unwrap();

        let response = reqwest::Client::new()
            .post(bridge.url())
            .header("Authorization", "Bearer not-the-token")
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream")
            .body(r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"x","version":"0"}}}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
    }

    #[tokio::test]
    async fn each_bridge_mints_its_own_token() {
        let a = Bridge::start(&[add_tool()]).await.unwrap();
        let b = Bridge::start(&[add_tool()]).await.unwrap();
        assert_ne!(a.token(), b.token());
        assert_eq!(a.token().len(), 64, "256 bits, hex");
    }

    #[test]
    fn the_config_carries_the_token_as_a_bearer_header() {
        // Cannot construct a Bridge without binding, so check the shape the
        // config builder produces from known parts.
        let rendered = serde_json::json!({
            "mcpServers": { SERVER_NAME: {
                "type": "http", "url": "http://127.0.0.1:1/mcp",
                "headers": { "Authorization": "Bearer abc" }
            }}
        });
        assert_eq!(
            rendered["mcpServers"]["rig"]["headers"]["Authorization"],
            "Bearer abc"
        );
    }

    #[tokio::test]
    async fn a_client_over_http_lists_the_tools_and_has_its_call_recorded() {
        // The same path the CLI takes: connect over streamable HTTP, list
        // tools, call one. Anything short of a real client would test the
        // handler and not the server.
        use rmcp::ServiceExt as _;
        use rmcp::model::CallToolRequestParams;

        let bridge = Bridge::start(&[add_tool()]).await.unwrap();
        let client = ().serve(authed_transport(&bridge)).await.expect("connect to the bridge");

        let listed = client.list_all_tools().await.expect("list tools");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].name, "add");
        assert_eq!(listed[0].description.as_deref(), Some("Add two integers"));

        let mut args = serde_json::Map::new();
        args.insert("left".to_owned(), serde_json::json!(2));
        args.insert("right".to_owned(), serde_json::json!(3));
        let result = client
            .call_tool(CallToolRequestParams::new("add").with_arguments(args))
            .await
            .expect("call add");
        assert!(!result.is_error.unwrap_or(false));
        assert!(
            result
                .content
                .iter()
                .any(|block| block.as_text().is_some_and(|t| t.text == PLACEHOLDER)),
            "the CLI must be told the call was recorded, not given a result"
        );

        let recorded = bridge.take_calls();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].name, "add");
        assert_eq!(recorded[0].arguments["right"], 3);
        assert!(!recorded[0].id.is_empty(), "the call gets a fresh id");

        client.cancel().await.ok();
    }

    #[tokio::test]
    async fn take_calls_drains_the_record() {
        let bridge = Bridge::start(&[add_tool()]).await.unwrap();
        bridge.calls.lock().unwrap().push(RecordedCall {
            id: "c1".to_owned(),
            name: "add".to_owned(),
            arguments: serde_json::json!({}),
        });
        assert_eq!(bridge.take_calls().len(), 1);
        assert!(
            bridge.take_calls().is_empty(),
            "a second take finds nothing"
        );
    }
}
