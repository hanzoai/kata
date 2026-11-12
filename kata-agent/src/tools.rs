use std::collections::BTreeMap;
use std::future::Future;
use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::env::ExecutionEnv;
use crate::storage::{ReplayPolicy, Storage};
use crate::{KataError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub replay: ReplayPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn ok(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn err(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
        }
    }
}

#[derive(Clone)]
pub struct ToolContext {
    pub conversation_id: String,
    pub task_id: String,
    pub env: Arc<dyn ExecutionEnv>,
    pub cancel: CancellationToken,
    pub(crate) storage: Arc<dyn Storage>,
}

impl ToolContext {
    /// Durable memoization scoped to `(conversation_id, key)`: survives crashes and replays
    /// without re-executing external side effects.
    pub async fn memo<T, F, Fut>(&self, key: &str, compute: F) -> Result<T>
    where
        T: Serialize + DeserializeOwned,
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        if let Some(cached) = self.storage.get_memo(&self.conversation_id, key).await? {
            return Ok(serde_json::from_value(cached)?);
        }
        let val = compute().await?;
        let raw = serde_json::to_value(&val)?;
        self.storage
            .put_memo(&self.conversation_id, key, &raw)
            .await?;
        Ok(val)
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> ToolDef;
    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput>;
}

pub type BoxToolFuture<'a> = Pin<Box<dyn Future<Output = Result<ToolOutput>> + Send + 'a>>;
pub type ToolNext<'a> = Box<dyn FnOnce(Value, &'a ToolContext) -> BoxToolFuture<'a> + Send + 'a>;

/// Middleware (`wrap_tool`) around tool execution for auditing, policy, or sandboxing.
pub trait ToolWrapper: Send + Sync {
    fn wrap<'a>(
        &self,
        tool_name: &'a str,
        args: Value,
        ctx: &'a ToolContext,
        next: ToolNext<'a>,
    ) -> BoxToolFuture<'a>;
}

/// Malleable tool registry supporting live installation, removal, and `wrap_tool` middleware.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: Arc<RwLock<BTreeMap<String, Arc<dyn Tool>>>>,
    wrappers: Arc<RwLock<Vec<Arc<dyn ToolWrapper>>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, tool: Arc<dyn Tool>) {
        let name = tool.definition().name;
        self.tools.write().unwrap().insert(name, tool);
    }

    pub fn remove(&self, name: &str) -> bool {
        self.tools.write().unwrap().remove(name).is_some()
    }

    pub fn wrap_tool(&self, wrapper: Arc<dyn ToolWrapper>) {
        self.wrappers.write().unwrap().push(wrapper);
    }

    pub fn install_coding_tools(&self) {
        CodingTools::install(self);
    }

    pub fn definitions(&self) -> Vec<ToolDef> {
        self.tools
            .read()
            .unwrap()
            .values()
            .map(|t| t.definition())
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.read().unwrap().get(name).cloned()
    }

    pub async fn execute(&self, name: &str, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let tool = self
            .get(name)
            .ok_or_else(|| KataError::Tool(format!("unknown tool `{name}`")))?;
        let wrappers = self.wrappers.read().unwrap().clone();
        let mut chain: ToolNext<'_> = Box::new(move |a, c| Box::pin(async move { tool.call(a, c).await }));
        for wrapper in wrappers.into_iter().rev() {
            let prev = chain;
            chain = Box::new(move |a, c| wrapper.wrap(name, a, c, prev));
        }
        chain(args, ctx).await
    }
}

/// Built-in coding tool suite (`read`, `write`, `edit`, `bash`).
pub struct CodingTools;

impl CodingTools {
    pub fn install(registry: &ToolRegistry) {
        registry.register(Arc::new(ReadTool));
        registry.register(Arc::new(WriteTool));
        registry.register(Arc::new(EditTool));
        registry.register(Arc::new(BashTool));
    }
}

struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "read".into(),
            description: "Read UTF-8 file contents relative to the workspace.".into(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"]
            }),
            replay: ReplayPolicy::Safe,
        }
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| KataError::InvalidArgument("missing `path`".into()))?;
        match ctx.env.read_file(Path::new(path)).await {
            Ok(text) => Ok(ToolOutput::ok(text)),
            Err(e) => Ok(ToolOutput::err(e.to_string())),
        }
    }
}

struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "write".into(),
            description: "Write UTF-8 file contents relative to the workspace.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"]
            }),
            replay: ReplayPolicy::Interrupt,
        }
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| KataError::InvalidArgument("missing `path`".into()))?;
        let content = args
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| KataError::InvalidArgument("missing `content`".into()))?;
        ctx.env.write_file(Path::new(path), content).await?;
        Ok(ToolOutput::ok(format!("wrote {} bytes to {path}", content.len())))
    }
}

struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "edit".into(),
            description: "Replace exact text in a workspace file.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_text": { "type": "string" },
                    "new_text": { "type": "string" }
                },
                "required": ["path", "old_text", "new_text"]
            }),
            replay: ReplayPolicy::Interrupt,
        }
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let path = args
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| KataError::InvalidArgument("missing `path`".into()))?;
        let old_text = args
            .get("old_text")
            .and_then(Value::as_str)
            .ok_or_else(|| KataError::InvalidArgument("missing `old_text`".into()))?;
        let new_text = args
            .get("new_text")
            .and_then(Value::as_str)
            .ok_or_else(|| KataError::InvalidArgument("missing `new_text`".into()))?;
        let existing = ctx.env.read_file(Path::new(path)).await?;
        if !existing.contains(old_text) {
            return Ok(ToolOutput::err(format!("target snippet not found in {path}")));
        }
        let updated = existing.replacen(old_text, new_text, 1);
        ctx.env.write_file(Path::new(path), &updated).await?;
        Ok(ToolOutput::ok(format!("edited {path}")))
    }
}

struct BashTool;

#[async_trait]
impl Tool for BashTool {
    fn definition(&self) -> ToolDef {
        ToolDef {
            name: "bash".into(),
            description: "Execute a shell command in the workspace environment.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "timeout_ms": { "type": "integer" }
                },
                "required": ["command"]
            }),
            replay: ReplayPolicy::Interrupt,
        }
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let command = args
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| KataError::InvalidArgument("missing `command`".into()))?;
        let timeout = args
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .map(Duration::from_millis);
        let out = ctx.env.exec(command, timeout, &ctx.cancel).await?;
        let combined = if out.stderr.is_empty() {
            out.stdout
        } else if out.stdout.is_empty() {
            out.stderr
        } else {
            format!("{}\n{}", out.stdout, out.stderr)
        };
        Ok(ToolOutput {
            content: combined,
            is_error: out.exit_code != 0,
        })
    }
}
