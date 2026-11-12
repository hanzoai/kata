//! `kata-agent`: Durable, minimal, malleable Rust agent harness.

pub mod env;
pub mod harness;
pub mod model;
pub mod storage;
pub mod tools;

pub use env::{EnvFactory, ExecOutput, ExecutionEnv, LocalExecutionEnv, MemoryExecutionEnv};
pub use harness::{
    ConversationHandle, CreateOptions, DocSpec, ForkDocPolicy, ForkOptions, Harness,
    HarnessOptions, RootOptions, SettledSubmission, SubmissionHandle, SubmissionInput,
};
pub use model::{
    ModelChoice, ModelMessage, ModelProvider, ModelRegistry, ModelRequest, ModelResponse, Role,
    ToolCall,
};
pub use storage::{
    ConversationMeta, EntryKind, ForkOrigin, JsonlStorage, MemoryStorage, Ownership, ReplayPolicy,
    SqliteStorage, Storage, SubmissionRecord, SubmissionStatus, TaskKind, TaskRecord, TaskStatus,
    TranscriptEntry, WhenBusy,
};
pub use tools::{
    CodingTools, Tool, ToolContext, ToolDef, ToolOutput, ToolRegistry, ToolWrapper,
};

#[derive(Debug, thiserror::Error)]
pub enum KataError {
    #[error("storage error: {0}")]
    Storage(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("aborted")]
    Aborted,
}

pub type Result<T> = std::result::Result<T, KataError>;

pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
