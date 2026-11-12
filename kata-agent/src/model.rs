use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::tools::ToolDef;
use crate::{KataError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelChoice {
    pub provider: String,
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelMessage {
    pub role: Role,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequest {
    pub model_id: String,
    pub messages: Vec<ModelMessage>,
    pub tools: Vec<ToolDef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelResponse {
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
}

#[async_trait]
pub trait ModelProvider: Send + Sync {
    async fn complete(
        &self,
        request: ModelRequest,
        cancel: &CancellationToken,
    ) -> Result<ModelResponse>;
}

/// Malleable model provider registry supporting live hot-swapping mid-session.
#[derive(Clone, Default)]
pub struct ModelRegistry {
    providers: Arc<RwLock<HashMap<String, Arc<dyn ModelProvider>>>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_provider(&self, id: impl Into<String>, provider: Arc<dyn ModelProvider>) {
        self.providers.write().unwrap().insert(id.into(), provider);
    }

    pub fn remove_provider(&self, id: &str) -> bool {
        self.providers.write().unwrap().remove(id).is_some()
    }

    pub fn get(&self, id: &str) -> Result<Arc<dyn ModelProvider>> {
        self.providers
            .read()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| KataError::Model(format!("unknown model provider `{id}`")))
    }
}
