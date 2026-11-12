use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::AsyncWriteExt;

use crate::model::ToolCall;
use crate::{KataError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayPolicy {
    Safe,
    Interrupt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WhenBusy {
    #[default]
    Queue,
    Steer,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Ownership {
    Root,
    Task {
        parent_conversation_id: String,
        parent_task_id: String,
    },
    Conversation {
        parent_conversation_id: String,
    },
    Ownerless,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkOrigin {
    pub parent_conversation_id: String,
    pub up_to_seq: u64,
    pub entry_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationMeta {
    pub id: String,
    pub is_root: bool,
    pub model_provider: String,
    pub model_id: String,
    pub cwd: PathBuf,
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub ownership: Ownership,
    pub fork_origin: Option<ForkOrigin>,
    /// Entries with `seq <= compacted_up_to_seq` are excluded from the active working set
    /// (replaced by the compaction/handoff summary entry) while remaining searchable on disk.
    pub compacted_up_to_seq: u64,
    pub closed: bool,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntryKind {
    User,
    Assistant {
        #[serde(default)]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        aborted: bool,
    },
    ToolResult {
        call_id: String,
        tool_name: String,
        is_error: bool,
        #[serde(default)]
        interrupted: bool,
    },
    Compaction {
        up_to_seq: u64,
    },
    Handoff {
        from_seq: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub id: String,
    pub conversation_id: String,
    pub seq: u64,
    pub kind: EntryKind,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    pub created_at_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubmissionStatus {
    Pending,
    Running,
    Completed,
    Aborted,
    Failed,
}

impl SubmissionStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Aborted | Self::Failed)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubmissionRecord {
    pub id: String,
    pub conversation_id: String,
    pub request_id: Option<String>,
    pub content: String,
    pub when_busy: WhenBusy,
    pub status: SubmissionStatus,
    pub answer_entry_id: Option<String>,
    pub answer: Option<String>,
    pub error: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TaskKind {
    ModelCall {
        partial_output: Option<String>,
    },
    ToolCall {
        call_id: String,
        tool_name: String,
        arguments: Value,
        replay: ReplayPolicy,
    },
    Custom {
        name: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Running,
    Completed,
    Interrupted,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRecord {
    pub id: String,
    pub conversation_id: String,
    pub submission_id: Option<String>,
    pub kind: TaskKind,
    pub status: TaskStatus,
    pub checkpoint: Option<Value>,
    pub updated_at_ms: u64,
}

#[async_trait]
pub trait Storage: Send + Sync {
    async fn get_root_conversation(&self) -> Result<Option<ConversationMeta>>;
    async fn get_conversation(&self, id: &str) -> Result<Option<ConversationMeta>>;
    async fn put_conversation(&self, meta: &ConversationMeta) -> Result<()>;
    async fn list_conversations(&self) -> Result<Vec<ConversationMeta>>;

    async fn append_entry(
        &self,
        conversation_id: &str,
        id: String,
        kind: EntryKind,
        content: String,
        request_id: Option<String>,
        created_at_ms: u64,
    ) -> Result<TranscriptEntry>;

    /// Loads entries with `after_seq < seq` and (when `max_seq` is set) `seq <= max_seq`.
    async fn load_entries(
        &self,
        conversation_id: &str,
        after_seq: u64,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>>;

    async fn get_entry(&self, conversation_id: &str, entry_id: &str) -> Result<Option<TranscriptEntry>>;

    /// Searches the full uncompacted log of `conversation_id` (up to `max_seq` if specified).
    async fn search_entries(
        &self,
        conversation_id: &str,
        query: &str,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>>;

    async fn latest_seq(&self, conversation_id: &str) -> Result<u64>;

    async fn put_submission(&self, record: &SubmissionRecord) -> Result<()>;
    async fn get_submission(&self, id: &str) -> Result<Option<SubmissionRecord>>;
    async fn find_submission_by_request_id(
        &self,
        conversation_id: &str,
        request_id: &str,
    ) -> Result<Option<SubmissionRecord>>;
    async fn list_unfinished_submissions(&self) -> Result<Vec<SubmissionRecord>>;

    async fn put_task(&self, task: &TaskRecord) -> Result<()>;
    async fn get_task(&self, id: &str) -> Result<Option<TaskRecord>>;
    async fn list_unfinished_tasks(&self) -> Result<Vec<TaskRecord>>;

    async fn put_doc(
        &self,
        conversation_id: &str,
        name: &str,
        seq: u64,
        value: &Value,
    ) -> Result<()>;
    async fn get_doc_as_of(
        &self,
        conversation_id: &str,
        name: &str,
        max_seq: Option<u64>,
    ) -> Result<Option<(u64, Value)>>;

    async fn put_memo(&self, conversation_id: &str, key: &str, value: &Value) -> Result<()>;
    async fn get_memo(&self, conversation_id: &str, key: &str) -> Result<Option<Value>>;
}

#[derive(Default, Clone, Serialize, Deserialize)]
struct InMemoryState {
    conversations: BTreeMap<String, ConversationMeta>,
    entries: HashMap<String, Vec<TranscriptEntry>>,
    submissions: BTreeMap<String, SubmissionRecord>,
    request_index: HashMap<(String, String), String>,
    tasks: BTreeMap<String, TaskRecord>,
    docs: HashMap<(String, String), Vec<(u64, Value)>>,
    memos: HashMap<(String, String), Value>,
}

#[derive(Clone, Default)]
pub struct MemoryStorage {
    state: Arc<Mutex<InMemoryState>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Storage for MemoryStorage {
    async fn get_root_conversation(&self) -> Result<Option<ConversationMeta>> {
        let guard = self.state.lock().unwrap();
        Ok(guard.conversations.values().find(|c| c.is_root).cloned())
    }

    async fn get_conversation(&self, id: &str) -> Result<Option<ConversationMeta>> {
        Ok(self.state.lock().unwrap().conversations.get(id).cloned())
    }

    async fn put_conversation(&self, meta: &ConversationMeta) -> Result<()> {
        self.state
            .lock()
            .unwrap()
            .conversations
            .insert(meta.id.clone(), meta.clone());
        Ok(())
    }

    async fn list_conversations(&self) -> Result<Vec<ConversationMeta>> {
        Ok(self.state.lock().unwrap().conversations.values().cloned().collect())
    }

    async fn append_entry(
        &self,
        conversation_id: &str,
        id: String,
        kind: EntryKind,
        content: String,
        request_id: Option<String>,
        created_at_ms: u64,
    ) -> Result<TranscriptEntry> {
        let mut guard = self.state.lock().unwrap();
        let list = guard.entries.entry(conversation_id.to_owned()).or_default();
        let seq = list.last().map_or(1, |last| last.seq + 1);
        let entry = TranscriptEntry {
            id,
            conversation_id: conversation_id.to_owned(),
            seq,
            kind,
            content,
            request_id,
            created_at_ms,
        };
        list.push(entry.clone());
        Ok(entry)
    }

    async fn load_entries(
        &self,
        conversation_id: &str,
        after_seq: u64,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>> {
        let guard = self.state.lock().unwrap();
        let Some(list) = guard.entries.get(conversation_id) else {
            return Ok(Vec::new());
        };
        Ok(list
            .iter()
            .filter(|e| e.seq > after_seq && max_seq.is_none_or(|m| e.seq <= m))
            .cloned()
            .collect())
    }

    async fn get_entry(&self, conversation_id: &str, entry_id: &str) -> Result<Option<TranscriptEntry>> {
        let guard = self.state.lock().unwrap();
        Ok(guard
            .entries
            .get(conversation_id)
            .and_then(|list| list.iter().find(|e| e.id == entry_id))
            .cloned())
    }

    async fn search_entries(
        &self,
        conversation_id: &str,
        query: &str,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>> {
        let needle = query.to_lowercase();
        let guard = self.state.lock().unwrap();
        let Some(list) = guard.entries.get(conversation_id) else {
            return Ok(Vec::new());
        };
        Ok(list
            .iter()
            .filter(|e| {
                max_seq.is_none_or(|m| e.seq <= m) && e.content.to_lowercase().contains(&needle)
            })
            .cloned()
            .collect())
    }

    async fn latest_seq(&self, conversation_id: &str) -> Result<u64> {
        let guard = self.state.lock().unwrap();
        Ok(guard
            .entries
            .get(conversation_id)
            .and_then(|list| list.last())
            .map_or(0, |e| e.seq))
    }

    async fn put_submission(&self, record: &SubmissionRecord) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        if let Some(req_id) = &record.request_id {
            guard
                .request_index
                .insert((record.conversation_id.clone(), req_id.clone()), record.id.clone());
        }
        guard.submissions.insert(record.id.clone(), record.clone());
        Ok(())
    }

    async fn get_submission(&self, id: &str) -> Result<Option<SubmissionRecord>> {
        Ok(self.state.lock().unwrap().submissions.get(id).cloned())
    }

    async fn find_submission_by_request_id(
        &self,
        conversation_id: &str,
        request_id: &str,
    ) -> Result<Option<SubmissionRecord>> {
        let guard = self.state.lock().unwrap();
        let Some(sub_id) = guard
            .request_index
            .get(&(conversation_id.to_owned(), request_id.to_owned()))
        else {
            return Ok(None);
        };
        Ok(guard.submissions.get(sub_id).cloned())
    }

    async fn list_unfinished_submissions(&self) -> Result<Vec<SubmissionRecord>> {
        let guard = self.state.lock().unwrap();
        Ok(guard
            .submissions
            .values()
            .filter(|s| !s.status.is_terminal())
            .cloned()
            .collect())
    }

    async fn put_task(&self, task: &TaskRecord) -> Result<()> {
        self.state
            .lock()
            .unwrap()
            .tasks
            .insert(task.id.clone(), task.clone());
        Ok(())
    }

    async fn get_task(&self, id: &str) -> Result<Option<TaskRecord>> {
        Ok(self.state.lock().unwrap().tasks.get(id).cloned())
    }

    async fn list_unfinished_tasks(&self) -> Result<Vec<TaskRecord>> {
        let guard = self.state.lock().unwrap();
        Ok(guard
            .tasks
            .values()
            .filter(|t| t.status == TaskStatus::Running)
            .cloned()
            .collect())
    }

    async fn put_doc(
        &self,
        conversation_id: &str,
        name: &str,
        seq: u64,
        value: &Value,
    ) -> Result<()> {
        let mut guard = self.state.lock().unwrap();
        let versions = guard
            .docs
            .entry((conversation_id.to_owned(), name.to_owned()))
            .or_default();
        versions.push((seq, value.clone()));
        versions.sort_by_key(|(s, _)| *s);
        Ok(())
    }

    async fn get_doc_as_of(
        &self,
        conversation_id: &str,
        name: &str,
        max_seq: Option<u64>,
    ) -> Result<Option<(u64, Value)>> {
        let guard = self.state.lock().unwrap();
        let Some(versions) = guard.docs.get(&(conversation_id.to_owned(), name.to_owned())) else {
            return Ok(None);
        };
        Ok(versions
            .iter()
            .rev()
            .find(|(s, _)| max_seq.is_none_or(|m| *s <= m))
            .cloned())
    }

    async fn put_memo(&self, conversation_id: &str, key: &str, value: &Value) -> Result<()> {
        self.state
            .lock()
            .unwrap()
            .memos
            .insert((conversation_id.to_owned(), key.to_owned()), value.clone());
        Ok(())
    }

    async fn get_memo(&self, conversation_id: &str, key: &str) -> Result<Option<Value>> {
        Ok(self
            .state
            .lock()
            .unwrap()
            .memos
            .get(&(conversation_id.to_owned(), key.to_owned()))
            .cloned())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum JsonlOp {
    Conversation { meta: ConversationMeta },
    Entry { entry: TranscriptEntry },
    Submission { record: SubmissionRecord },
    Task { task: TaskRecord },
    Doc {
        conversation_id: String,
        name: String,
        seq: u64,
        value: Value,
    },
    Memo {
        conversation_id: String,
        key: String,
        value: Value,
    },
}

/// Append-only crash-safe JSONL storage backend.
#[derive(Clone)]
pub struct JsonlStorage {
    path: PathBuf,
    mem: MemoryStorage,
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

impl JsonlStorage {
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                tokio::fs::create_dir_all(parent).await?;
            }
        }
        let mem = MemoryStorage::new();
        if tokio::fs::try_exists(&path).await? {
            let content = tokio::fs::read_to_string(&path).await?;
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                // Ignore a torn final line on unclean crash.
                let Ok(op) = serde_json::from_str::<JsonlOp>(trimmed) else {
                    continue;
                };
                match op {
                    JsonlOp::Conversation { meta } => mem.put_conversation(&meta).await?,
                    JsonlOp::Entry { entry } => {
                        let mut guard = mem.state.lock().unwrap();
                        let list = guard.entries.entry(entry.conversation_id.clone()).or_default();
                        list.push(entry);
                    }
                    JsonlOp::Submission { record } => mem.put_submission(&record).await?,
                    JsonlOp::Task { task } => mem.put_task(&task).await?,
                    JsonlOp::Doc {
                        conversation_id,
                        name,
                        seq,
                        value,
                    } => mem.put_doc(&conversation_id, &name, seq, &value).await?,
                    JsonlOp::Memo {
                        conversation_id,
                        key,
                        value,
                    } => mem.put_memo(&conversation_id, &key, &value).await?,
                }
            }
        }
        Ok(Self {
            path,
            mem,
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    async fn append_op(&self, op: &JsonlOp) -> Result<()> {
        let _guard = self.write_lock.lock().await;
        let mut line = serde_json::to_vec(op)?;
        line.push(b'\n');
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await?;
        file.write_all(&line).await?;
        file.flush().await?;
        Ok(())
    }
}

#[async_trait]
impl Storage for JsonlStorage {
    async fn get_root_conversation(&self) -> Result<Option<ConversationMeta>> {
        self.mem.get_root_conversation().await
    }

    async fn get_conversation(&self, id: &str) -> Result<Option<ConversationMeta>> {
        self.mem.get_conversation(id).await
    }

    async fn put_conversation(&self, meta: &ConversationMeta) -> Result<()> {
        self.append_op(&JsonlOp::Conversation { meta: meta.clone() })
            .await?;
        self.mem.put_conversation(meta).await
    }

    async fn list_conversations(&self) -> Result<Vec<ConversationMeta>> {
        self.mem.list_conversations().await
    }

    async fn append_entry(
        &self,
        conversation_id: &str,
        id: String,
        kind: EntryKind,
        content: String,
        request_id: Option<String>,
        created_at_ms: u64,
    ) -> Result<TranscriptEntry> {
        let entry = self
            .mem
            .append_entry(conversation_id, id, kind, content, request_id, created_at_ms)
            .await?;
        self.append_op(&JsonlOp::Entry {
            entry: entry.clone(),
        })
        .await?;
        Ok(entry)
    }

    async fn load_entries(
        &self,
        conversation_id: &str,
        after_seq: u64,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>> {
        self.mem
            .load_entries(conversation_id, after_seq, max_seq)
            .await
    }

    async fn get_entry(&self, conversation_id: &str, entry_id: &str) -> Result<Option<TranscriptEntry>> {
        self.mem.get_entry(conversation_id, entry_id).await
    }

    async fn search_entries(
        &self,
        conversation_id: &str,
        query: &str,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>> {
        self.mem
            .search_entries(conversation_id, query, max_seq)
            .await
    }

    async fn latest_seq(&self, conversation_id: &str) -> Result<u64> {
        self.mem.latest_seq(conversation_id).await
    }

    async fn put_submission(&self, record: &SubmissionRecord) -> Result<()> {
        self.append_op(&JsonlOp::Submission {
            record: record.clone(),
        })
        .await?;
        self.mem.put_submission(record).await
    }

    async fn get_submission(&self, id: &str) -> Result<Option<SubmissionRecord>> {
        self.mem.get_submission(id).await
    }

    async fn find_submission_by_request_id(
        &self,
        conversation_id: &str,
        request_id: &str,
    ) -> Result<Option<SubmissionRecord>> {
        self.mem
            .find_submission_by_request_id(conversation_id, request_id)
            .await
    }

    async fn list_unfinished_submissions(&self) -> Result<Vec<SubmissionRecord>> {
        self.mem.list_unfinished_submissions().await
    }

    async fn put_task(&self, task: &TaskRecord) -> Result<()> {
        self.append_op(&JsonlOp::Task { task: task.clone() }).await?;
        self.mem.put_task(task).await
    }

    async fn get_task(&self, id: &str) -> Result<Option<TaskRecord>> {
        self.mem.get_task(id).await
    }

    async fn list_unfinished_tasks(&self) -> Result<Vec<TaskRecord>> {
        self.mem.list_unfinished_tasks().await
    }

    async fn put_doc(
        &self,
        conversation_id: &str,
        name: &str,
        seq: u64,
        value: &Value,
    ) -> Result<()> {
        self.append_op(&JsonlOp::Doc {
            conversation_id: conversation_id.to_owned(),
            name: name.to_owned(),
            seq,
            value: value.clone(),
        })
        .await?;
        self.mem.put_doc(conversation_id, name, seq, value).await
    }

    async fn get_doc_as_of(
        &self,
        conversation_id: &str,
        name: &str,
        max_seq: Option<u64>,
    ) -> Result<Option<(u64, Value)>> {
        self.mem.get_doc_as_of(conversation_id, name, max_seq).await
    }

    async fn put_memo(&self, conversation_id: &str, key: &str, value: &Value) -> Result<()> {
        self.append_op(&JsonlOp::Memo {
            conversation_id: conversation_id.to_owned(),
            key: key.to_owned(),
            value: value.clone(),
        })
        .await?;
        self.mem.put_memo(conversation_id, key, value).await
    }

    async fn get_memo(&self, conversation_id: &str, key: &str) -> Result<Option<Value>> {
        self.mem.get_memo(conversation_id, key).await
    }
}

/// SQLite WAL-backed durable storage keeping inactive history on disk until requested.
#[derive(Clone)]
pub struct SqliteStorage {
    conn: Arc<Mutex<Connection>>,
}

fn sq_err(e: rusqlite::Error) -> KataError {
    KataError::Storage(e.to_string())
}

impl SqliteStorage {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(path).map_err(sq_err)?;
        Self::init(conn)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(sq_err)?;
        Self::init(conn)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA synchronous = NORMAL;
             CREATE TABLE IF NOT EXISTS conversations (
                 id TEXT PRIMARY KEY,
                 is_root INTEGER NOT NULL,
                 json TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS entries (
                 conversation_id TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 id TEXT NOT NULL,
                 content TEXT NOT NULL,
                 json TEXT NOT NULL,
                 PRIMARY KEY (conversation_id, seq)
             );
             CREATE INDEX IF NOT EXISTS idx_entries_id ON entries (conversation_id, id);
             CREATE TABLE IF NOT EXISTS submissions (
                 id TEXT PRIMARY KEY,
                 conversation_id TEXT NOT NULL,
                 request_id TEXT,
                 terminal INTEGER NOT NULL,
                 json TEXT NOT NULL
             );
             CREATE UNIQUE INDEX IF NOT EXISTS idx_submissions_req
                 ON submissions (conversation_id, request_id)
                 WHERE request_id IS NOT NULL;
             CREATE TABLE IF NOT EXISTS tasks (
                 id TEXT PRIMARY KEY,
                 running INTEGER NOT NULL,
                 json TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS docs (
                 conversation_id TEXT NOT NULL,
                 name TEXT NOT NULL,
                 seq INTEGER NOT NULL,
                 json TEXT NOT NULL,
                 PRIMARY KEY (conversation_id, name, seq)
             );
             CREATE TABLE IF NOT EXISTS memos (
                 conversation_id TEXT NOT NULL,
                 key TEXT NOT NULL,
                 json TEXT NOT NULL,
                 PRIMARY KEY (conversation_id, key)
             );",
        )
        .map_err(sq_err)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl Storage for SqliteStorage {
    async fn get_root_conversation(&self) -> Result<Option<ConversationMeta>> {
        let conn = self.conn.lock().unwrap();
        let raw: Option<String> = conn
            .query_row(
                "SELECT json FROM conversations WHERE is_root = 1 LIMIT 1",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(sq_err)?;
        raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    async fn get_conversation(&self, id: &str) -> Result<Option<ConversationMeta>> {
        let conn = self.conn.lock().unwrap();
        let raw: Option<String> = conn
            .query_row(
                "SELECT json FROM conversations WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .optional()
            .map_err(sq_err)?;
        raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    async fn put_conversation(&self, meta: &ConversationMeta) -> Result<()> {
        let raw = serde_json::to_string(meta)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO conversations (id, is_root, json) VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET is_root = excluded.is_root, json = excluded.json",
            params![meta.id, i32::from(meta.is_root), raw],
        )
        .map_err(sq_err)?;
        Ok(())
    }

    async fn list_conversations(&self) -> Result<Vec<ConversationMeta>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT json FROM conversations ORDER BY id ASC")
            .map_err(sq_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(sq_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row.map_err(sq_err)?)?);
        }
        Ok(out)
    }

    async fn append_entry(
        &self,
        conversation_id: &str,
        id: String,
        kind: EntryKind,
        content: String,
        request_id: Option<String>,
        created_at_ms: u64,
    ) -> Result<TranscriptEntry> {
        let conn = self.conn.lock().unwrap();
        let next_seq: u64 = conn
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM entries WHERE conversation_id = ?1",
                params![conversation_id],
                |r| r.get(0),
            )
            .map_err(sq_err)?;
        let entry = TranscriptEntry {
            id: id.clone(),
            conversation_id: conversation_id.to_owned(),
            seq: next_seq,
            kind,
            content: content.clone(),
            request_id,
            created_at_ms,
        };
        let raw = serde_json::to_string(&entry)?;
        conn.execute(
            "INSERT INTO entries (conversation_id, seq, id, content, json) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![conversation_id, next_seq, id, content, raw],
        )
        .map_err(sq_err)?;
        Ok(entry)
    }

    async fn load_entries(
        &self,
        conversation_id: &str,
        after_seq: u64,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>> {
        let conn = self.conn.lock().unwrap();
        let upper = max_seq.unwrap_or(i64::MAX as u64);
        let mut stmt = conn
            .prepare(
                "SELECT json FROM entries
                 WHERE conversation_id = ?1 AND seq > ?2 AND seq <= ?3
                 ORDER BY seq ASC",
            )
            .map_err(sq_err)?;
        let rows = stmt
            .query_map(params![conversation_id, after_seq, upper], |r| {
                r.get::<_, String>(0)
            })
            .map_err(sq_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row.map_err(sq_err)?)?);
        }
        Ok(out)
    }

    async fn get_entry(&self, conversation_id: &str, entry_id: &str) -> Result<Option<TranscriptEntry>> {
        let conn = self.conn.lock().unwrap();
        let raw: Option<String> = conn
            .query_row(
                "SELECT json FROM entries WHERE conversation_id = ?1 AND id = ?2 LIMIT 1",
                params![conversation_id, entry_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(sq_err)?;
        raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    async fn search_entries(
        &self,
        conversation_id: &str,
        query: &str,
        max_seq: Option<u64>,
    ) -> Result<Vec<TranscriptEntry>> {
        let conn = self.conn.lock().unwrap();
        let upper = max_seq.unwrap_or(i64::MAX as u64);
        let pattern = format!("%{}%", query.replace('%', "\\%").replace('_', "\\_"));
        let mut stmt = conn
            .prepare(
                "SELECT json FROM entries
                 WHERE conversation_id = ?1 AND seq <= ?2 AND LOWER(content) LIKE LOWER(?3) ESCAPE '\\'
                 ORDER BY seq ASC",
            )
            .map_err(sq_err)?;
        let rows = stmt
            .query_map(params![conversation_id, upper, pattern], |r| {
                r.get::<_, String>(0)
            })
            .map_err(sq_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row.map_err(sq_err)?)?);
        }
        Ok(out)
    }

    async fn latest_seq(&self, conversation_id: &str) -> Result<u64> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT COALESCE(MAX(seq), 0) FROM entries WHERE conversation_id = ?1",
            params![conversation_id],
            |r| r.get(0),
        )
        .map_err(sq_err)
    }

    async fn put_submission(&self, record: &SubmissionRecord) -> Result<()> {
        let raw = serde_json::to_string(record)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO submissions (id, conversation_id, request_id, terminal, json)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(id) DO UPDATE SET terminal = excluded.terminal, json = excluded.json",
            params![
                record.id,
                record.conversation_id,
                record.request_id,
                i32::from(record.status.is_terminal()),
                raw
            ],
        )
        .map_err(sq_err)?;
        Ok(())
    }

    async fn get_submission(&self, id: &str) -> Result<Option<SubmissionRecord>> {
        let conn = self.conn.lock().unwrap();
        let raw: Option<String> = conn
            .query_row(
                "SELECT json FROM submissions WHERE id = ?1",
                params![id],
                |r| r.get(0),
            )
            .optional()
            .map_err(sq_err)?;
        raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    async fn find_submission_by_request_id(
        &self,
        conversation_id: &str,
        request_id: &str,
    ) -> Result<Option<SubmissionRecord>> {
        let conn = self.conn.lock().unwrap();
        let raw: Option<String> = conn
            .query_row(
                "SELECT json FROM submissions WHERE conversation_id = ?1 AND request_id = ?2 LIMIT 1",
                params![conversation_id, request_id],
                |r| r.get(0),
            )
            .optional()
            .map_err(sq_err)?;
        raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    async fn list_unfinished_submissions(&self) -> Result<Vec<SubmissionRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT json FROM submissions WHERE terminal = 0 ORDER BY rowid ASC")
            .map_err(sq_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(sq_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row.map_err(sq_err)?)?);
        }
        Ok(out)
    }

    async fn put_task(&self, task: &TaskRecord) -> Result<()> {
        let raw = serde_json::to_string(task)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO tasks (id, running, json) VALUES (?1, ?2, ?3)
             ON CONFLICT(id) DO UPDATE SET running = excluded.running, json = excluded.json",
            params![task.id, i32::from(task.status == TaskStatus::Running), raw],
        )
        .map_err(sq_err)?;
        Ok(())
    }

    async fn get_task(&self, id: &str) -> Result<Option<TaskRecord>> {
        let conn = self.conn.lock().unwrap();
        let raw: Option<String> = conn
            .query_row("SELECT json FROM tasks WHERE id = ?1", params![id], |r| {
                r.get(0)
            })
            .optional()
            .map_err(sq_err)?;
        raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }

    async fn list_unfinished_tasks(&self) -> Result<Vec<TaskRecord>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare("SELECT json FROM tasks WHERE running = 1 ORDER BY rowid ASC")
            .map_err(sq_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(sq_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(serde_json::from_str(&row.map_err(sq_err)?)?);
        }
        Ok(out)
    }

    async fn put_doc(
        &self,
        conversation_id: &str,
        name: &str,
        seq: u64,
        value: &Value,
    ) -> Result<()> {
        let raw = serde_json::to_string(value)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO docs (conversation_id, name, seq, json) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(conversation_id, name, seq) DO UPDATE SET json = excluded.json",
            params![conversation_id, name, seq, raw],
        )
        .map_err(sq_err)?;
        Ok(())
    }

    async fn get_doc_as_of(
        &self,
        conversation_id: &str,
        name: &str,
        max_seq: Option<u64>,
    ) -> Result<Option<(u64, Value)>> {
        let conn = self.conn.lock().unwrap();
        let upper = max_seq.unwrap_or(i64::MAX as u64);
        let row: Option<(u64, String)> = conn
            .query_row(
                "SELECT seq, json FROM docs
                 WHERE conversation_id = ?1 AND name = ?2 AND seq <= ?3
                 ORDER BY seq DESC LIMIT 1",
                params![conversation_id, name, upper],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .map_err(sq_err)?;
        match row {
            Some((seq, s)) => Ok(Some((seq, serde_json::from_str(&s)?))),
            None => Ok(None),
        }
    }

    async fn put_memo(&self, conversation_id: &str, key: &str, value: &Value) -> Result<()> {
        let raw = serde_json::to_string(value)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memos (conversation_id, key, json) VALUES (?1, ?2, ?3)
             ON CONFLICT(conversation_id, key) DO UPDATE SET json = excluded.json",
            params![conversation_id, key, raw],
        )
        .map_err(sq_err)?;
        Ok(())
    }

    async fn get_memo(&self, conversation_id: &str, key: &str) -> Result<Option<Value>> {
        let conn = self.conn.lock().unwrap();
        let raw: Option<String> = conn
            .query_row(
                "SELECT json FROM memos WHERE conversation_id = ?1 AND key = ?2",
                params![conversation_id, key],
                |r| r.get(0),
            )
            .optional()
            .map_err(sq_err)?;
        raw.map(|s| serde_json::from_str(&s).map_err(Into::into))
            .transpose()
    }
}

/// Reusable conformance check for any [`Storage`] implementation.
pub async fn run_storage_conformance(storage: &dyn Storage) -> Result<()> {
    let meta = ConversationMeta {
        id: "conv-root".into(),
        is_root: true,
        model_provider: "hanzo".into(),
        model_id: "enso-auto".into(),
        cwd: PathBuf::from("/work"),
        system_prompt: None,
        ownership: Ownership::Root,
        fork_origin: None,
        compacted_up_to_seq: 0,
        closed: false,
        created_at_ms: 1,
        updated_at_ms: 1,
    };
    storage.put_conversation(&meta).await?;
    assert_eq!(storage.get_root_conversation().await?, Some(meta.clone()));

    let e1 = storage
        .append_entry("conv-root", "e1".into(), EntryKind::User, "alpha bug".into(), Some("req-1".into()), 10)
        .await?;
    let e2 = storage
        .append_entry(
            "conv-root",
            "e2".into(),
            EntryKind::Assistant {
                tool_calls: vec![],
                aborted: false,
            },
            "beta fix".into(),
            None,
            20,
        )
        .await?;
    assert_eq!(e1.seq, 1);
    assert_eq!(e2.seq, 2);
    assert_eq!(storage.latest_seq("conv-root").await?, 2);

    // Bounded working-set slice (`after_seq = 1`).
    let working = storage.load_entries("conv-root", 1, None).await?;
    assert_eq!(working.len(), 1);
    assert_eq!(working[0].id, "e2");

    // Full-history search still finds compacted entry `e1`.
    let hits = storage.search_entries("conv-root", "ALPHA", None).await?;
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, "e1");

    // Versioned docs (`AsOf`).
    storage
        .put_doc("conv-root", "plan", 1, &serde_json::json!({"step": 1}))
        .await?;
    storage
        .put_doc("conv-root", "plan", 2, &serde_json::json!({"step": 2}))
        .await?;
    assert_eq!(
        storage.get_doc_as_of("conv-root", "plan", Some(1)).await?,
        Some((1, serde_json::json!({"step": 1})))
    );
    assert_eq!(
        storage.get_doc_as_of("conv-root", "plan", None).await?,
        Some((2, serde_json::json!({"step": 2})))
    );

    // Request-ID lookup + unfinished task tracking.
    let sub = SubmissionRecord {
        id: "sub-1".into(),
        conversation_id: "conv-root".into(),
        request_id: Some("job-42".into()),
        content: "fix test".into(),
        when_busy: WhenBusy::Queue,
        status: SubmissionStatus::Running,
        answer_entry_id: None,
        answer: None,
        error: None,
        created_at_ms: 10,
        updated_at_ms: 10,
    };
    storage.put_submission(&sub).await?;
    assert_eq!(
        storage
            .find_submission_by_request_id("conv-root", "job-42")
            .await?,
        Some(sub)
    );
    assert_eq!(storage.list_unfinished_submissions().await?.len(), 1);

    let task = TaskRecord {
        id: "task-1".into(),
        conversation_id: "conv-root".into(),
        submission_id: Some("sub-1".into()),
        kind: TaskKind::Custom {
            name: "triage".into(),
        },
        status: TaskStatus::Running,
        checkpoint: Some(serde_json::json!({"phase": "patched"})),
        updated_at_ms: 15,
    };
    storage.put_task(&task).await?;
    assert_eq!(storage.list_unfinished_tasks().await?, vec![task.clone()]);

    storage
        .put_memo("conv-root", "k1", &serde_json::json!(99))
        .await?;
    assert_eq!(
        storage.get_memo("conv-root", "k1").await?,
        Some(serde_json::json!(99))
    );
    Ok(())
}
