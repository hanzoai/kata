use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::env::{EnvFactory, LocalExecutionEnv};
use crate::model::{
    ModelChoice, ModelMessage, ModelRegistry, ModelRequest, Role,
};
use crate::storage::{
    ConversationMeta, EntryKind, ForkOrigin, Ownership, ReplayPolicy, Storage, SubmissionRecord,
    SubmissionStatus, TaskKind, TaskRecord, TaskStatus, TranscriptEntry, WhenBusy,
};
use crate::tools::{ToolContext, ToolRegistry};
use crate::{now_ms, KataError, Result};

const MAX_TOOL_ROUNDS: usize = 32;

#[derive(Clone)]
pub struct HarnessOptions {
    pub models: ModelRegistry,
    pub tools: ToolRegistry,
    pub env: EnvFactory,
}

impl HarnessOptions {
    pub fn new(models: ModelRegistry, tools: ToolRegistry) -> Self {
        Self {
            models,
            tools,
            env: Arc::new(|cwd| Arc::new(LocalExecutionEnv::new(cwd))),
        }
    }

    pub fn with_env(mut self, env: EnvFactory) -> Self {
        self.env = env;
        self
    }
}

#[derive(Debug, Clone)]
pub struct RootOptions {
    pub model: ModelChoice,
    pub cwd: PathBuf,
    pub system_prompt: Option<String>,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            model: ModelChoice {
                provider: "hanzo".into(),
                model_id: "enso-auto".into(),
            },
            cwd: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            system_prompt: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CreateOptions {
    pub model: ModelChoice,
    pub cwd: PathBuf,
    pub system_prompt: Option<String>,
    pub ownership: Ownership,
}

#[derive(Debug, Clone)]
pub struct ForkOptions {
    pub ownership: Ownership,
    pub model: Option<ModelChoice>,
    pub cwd: Option<PathBuf>,
}

impl Default for ForkOptions {
    fn default() -> Self {
        Self {
            ownership: Ownership::Ownerless,
            model: None,
            cwd: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubmissionInput {
    pub content: String,
    pub request_id: Option<String>,
    pub when_busy: WhenBusy,
}

impl SubmissionInput {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            request_id: None,
            when_busy: WhenBusy::Queue,
        }
    }

    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    pub fn with_when_busy(mut self, when_busy: WhenBusy) -> Self {
        self.when_busy = when_busy;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SettledSubmission {
    pub submission_id: String,
    pub status: SubmissionStatus,
    pub answer_entry_id: Option<String>,
    pub answer: Option<String>,
    pub error: Option<String>,
}

impl From<SubmissionRecord> for SettledSubmission {
    fn from(r: SubmissionRecord) -> Self {
        Self {
            submission_id: r.id,
            status: r.status,
            answer_entry_id: r.answer_entry_id,
            answer: r.answer,
            error: r.error,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForkDocPolicy {
    /// Forked conversations inherit the parent's document snapshot as of the fork point.
    AsOf,
    /// Forked conversations start with an empty document.
    Independent,
}

pub struct DocSpec<T> {
    pub name: &'static str,
    pub fork_policy: ForkDocPolicy,
    _marker: PhantomData<fn() -> T>,
}

impl<T> DocSpec<T> {
    pub const fn new(name: &'static str, fork_policy: ForkDocPolicy) -> Self {
        Self {
            name,
            fork_policy,
            _marker: PhantomData,
        }
    }
}

struct ConvRuntime {
    submit_gate: tokio::sync::Mutex<()>,
    queue: Mutex<VecDeque<String>>,
    steer_queue: Mutex<VecDeque<String>>,
    worker_running: Mutex<bool>,
    active_turn: Mutex<Option<(String, CancellationToken)>>,
    conv_cancel: CancellationToken,
}

impl Default for ConvRuntime {
    fn default() -> Self {
        Self {
            submit_gate: tokio::sync::Mutex::new(()),
            queue: Mutex::new(VecDeque::new()),
            steer_queue: Mutex::new(VecDeque::new()),
            worker_running: Mutex::new(false),
            active_turn: Mutex::new(None),
            conv_cancel: CancellationToken::new(),
        }
    }
}

struct HarnessInner {
    storage: Arc<dyn Storage>,
    options: HarnessOptions,
    runtimes: Mutex<HashMap<String, Arc<ConvRuntime>>>,
    watchers: Mutex<HashMap<String, watch::Sender<SubmissionRecord>>>,
}

#[derive(Clone)]
pub struct Harness {
    inner: Arc<HarnessInner>,
}

impl Harness {
    pub async fn open(storage: Arc<dyn Storage>, options: HarnessOptions) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(HarnessInner {
                storage,
                options,
                runtimes: Mutex::new(HashMap::new()),
                watchers: Mutex::new(HashMap::new()),
            }),
        })
    }

    pub fn models(&self) -> &ModelRegistry {
        &self.inner.options.models
    }

    pub fn tools(&self) -> &ToolRegistry {
        &self.inner.options.tools
    }

    pub async fn root(&self, defaults: RootOptions) -> Result<ConversationHandle> {
        if let Some(existing) = self.inner.storage.get_root_conversation().await? {
            return Ok(ConversationHandle {
                id: existing.id,
                harness: self.clone(),
            });
        }
        let now = now_ms();
        let meta = ConversationMeta {
            id: format!("conv-{}", Uuid::now_v7()),
            is_root: true,
            model_provider: defaults.model.provider,
            model_id: defaults.model.model_id,
            cwd: defaults.cwd,
            system_prompt: defaults.system_prompt,
            ownership: Ownership::Root,
            fork_origin: None,
            compacted_up_to_seq: 0,
            closed: false,
            created_at_ms: now,
            updated_at_ms: now,
        };
        self.inner.storage.put_conversation(&meta).await?;
        Ok(ConversationHandle {
            id: meta.id,
            harness: self.clone(),
        })
    }

    pub async fn create(&self, opts: CreateOptions) -> Result<ConversationHandle> {
        let now = now_ms();
        let meta = ConversationMeta {
            id: format!("conv-{}", Uuid::now_v7()),
            is_root: false,
            model_provider: opts.model.provider,
            model_id: opts.model.model_id,
            cwd: opts.cwd,
            system_prompt: opts.system_prompt,
            ownership: opts.ownership,
            fork_origin: None,
            compacted_up_to_seq: 0,
            closed: false,
            created_at_ms: now,
            updated_at_ms: now,
        };
        self.inner.storage.put_conversation(&meta).await?;
        Ok(ConversationHandle {
            id: meta.id,
            harness: self.clone(),
        })
    }

    pub async fn conversation(&self, id: &str) -> Result<ConversationHandle> {
        self.inner
            .storage
            .get_conversation(id)
            .await?
            .ok_or_else(|| KataError::NotFound(format!("conversation `{id}`")))?;
        Ok(ConversationHandle {
            id: id.to_owned(),
            harness: self.clone(),
        })
    }

    /// Recovers unfinished tasks and resumes interrupted/pending submissions from durable storage.
    pub async fn resume(&self) -> Result<()> {
        let unfinished_tasks = self.inner.storage.list_unfinished_tasks().await?;
        for mut task in unfinished_tasks {
            let Some(conv) = self
                .inner
                .storage
                .get_conversation(&task.conversation_id)
                .await?
            else {
                continue;
            };
            match &task.kind {
                TaskKind::ModelCall { partial_output } => {
                    if let Some(partial) = partial_output {
                        if !partial.is_empty() {
                            self.inner
                                .storage
                                .append_entry(
                                    &conv.id,
                                    format!("entry-{}", Uuid::now_v7()),
                                    EntryKind::Assistant {
                                        tool_calls: vec![],
                                        aborted: true,
                                    },
                                    partial.clone(),
                                    None,
                                    now_ms(),
                                )
                                .await?;
                        }
                    }
                    task.status = TaskStatus::Interrupted;
                    task.updated_at_ms = now_ms();
                    self.inner.storage.put_task(&task).await?;
                }
                TaskKind::ToolCall {
                    call_id,
                    tool_name,
                    arguments,
                    replay,
                } => match replay {
                    ReplayPolicy::Safe => {
                        let env = (self.inner.options.env)(&conv.cwd);
                        let ctx = ToolContext {
                            conversation_id: conv.id.clone(),
                            task_id: task.id.clone(),
                            env,
                            cancel: CancellationToken::new(),
                            storage: Arc::clone(&self.inner.storage),
                        };
                        let out = match self
                            .inner
                            .options
                            .tools
                            .execute(tool_name, arguments.clone(), &ctx)
                            .await
                        {
                            Ok(o) => o,
                            Err(e) => crate::tools::ToolOutput::err(e.to_string()),
                        };
                        self.inner
                            .storage
                            .append_entry(
                                &conv.id,
                                format!("entry-{}", Uuid::now_v7()),
                                EntryKind::ToolResult {
                                    call_id: call_id.clone(),
                                    tool_name: tool_name.clone(),
                                    is_error: out.is_error,
                                    interrupted: false,
                                },
                                out.content,
                                None,
                                now_ms(),
                            )
                            .await?;
                        task.status = TaskStatus::Completed;
                        task.updated_at_ms = now_ms();
                        self.inner.storage.put_task(&task).await?;
                    }
                    ReplayPolicy::Interrupt => {
                        self.inner
                            .storage
                            .append_entry(
                                &conv.id,
                                format!("entry-{}", Uuid::now_v7()),
                                EntryKind::ToolResult {
                                    call_id: call_id.clone(),
                                    tool_name: tool_name.clone(),
                                    is_error: true,
                                    interrupted: true,
                                },
                                format!(
                                    "Tool `{tool_name}` was interrupted by a process restart before finishing; inspect current workspace state before retrying."
                                ),
                                None,
                                now_ms(),
                            )
                            .await?;
                        task.status = TaskStatus::Interrupted;
                        task.updated_at_ms = now_ms();
                        self.inner.storage.put_task(&task).await?;
                    }
                },
                TaskKind::Custom { .. } => {
                    task.status = TaskStatus::Interrupted;
                    task.updated_at_ms = now_ms();
                    self.inner.storage.put_task(&task).await?;
                }
            }
        }

        let unfinished_subs = self.inner.storage.list_unfinished_submissions().await?;
        let mut by_conv: HashMap<String, Vec<SubmissionRecord>> = HashMap::new();
        for sub in unfinished_subs {
            self.ensure_watcher(&sub);
            by_conv
                .entry(sub.conversation_id.clone())
                .or_default()
                .push(sub);
        }
        for (conv_id, subs) in by_conv {
            let rt = self.runtime_for(&conv_id);
            {
                let mut q = rt.queue.lock().unwrap();
                for sub in subs {
                    if !q.iter().any(|id| id == &sub.id) {
                        q.push_back(sub.id);
                    }
                }
            }
            self.ensure_worker_started(&conv_id, rt);
        }
        Ok(())
    }

    fn runtime_for(&self, conv_id: &str) -> Arc<ConvRuntime> {
        let mut map = self.inner.runtimes.lock().unwrap();
        Arc::clone(map.entry(conv_id.to_owned()).or_default())
    }

    fn ensure_watcher(&self, record: &SubmissionRecord) -> watch::Receiver<SubmissionRecord> {
        let mut watchers = self.inner.watchers.lock().unwrap();
        if let Some(tx) = watchers.get(&record.id) {
            return tx.subscribe();
        }
        let (tx, rx) = watch::channel(record.clone());
        watchers.insert(record.id.clone(), tx);
        rx
    }

    async fn update_submission(&self, record: &SubmissionRecord) -> Result<()> {
        self.inner.storage.put_submission(record).await?;
        let watchers = self.inner.watchers.lock().unwrap();
        if let Some(tx) = watchers.get(&record.id) {
            let _ = tx.send(record.clone());
        }
        Ok(())
    }

    fn ensure_worker_started(&self, conv_id: &str, rt: Arc<ConvRuntime>) {
        {
            let mut running = rt.worker_running.lock().unwrap();
            if *running {
                return;
            }
            *running = true;
        }
        let harness = self.clone();
        let cid = conv_id.to_owned();
        tokio::spawn(async move {
            harness.worker_loop(cid, rt).await;
        });
    }

    async fn worker_loop(&self, conv_id: String, rt: Arc<ConvRuntime>) {
        loop {
            let next_sub_id = {
                let mut q = rt.queue.lock().unwrap();
                match q.pop_front() {
                    Some(id) => id,
                    None => {
                        *rt.worker_running.lock().unwrap() = false;
                        return;
                    }
                }
            };
            let turn_id = format!("turn-{}", Uuid::now_v7());
            let turn_cancel = rt.conv_cancel.child_token();
            *rt.active_turn.lock().unwrap() = Some((turn_id.clone(), turn_cancel.clone()));
            let _ = self
                .run_submission(&conv_id, &next_sub_id, &turn_id, &rt, &turn_cancel)
                .await;
            *rt.active_turn.lock().unwrap() = None;
        }
    }

    async fn run_submission(
        &self,
        conv_id: &str,
        sub_id: &str,
        turn_id: &str,
        rt: &ConvRuntime,
        turn_cancel: &CancellationToken,
    ) -> Result<()> {
        let Some(mut sub) = self.inner.storage.get_submission(sub_id).await? else {
            return Ok(());
        };
        if sub.status.is_terminal() {
            return Ok(());
        }
        if turn_cancel.is_cancelled() {
            sub.status = SubmissionStatus::Aborted;
            sub.updated_at_ms = now_ms();
            self.update_submission(&sub).await?;
            return Ok(());
        }

        // Only append the user message if it has not already been recorded prior to a crash.
        let user_tag = format!("sub:{}", sub.id);
        let existing_entries = self.inner.storage.load_entries(conv_id, 0, None).await?;
        let already_appended = existing_entries.iter().any(|e| {
            matches!(e.kind, EntryKind::User)
                && (e.request_id.as_deref() == sub.request_id.as_deref() && sub.request_id.is_some()
                    || e.id == user_tag)
        });
        if !already_appended {
            self.inner
                .storage
                .append_entry(
                    conv_id,
                    user_tag,
                    EntryKind::User,
                    sub.content.clone(),
                    sub.request_id.clone(),
                    now_ms(),
                )
                .await?;
        }
        sub.status = SubmissionStatus::Running;
        sub.updated_at_ms = now_ms();
        self.update_submission(&sub).await?;

        let mut steered_subs: Vec<SubmissionRecord> = Vec::new();
        let outcome = self
            .drive_model_steps(conv_id, sub_id, turn_id, rt, turn_cancel, &mut steered_subs)
            .await;

        match outcome {
            Ok((answer_id, answer_text)) => {
                sub.status = SubmissionStatus::Completed;
                sub.answer_entry_id = Some(answer_id.clone());
                sub.answer = Some(answer_text.clone());
                sub.updated_at_ms = now_ms();
                self.update_submission(&sub).await?;
                for mut s in steered_subs {
                    s.status = SubmissionStatus::Completed;
                    s.answer_entry_id = Some(answer_id.clone());
                    s.answer = Some(answer_text.clone());
                    s.updated_at_ms = now_ms();
                    self.update_submission(&s).await?;
                }
            }
            Err(KataError::Aborted) => {
                self.abort_task_owned_children(conv_id, turn_id).await;
                sub.status = SubmissionStatus::Aborted;
                sub.updated_at_ms = now_ms();
                self.update_submission(&sub).await?;
                for mut s in steered_subs {
                    s.status = SubmissionStatus::Aborted;
                    s.updated_at_ms = now_ms();
                    self.update_submission(&s).await?;
                }
            }
            Err(err) => {
                let msg = err.to_string();
                sub.status = SubmissionStatus::Failed;
                sub.error = Some(msg.clone());
                sub.updated_at_ms = now_ms();
                self.update_submission(&sub).await?;
                for mut s in steered_subs {
                    s.status = SubmissionStatus::Failed;
                    s.error = Some(msg.clone());
                    s.updated_at_ms = now_ms();
                    self.update_submission(&s).await?;
                }
            }
        }
        Ok(())
    }

    async fn drain_steers(
        &self,
        conv_id: &str,
        rt: &ConvRuntime,
        steered_subs: &mut Vec<SubmissionRecord>,
    ) -> Result<()> {
        let ids: Vec<String> = {
            let mut sq = rt.steer_queue.lock().unwrap();
            sq.drain(..).collect()
        };
        for id in ids {
            if let Some(mut s) = self.inner.storage.get_submission(&id).await? {
                self.inner
                    .storage
                    .append_entry(
                        conv_id,
                        format!("sub:{}", s.id),
                        EntryKind::User,
                        s.content.clone(),
                        s.request_id.clone(),
                        now_ms(),
                    )
                    .await?;
                s.status = SubmissionStatus::Running;
                s.updated_at_ms = now_ms();
                self.update_submission(&s).await?;
                steered_subs.push(s);
            }
        }
        Ok(())
    }

    async fn drive_model_steps(
        &self,
        conv_id: &str,
        sub_id: &str,
        _turn_id: &str,
        rt: &ConvRuntime,
        turn_cancel: &CancellationToken,
        steered_subs: &mut Vec<SubmissionRecord>,
    ) -> Result<(String, String)> {
        for _ in 0..MAX_TOOL_ROUNDS {
            if turn_cancel.is_cancelled() {
                return Err(KataError::Aborted);
            }
            self.drain_steers(conv_id, rt, steered_subs).await?;

            let conv = self
                .inner
                .storage
                .get_conversation(conv_id)
                .await?
                .ok_or_else(|| KataError::NotFound(conv_id.to_owned()))?;
            let handle = ConversationHandle {
                id: conv.id.clone(),
                harness: self.clone(),
            };
            let entries = handle.working_transcript().await?;
            let messages = entries_to_messages(conv.system_prompt.as_deref(), &entries);
            let tools = self.inner.options.tools.definitions();
            let provider = self.inner.options.models.get(&conv.model_provider)?;

            let model_task_id = format!("task-{}", Uuid::now_v7());
            let mut model_task = TaskRecord {
                id: model_task_id,
                conversation_id: conv.id.clone(),
                submission_id: Some(sub_id.to_owned()),
                kind: TaskKind::ModelCall {
                    partial_output: None,
                },
                status: TaskStatus::Running,
                checkpoint: None,
                updated_at_ms: now_ms(),
            };
            self.inner.storage.put_task(&model_task).await?;

            let resp = match provider
                .complete(
                    ModelRequest {
                        model_id: conv.model_id.clone(),
                        messages,
                        tools,
                    },
                    turn_cancel,
                )
                .await
            {
                Ok(r) => r,
                Err(KataError::Aborted) => {
                    model_task.status = TaskStatus::Interrupted;
                    model_task.updated_at_ms = now_ms();
                    self.inner.storage.put_task(&model_task).await?;
                    return Err(KataError::Aborted);
                }
                Err(e) => {
                    model_task.status = TaskStatus::Failed;
                    model_task.updated_at_ms = now_ms();
                    self.inner.storage.put_task(&model_task).await?;
                    return Err(e);
                }
            };

            model_task.status = TaskStatus::Completed;
            model_task.updated_at_ms = now_ms();
            self.inner.storage.put_task(&model_task).await?;

            let assistant_entry = self
                .inner
                .storage
                .append_entry(
                    &conv.id,
                    format!("entry-{}", Uuid::now_v7()),
                    EntryKind::Assistant {
                        tool_calls: resp.tool_calls.clone(),
                        aborted: false,
                    },
                    resp.content.clone(),
                    None,
                    now_ms(),
                )
                .await?;

            if resp.tool_calls.is_empty() {
                return Ok((assistant_entry.id, resp.content));
            }

            // Checkpoint every tool call before running so a crash knows what was in flight.
            let mut tool_tasks = Vec::with_capacity(resp.tool_calls.len());
            for tc in &resp.tool_calls {
                let replay = self
                    .inner
                    .options
                    .tools
                    .get(&tc.name)
                    .map(|t| t.definition().replay)
                    .unwrap_or(ReplayPolicy::Interrupt);
                let tr = TaskRecord {
                    id: format!("task-{}", Uuid::now_v7()),
                    conversation_id: conv.id.clone(),
                    submission_id: Some(sub_id.to_owned()),
                    kind: TaskKind::ToolCall {
                        call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                        replay,
                    },
                    status: TaskStatus::Running,
                    checkpoint: None,
                    updated_at_ms: now_ms(),
                };
                self.inner.storage.put_task(&tr).await?;
                tool_tasks.push((tc.clone(), tr));
            }

            for (tc, mut tr) in tool_tasks {
                if turn_cancel.is_cancelled() {
                    tr.status = TaskStatus::Interrupted;
                    tr.updated_at_ms = now_ms();
                    self.inner.storage.put_task(&tr).await?;
                    return Err(KataError::Aborted);
                }
                let env = (self.inner.options.env)(&conv.cwd);
                let ctx = ToolContext {
                    conversation_id: conv.id.clone(),
                    task_id: tr.id.clone(),
                    env,
                    cancel: turn_cancel.clone(),
                    storage: Arc::clone(&self.inner.storage),
                };
                let out = match self
                    .inner
                    .options
                    .tools
                    .execute(&tc.name, tc.arguments.clone(), &ctx)
                    .await
                {
                    Ok(o) => o,
                    Err(KataError::Aborted) => {
                        tr.status = TaskStatus::Interrupted;
                        tr.updated_at_ms = now_ms();
                        self.inner.storage.put_task(&tr).await?;
                        return Err(KataError::Aborted);
                    }
                    Err(e) => crate::tools::ToolOutput::err(e.to_string()),
                };
                self.inner
                    .storage
                    .append_entry(
                        &conv.id,
                        format!("entry-{}", Uuid::now_v7()),
                        EntryKind::ToolResult {
                            call_id: tc.id,
                            tool_name: tc.name,
                            is_error: out.is_error,
                            interrupted: false,
                        },
                        out.content,
                        None,
                        now_ms(),
                    )
                    .await?;
                tr.status = TaskStatus::Completed;
                tr.updated_at_ms = now_ms();
                self.inner.storage.put_task(&tr).await?;
            }
        }
        Err(KataError::Model(format!(
            "exceeded max tool rounds ({MAX_TOOL_ROUNDS})"
        )))
    }

    async fn abort_task_owned_children(&self, parent_conv_id: &str, parent_task_id: &str) {
        let Ok(all) = self.inner.storage.list_conversations().await else {
            return;
        };
        for meta in all {
            if let Ownership::Task {
                parent_conversation_id,
                parent_task_id: tid,
            } = &meta.ownership
            {
                if parent_conversation_id == parent_conv_id
                    && (tid == parent_task_id || tid == "*")
                {
                    let child = ConversationHandle {
                        id: meta.id,
                        harness: self.clone(),
                    };
                    let _ = Box::pin(child.abort_turn()).await;
                }
            }
        }
    }
}

fn entries_to_messages(system_prompt: Option<&str>, entries: &[TranscriptEntry]) -> Vec<ModelMessage> {
    let mut out = Vec::new();
    if let Some(sys) = system_prompt {
        out.push(ModelMessage {
            role: Role::System,
            content: sys.to_owned(),
            tool_calls: vec![],
            tool_call_id: None,
        });
    }
    for e in entries {
        match &e.kind {
            EntryKind::User => out.push(ModelMessage {
                role: Role::User,
                content: e.content.clone(),
                tool_calls: vec![],
                tool_call_id: None,
            }),
            EntryKind::Assistant {
                tool_calls,
                aborted,
            } => {
                let text = if *aborted {
                    format!("[aborted partial response]\n{}", e.content)
                } else {
                    e.content.clone()
                };
                out.push(ModelMessage {
                    role: Role::Assistant,
                    content: text,
                    tool_calls: tool_calls.clone(),
                    tool_call_id: None,
                });
            }
            EntryKind::ToolResult { call_id, .. } => out.push(ModelMessage {
                role: Role::Tool,
                content: e.content.clone(),
                tool_calls: vec![],
                tool_call_id: Some(call_id.clone()),
            }),
            EntryKind::Compaction { .. } | EntryKind::Handoff { .. } => out.push(ModelMessage {
                role: Role::System,
                content: e.content.clone(),
                tool_calls: vec![],
                tool_call_id: None,
            }),
        }
    }
    out
}

#[derive(Clone)]
pub struct ConversationHandle {
    id: String,
    harness: Harness,
}

impl ConversationHandle {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub async fn metadata(&self) -> Result<ConversationMeta> {
        self.harness
            .inner
            .storage
            .get_conversation(&self.id)
            .await?
            .ok_or_else(|| KataError::NotFound(self.id.clone()))
    }

    /// Hot-switches this conversation's model/provider in place for subsequent turns.
    pub async fn set_model(&self, model: ModelChoice) -> Result<()> {
        let mut meta = self.metadata().await?;
        meta.model_provider = model.provider;
        meta.model_id = model.model_id;
        meta.updated_at_ms = now_ms();
        self.harness.inner.storage.put_conversation(&meta).await
    }

    pub async fn submit(&self, input: SubmissionInput) -> Result<SubmissionHandle> {
        let rt = self.harness.runtime_for(&self.id);
        let _gate = rt.submit_gate.lock().await;

        if let Some(req_id) = &input.request_id {
            if let Some(existing) = self
                .harness
                .inner
                .storage
                .find_submission_by_request_id(&self.id, req_id)
                .await?
            {
                let rx = self.harness.ensure_watcher(&existing);
                return Ok(SubmissionHandle {
                    id: existing.id,
                    rx,
                });
            }
        }

        let now = now_ms();
        let record = SubmissionRecord {
            id: format!("sub-{}", Uuid::now_v7()),
            conversation_id: self.id.clone(),
            request_id: input.request_id,
            content: input.content,
            when_busy: input.when_busy,
            status: SubmissionStatus::Pending,
            answer_entry_id: None,
            answer: None,
            error: None,
            created_at_ms: now,
            updated_at_ms: now,
        };
        self.harness.inner.storage.put_submission(&record).await?;
        let rx = self.harness.ensure_watcher(&record);

        let busy = rt.active_turn.lock().unwrap().is_some();
        if busy && input.when_busy == WhenBusy::Steer {
            rt.steer_queue.lock().unwrap().push_back(record.id.clone());
        } else {
            rt.queue.lock().unwrap().push_back(record.id.clone());
            self.harness.ensure_worker_started(&self.id, Arc::clone(&rt));
        }

        Ok(SubmissionHandle { id: record.id, rx })
    }

    /// Creates an O(1) copy-on-write fork at `entry_id` without copying parent transcript entries.
    pub async fn fork(&self, entry_id: &str, opts: ForkOptions) -> Result<ConversationHandle> {
        let parent_meta = self.metadata().await?;
        let resolved = self.resolve_entry_in_lineage(entry_id).await?;
        let now = now_ms();
        let model = opts.model.unwrap_or(ModelChoice {
            provider: parent_meta.model_provider,
            model_id: parent_meta.model_id,
        });
        let fork_meta = ConversationMeta {
            id: format!("conv-{}", Uuid::now_v7()),
            is_root: false,
            model_provider: model.provider,
            model_id: model.model_id,
            cwd: opts.cwd.unwrap_or(parent_meta.cwd),
            system_prompt: parent_meta.system_prompt,
            ownership: opts.ownership,
            fork_origin: Some(ForkOrigin {
                parent_conversation_id: resolved.conversation_id,
                up_to_seq: resolved.seq,
                entry_id: resolved.id,
            }),
            compacted_up_to_seq: 0,
            closed: false,
            created_at_ms: now,
            updated_at_ms: now,
        };
        self.harness
            .inner
            .storage
            .put_conversation(&fork_meta)
            .await?;
        Ok(ConversationHandle {
            id: fork_meta.id,
            harness: self.harness.clone(),
        })
    }

    async fn resolve_entry_in_lineage(&self, entry_id: &str) -> Result<TranscriptEntry> {
        let mut cursor_id = self.id.clone();
        let mut max_seq: Option<u64> = None;
        loop {
            let meta = self
                .harness
                .inner
                .storage
                .get_conversation(&cursor_id)
                .await?
                .ok_or_else(|| KataError::NotFound(cursor_id.clone()))?;
            if let Some(entry) = self
                .harness
                .inner
                .storage
                .get_entry(&cursor_id, entry_id)
                .await?
            {
                if max_seq.is_none_or(|cap| entry.seq <= cap) {
                    return Ok(entry);
                }
            }
            let Some(origin) = meta.fork_origin else {
                return Err(KataError::NotFound(format!(
                    "entry `{entry_id}` not found in conversation lineage of `{}`",
                    self.id
                )));
            };
            cursor_id = origin.parent_conversation_id;
            max_seq = Some(origin.up_to_seq);
        }
    }

    /// Returns the active working-set transcript (post-compaction/handoff), resolving CoW forks.
    pub async fn working_transcript(&self) -> Result<Vec<TranscriptEntry>> {
        let meta = self.metadata().await?;
        if meta.compacted_up_to_seq > 0 {
            return self
                .harness
                .inner
                .storage
                .load_entries(&self.id, meta.compacted_up_to_seq, None)
                .await;
        }
        self.full_lineage_entries(None).await
    }

    async fn full_lineage_entries(&self, cap: Option<u64>) -> Result<Vec<TranscriptEntry>> {
        let mut segments: Vec<(String, u64, Option<u64>)> = Vec::new();
        let mut cursor_id = self.id.clone();
        let mut max_seq = cap;
        loop {
            let meta = self
                .harness
                .inner
                .storage
                .get_conversation(&cursor_id)
                .await?
                .ok_or_else(|| KataError::NotFound(cursor_id.clone()))?;
            segments.push((cursor_id.clone(), 0, max_seq));
            let Some(origin) = meta.fork_origin else {
                break;
            };
            cursor_id = origin.parent_conversation_id;
            max_seq = Some(origin.up_to_seq);
        }
        segments.reverse();
        let mut all = Vec::new();
        for (cid, after, upper) in segments {
            let mut part = self
                .harness
                .inner
                .storage
                .load_entries(&cid, after, upper)
                .await?;
            all.append(&mut part);
        }
        Ok(all)
    }

    /// Compacts older history into `summary`, bounding the active in-memory working set.
    pub async fn compact(&self, summary: impl Into<String>) -> Result<TranscriptEntry> {
        let latest = self.harness.inner.storage.latest_seq(&self.id).await?;
        let entry = self
            .harness
            .inner
            .storage
            .append_entry(
                &self.id,
                format!("entry-{}", Uuid::now_v7()),
                EntryKind::Compaction { up_to_seq: latest },
                summary.into(),
                None,
                now_ms(),
            )
            .await?;
        let mut meta = self.metadata().await?;
        // Keep `entry` itself (`entry.seq == latest + 1`) as the start of the working set.
        meta.compacted_up_to_seq = latest;
        meta.updated_at_ms = now_ms();
        self.harness.inner.storage.put_conversation(&meta).await?;
        Ok(entry)
    }

    /// Resets the active context window with an agent-authored `handoff` while keeping full history searchable.
    pub async fn reset_with_handoff(&self, handoff: impl Into<String>) -> Result<TranscriptEntry> {
        let latest = self.harness.inner.storage.latest_seq(&self.id).await?;
        let entry = self
            .harness
            .inner
            .storage
            .append_entry(
                &self.id,
                format!("entry-{}", Uuid::now_v7()),
                EntryKind::Handoff { from_seq: latest },
                handoff.into(),
                None,
                now_ms(),
            )
            .await?;
        let mut meta = self.metadata().await?;
        meta.compacted_up_to_seq = latest;
        meta.updated_at_ms = now_ms();
        self.harness.inner.storage.put_conversation(&meta).await?;
        Ok(entry)
    }

    /// Searches the full uncompacted transcript (including inherited fork history).
    pub async fn search_history(&self, query: &str) -> Result<Vec<TranscriptEntry>> {
        let mut segments: Vec<(String, Option<u64>)> = Vec::new();
        let mut cursor_id = self.id.clone();
        let mut max_seq: Option<u64> = None;
        loop {
            let meta = self
                .harness
                .inner
                .storage
                .get_conversation(&cursor_id)
                .await?
                .ok_or_else(|| KataError::NotFound(cursor_id.clone()))?;
            segments.push((cursor_id.clone(), max_seq));
            let Some(origin) = meta.fork_origin else {
                break;
            };
            cursor_id = origin.parent_conversation_id;
            max_seq = Some(origin.up_to_seq);
        }
        segments.reverse();
        let mut hits = Vec::new();
        for (cid, upper) in segments {
            let mut part = self
                .harness
                .inner
                .storage
                .search_entries(&cid, query, upper)
                .await?;
            hits.append(&mut part);
        }
        Ok(hits)
    }

    pub async fn put_doc<T: Serialize>(&self, spec: &DocSpec<T>, value: &T) -> Result<()> {
        let seq = self.harness.inner.storage.latest_seq(&self.id).await?;
        let raw = serde_json::to_value(value)?;
        self.harness
            .inner
            .storage
            .put_doc(&self.id, spec.name, seq, &raw)
            .await
    }

    pub async fn get_doc<T: DeserializeOwned>(&self, spec: &DocSpec<T>) -> Result<Option<T>> {
        if let Some((_, val)) = self
            .harness
            .inner
            .storage
            .get_doc_as_of(&self.id, spec.name, None)
            .await?
        {
            return Ok(Some(serde_json::from_value(val)?));
        }
        if spec.fork_policy == ForkDocPolicy::Independent {
            return Ok(None);
        }
        let mut cursor_meta = self.metadata().await?;
        while let Some(origin) = cursor_meta.fork_origin {
            if let Some((_, val)) = self
                .harness
                .inner
                .storage
                .get_doc_as_of(
                    &origin.parent_conversation_id,
                    spec.name,
                    Some(origin.up_to_seq),
                )
                .await?
            {
                return Ok(Some(serde_json::from_value(val)?));
            }
            let Some(next) = self
                .harness
                .inner
                .storage
                .get_conversation(&origin.parent_conversation_id)
                .await?
            else {
                break;
            };
            cursor_meta = next;
        }
        Ok(None)
    }

    /// Aborts the currently running turn and cascades to `Ownership::Task`-owned children.
    pub async fn abort_turn(&self) -> Result<()> {
        let rt = self.harness.runtime_for(&self.id);
        let active_turn_id = {
            let guard = rt.active_turn.lock().unwrap();
            if let Some((tid, token)) = guard.as_ref() {
                token.cancel();
                Some(tid.clone())
            } else {
                None
            }
        };
        let all = self.harness.inner.storage.list_conversations().await?;
        for meta in all {
            if let Ownership::Task {
                parent_conversation_id,
                parent_task_id,
            } = &meta.ownership
            {
                if parent_conversation_id == &self.id
                    && (parent_task_id == "*"
                        || active_turn_id.as_deref() == Some(parent_task_id.as_str()))
                {
                    let child = ConversationHandle {
                        id: meta.id,
                        harness: self.harness.clone(),
                    };
                    let _ = Box::pin(child.abort_turn()).await;
                }
            }
        }
        Ok(())
    }

    /// Closes the conversation, cascading to `Task`- and `Conversation`-owned children,
    /// while leaving `Ownerless` forks running independently.
    pub async fn close(&self) -> Result<()> {
        let rt = self.harness.runtime_for(&self.id);
        rt.conv_cancel.cancel();
        let mut meta = self.metadata().await?;
        meta.closed = true;
        meta.updated_at_ms = now_ms();
        self.harness.inner.storage.put_conversation(&meta).await?;

        let all = self.harness.inner.storage.list_conversations().await?;
        for child_meta in all {
            let cascade = match &child_meta.ownership {
                Ownership::Task {
                    parent_conversation_id,
                    ..
                }
                | Ownership::Conversation {
                    parent_conversation_id,
                } => parent_conversation_id == &self.id,
                Ownership::Root | Ownership::Ownerless => false,
            };
            if cascade {
                let child = ConversationHandle {
                    id: child_meta.id,
                    harness: self.harness.clone(),
                };
                let _ = Box::pin(child.close()).await;
            }
        }
        Ok(())
    }
}

pub struct SubmissionHandle {
    id: String,
    rx: watch::Receiver<SubmissionRecord>,
}

impl SubmissionHandle {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn current(&self) -> SubmissionRecord {
        self.rx.borrow().clone()
    }

    pub async fn wait(mut self, cancel: &CancellationToken) -> Result<SettledSubmission> {
        loop {
            {
                let cur = self.rx.borrow().clone();
                if cur.status.is_terminal() {
                    return Ok(cur.into());
                }
            }
            tokio::select! {
                _ = cancel.cancelled() => return Err(KataError::Aborted),
                changed = self.rx.changed() => {
                    if changed.is_err() {
                        return Ok(self.rx.borrow().clone().into());
                    }
                }
            }
        }
    }
}
