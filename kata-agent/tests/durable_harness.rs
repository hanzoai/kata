use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use kata_agent::storage::run_storage_conformance;
use kata_agent::tools::{BoxToolFuture, ToolNext};
use kata_agent::{
    DocSpec, ForkDocPolicy, ForkOptions, Harness, HarnessOptions, JsonlStorage,
    KataError, MemoryExecutionEnv, MemoryStorage, ModelChoice, ModelProvider, ModelRegistry,
    ModelRequest, ModelResponse, Ownership, ReplayPolicy, Result, Role, RootOptions, SqliteStorage,
    Storage, SubmissionInput, SubmissionStatus, TaskKind, TaskRecord, TaskStatus, Tool, ToolCall,
    ToolContext, ToolDef, ToolOutput, ToolRegistry, ToolWrapper, WhenBusy,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tempfile::tempdir;
use tokio_util::sync::CancellationToken;

struct EchoProvider {
    prefix: &'static str,
}

#[async_trait]
impl ModelProvider for EchoProvider {
    async fn complete(
        &self,
        request: ModelRequest,
        cancel: &CancellationToken,
    ) -> Result<ModelResponse> {
        if cancel.is_cancelled() {
            return Err(KataError::Aborted);
        }
        let user_msgs: Vec<&str> = request
            .messages
            .iter()
            .filter(|m| m.role == Role::User)
            .map(|m| m.content.as_str())
            .collect();
        Ok(ModelResponse {
            content: format!(
                "{}({}): {}",
                self.prefix,
                request.model_id,
                user_msgs.join(" | ")
            ),
            tool_calls: vec![],
        })
    }
}

#[tokio::test]
async fn storage_conformance_all_backends() -> Result<()> {
    let mem = MemoryStorage::new();
    run_storage_conformance(&mem).await?;

    let dir = tempdir()?;
    let jsonl_path = dir.path().join("agent.jsonl");
    let jsonl = JsonlStorage::open(&jsonl_path).await?;
    run_storage_conformance(&jsonl).await?;
    // Reopen JSONL from disk and verify durability.
    let jsonl_reopened = JsonlStorage::open(&jsonl_path).await?;
    assert!(jsonl_reopened.get_root_conversation().await?.is_some());
    assert_eq!(jsonl_reopened.latest_seq("conv-root").await?, 2);

    let sqlite_path = dir.path().join("agent.sqlite");
    let sqlite = SqliteStorage::open(&sqlite_path)?;
    run_storage_conformance(&sqlite).await?;
    let sqlite_reopened = SqliteStorage::open(&sqlite_path)?;
    assert!(sqlite_reopened.get_root_conversation().await?.is_some());
    assert_eq!(sqlite_reopened.latest_seq("conv-root").await?, 2);
    Ok(())
}

#[tokio::test]
async fn crash_recovery_and_request_id_idempotency_over_sqlite() -> Result<()> {
    let dir = tempdir()?;
    let db_path = dir.path().join("crash.sqlite");
    let cancel = CancellationToken::new();

    // Simulate a process that persisted a submission ("job-42") and crashed mid-tool-call
    // on both a Safe tool (`read`) and an Interrupt tool (`bash`).
    let root_id = {
        let storage = Arc::new(SqliteStorage::open(&db_path)?);
        let models = ModelRegistry::new();
        let tools = ToolRegistry::new();
        let harness = Harness::open(storage.clone(), HarnessOptions::new(models, tools)).await?;
        let root = harness
            .root(RootOptions {
                model: ModelChoice {
                    provider: "hanzo".into(),
                    model_id: "enso-auto".into(),
                },
                cwd: dir.path().to_path_buf(),
                system_prompt: None,
            })
            .await?;

        std::fs::write(dir.path().join("note.txt"), "recovered-value")?;

        storage
            .append_entry(
                root.id(),
                "sub:sub-crash".into(),
                kata_agent::EntryKind::User,
                "Fix the flaky login test".into(),
                Some("job-42".into()),
                100,
            )
            .await?;
        storage
            .append_entry(
                root.id(),
                "entry-a1".into(),
                kata_agent::EntryKind::Assistant {
                    tool_calls: vec![
                        ToolCall {
                            id: "call-read".into(),
                            name: "read".into(),
                            arguments: json!({"path": "note.txt"}),
                        },
                        ToolCall {
                            id: "call-bash".into(),
                            name: "bash".into(),
                            arguments: json!({"command": "pytest"}),
                        },
                    ],
                    aborted: false,
                },
                String::new(),
                None,
                101,
            )
            .await?;
        storage
            .put_submission(&kata_agent::SubmissionRecord {
                id: "sub-crash".into(),
                conversation_id: root.id().to_owned(),
                request_id: Some("job-42".into()),
                content: "Fix the flaky login test".into(),
                when_busy: WhenBusy::Queue,
                status: SubmissionStatus::Running,
                answer_entry_id: None,
                answer: None,
                error: None,
                created_at_ms: 100,
                updated_at_ms: 100,
            })
            .await?;
        storage
            .put_task(&TaskRecord {
                id: "task-safe-read".into(),
                conversation_id: root.id().to_owned(),
                submission_id: Some("sub-crash".into()),
                kind: TaskKind::ToolCall {
                    call_id: "call-read".into(),
                    tool_name: "read".into(),
                    arguments: json!({"path": "note.txt"}),
                    replay: ReplayPolicy::Safe,
                },
                status: TaskStatus::Running,
                checkpoint: None,
                updated_at_ms: 102,
            })
            .await?;
        storage
            .put_task(&TaskRecord {
                id: "task-unsafe-bash".into(),
                conversation_id: root.id().to_owned(),
                submission_id: Some("sub-crash".into()),
                kind: TaskKind::ToolCall {
                    call_id: "call-bash".into(),
                    tool_name: "bash".into(),
                    arguments: json!({"command": "pytest"}),
                    replay: ReplayPolicy::Interrupt,
                },
                status: TaskStatus::Running,
                checkpoint: None,
                updated_at_ms: 103,
            })
            .await?;
        root.id().to_owned()
    };

    struct InspectingProvider;
    #[async_trait]
    impl ModelProvider for InspectingProvider {
        async fn complete(
            &self,
            request: ModelRequest,
            _cancel: &CancellationToken,
        ) -> Result<ModelResponse> {
            let tool_outputs: Vec<&str> = request
                .messages
                .iter()
                .filter(|m| m.role == Role::Tool)
                .map(|m| m.content.as_str())
                .collect();
            Ok(ModelResponse {
                content: format!("resumed_tools={}", tool_outputs.join(";")),
                tool_calls: vec![],
            })
        }
    }

    // New process opens the same SQLite file, resumes, and client retries `job-42`.
    let storage2 = Arc::new(SqliteStorage::open(&db_path)?);
    let models2 = ModelRegistry::new();
    models2.set_provider("hanzo", Arc::new(InspectingProvider));
    let tools2 = ToolRegistry::new();
    tools2.install_coding_tools();

    let harness2 = Harness::open(storage2.clone(), HarnessOptions::new(models2, tools2)).await?;
    harness2.resume().await?;

    let root2 = harness2.root(RootOptions::default()).await?;
    assert_eq!(root2.id(), root_id);

    let job = SubmissionInput::new("Fix the flaky login test").with_request_id("job-42");
    let settled = root2.submit(job.clone()).await?.wait(&cancel).await?;
    assert_eq!(settled.submission_id, "sub-crash");
    assert_eq!(settled.status, SubmissionStatus::Completed);
    let ans = settled.answer.as_deref().unwrap();
    assert!(ans.contains("recovered-value"), "got: {ans}");
    assert!(ans.contains("interrupted by a process restart"), "got: {ans}");

    // Submitting `job-42` a third time is strictly idempotent and adds zero new user entries.
    let again = root2.submit(job).await?.wait(&cancel).await?;
    assert_eq!(again, settled);
    let user_count = root2
        .working_transcript()
        .await?
        .iter()
        .filter(|e| matches!(e.kind, kata_agent::EntryKind::User))
        .count();
    assert_eq!(user_count, 1);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PlanDoc {
    version: u32,
    goal: String,
}

const PLAN_DOC: DocSpec<PlanDoc> = DocSpec::new("plan", ForkDocPolicy::AsOf);

#[tokio::test]
async fn cow_fork_as_of_docs_compaction_and_search_history() -> Result<()> {
    let cancel = CancellationToken::new();
    let storage = Arc::new(MemoryStorage::new());
    let models = ModelRegistry::new();
    models.set_provider("hanzo", Arc::new(EchoProvider { prefix: "hanzo" }));
    models.set_provider("local", Arc::new(EchoProvider { prefix: "local" }));
    let tools = ToolRegistry::new();
    let harness = Harness::open(storage.clone(), HarnessOptions::new(models, tools)).await?;

    let channel = harness.root(RootOptions::default()).await?;
    let first = channel
        .submit(SubmissionInput::new("why did deploy-9000 fail?"))
        .await?
        .wait(&cancel)
        .await?;
    let fork_point = first.answer_entry_id.clone().unwrap();

    // Save v1 plan at the fork point, then advance the channel and write v2 plan.
    channel
        .put_doc(
            &PLAN_DOC,
            &PlanDoc {
                version: 1,
                goal: "investigate deploy-9000".into(),
            },
        )
        .await?;

    let thread = channel
        .fork(
            &fork_point,
            ForkOptions {
                ownership: Ownership::Ownerless,
                model: None,
                cwd: None,
            },
        )
        .await?;

    // O(1) CoW check: thread's own underlying entry table is empty before its first turn!
    assert_eq!(storage.latest_seq(thread.id()).await?, 0);

    // Channel moves on and updates plan to v2.
    channel
        .submit(SubmissionInput::new("meanwhile check metrics"))
        .await?
        .wait(&cancel)
        .await?;
    channel
        .put_doc(
            &PLAN_DOC,
            &PlanDoc {
                version: 2,
                goal: "metrics audit".into(),
            },
        )
        .await?;

    // Forked thread still sees v1 plan (`AsOf` fork_point) and only parent history up to fork_point!
    assert_eq!(
        thread.get_doc(&PLAN_DOC).await?,
        Some(PlanDoc {
            version: 1,
            goal: "investigate deploy-9000".into()
        })
    );

    // Hot-switch the forked thread to local Zen 6 Nano mid-session!
    thread
        .set_model(ModelChoice {
            provider: "local".into(),
            model_id: "zen6-nano".into(),
        })
        .await?;
    let thread_reply = thread
        .submit(SubmissionInput::new("can we roll it back?"))
        .await?
        .wait(&cancel)
        .await?;
    let text = thread_reply.answer.unwrap();
    assert!(
        text.starts_with("local(zen6-nano): why did deploy-9000 fail? | can we roll it back?"),
        "got: {text}"
    );
    assert!(!text.contains("metrics"));

    // Compact channel, verify working set shrinks while `search_history` still finds `deploy-9000`.
    channel.compact("Summary: deploy-9000 triaged.").await?;
    let working = channel.working_transcript().await?;
    assert_eq!(working.len(), 1);
    let hits = channel.search_history("deploy-9000").await?;
    assert!(!hits.is_empty());

    // Agent-authored `reset_with_handoff` starts clean context while preserving searchable history.
    channel
        .reset_with_handoff("Handoff: proceed to canary rollout.")
        .await?;
    let post_handoff = channel.working_transcript().await?;
    assert_eq!(post_handoff.len(), 1);
    assert!(post_handoff[0].content.contains("canary rollout"));
    assert!(!channel.search_history("metrics").await?.is_empty());
    Ok(())
}

#[tokio::test]
async fn three_tier_ownership_abort_cascade_and_steer_and_tools() -> Result<()> {
    let cancel = CancellationToken::new();
    let storage = Arc::new(MemoryStorage::new());
    let models = ModelRegistry::new();

    struct SlowProvider;
    #[async_trait]
    impl ModelProvider for SlowProvider {
        async fn complete(
            &self,
            _req: ModelRequest,
            cancel: &CancellationToken,
        ) -> Result<ModelResponse> {
            tokio::select! {
                _ = cancel.cancelled() => Err(KataError::Aborted),
                _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                    Ok(ModelResponse { content: "done".into(), tool_calls: vec![] })
                }
            }
        }
    }
    models.set_provider("slow", Arc::new(SlowProvider));

    let tools = ToolRegistry::new();
    let harness = Harness::open(storage, HarnessOptions::new(models, tools)).await?;

    let parent = harness
        .root(RootOptions {
            model: ModelChoice {
                provider: "slow".into(),
                model_id: "slow-1".into(),
            },
            cwd: PathBuf::from("/work"),
            system_prompt: None,
        })
        .await?;
    let task_child = harness
        .create(kata_agent::CreateOptions {
            model: ModelChoice {
                provider: "slow".into(),
                model_id: "slow-1".into(),
            },
            cwd: PathBuf::from("/work"),
            system_prompt: None,
            ownership: Ownership::Task {
                parent_conversation_id: parent.id().to_owned(),
                parent_task_id: "*".into(),
            },
        })
        .await?;
    let conv_child = harness
        .create(kata_agent::CreateOptions {
            model: ModelChoice {
                provider: "slow".into(),
                model_id: "slow-1".into(),
            },
            cwd: PathBuf::from("/work"),
            system_prompt: None,
            ownership: Ownership::Conversation {
                parent_conversation_id: parent.id().to_owned(),
            },
        })
        .await?;
    let ownerless_child = harness
        .create(kata_agent::CreateOptions {
            model: ModelChoice {
                provider: "slow".into(),
                model_id: "slow-1".into(),
            },
            cwd: PathBuf::from("/work"),
            system_prompt: None,
            ownership: Ownership::Ownerless,
        })
        .await?;

    let p_sub = parent.submit(SubmissionInput::new("work")).await?;
    let t_sub = task_child.submit(SubmissionInput::new("subtask")).await?;
    let c_sub = conv_child.submit(SubmissionInput::new("companion")).await?;
    let o_sub = ownerless_child
        .submit(SubmissionInput::new("detached"))
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    // 1. `abort_turn` aborts parent + `Task`-owned child, leaving `Conversation` and `Ownerless` running.
    parent.abort_turn().await?;
    assert_eq!(p_sub.wait(&cancel).await?.status, SubmissionStatus::Aborted);
    assert_eq!(t_sub.wait(&cancel).await?.status, SubmissionStatus::Aborted);
    assert_eq!(c_sub.current().status, SubmissionStatus::Running);
    assert_eq!(o_sub.current().status, SubmissionStatus::Running);

    // 2. `close` aborts `Conversation`-owned child, leaving `Ownerless` untouched!
    parent.close().await?;
    assert_eq!(c_sub.wait(&cancel).await?.status, SubmissionStatus::Aborted);
    assert_eq!(o_sub.current().status, SubmissionStatus::Running);
    ownerless_child.abort_turn().await?;
    assert_eq!(o_sub.wait(&cancel).await?.status, SubmissionStatus::Aborted);

    // 3. Verify `CodingTools` + `wrap_tool` + durable `memo` + `WhenBusy::Steer`.
    let mem_env = Arc::new(MemoryExecutionEnv::new("/sandbox"));
    let wrapped_calls = Arc::new(Mutex::new(Vec::<String>::new()));
    struct AuditWrapper(Arc<Mutex<Vec<String>>>);
    impl ToolWrapper for AuditWrapper {
        fn wrap<'a>(
            &self,
            tool_name: &'a str,
            args: Value,
            ctx: &'a ToolContext,
            next: ToolNext<'a>,
        ) -> BoxToolFuture<'a> {
            self.0.lock().unwrap().push(tool_name.to_owned());
            next(args, ctx)
        }
    }

    let memo_runs = Arc::new(AtomicUsize::new(0));
    struct MemoTool(Arc<AtomicUsize>);
    #[async_trait]
    impl Tool for MemoTool {
        fn definition(&self) -> ToolDef {
            ToolDef {
                name: "expensive_lookup".into(),
                description: "Memoized lookup".into(),
                parameters: json!({"type": "object"}),
                replay: ReplayPolicy::Safe,
            }
        }
        async fn call(&self, _args: Value, ctx: &ToolContext) -> Result<ToolOutput> {
            let counter = Arc::clone(&self.0);
            let val: u64 = ctx
                .memo("lookup-key", || async move {
                    counter.fetch_add(1, Ordering::SeqCst);
                    Ok(42)
                })
                .await?;
            Ok(ToolOutput::ok(val.to_string()))
        }
    }

    struct ScriptedCodingProvider {
        step: AtomicUsize,
    }
    #[async_trait]
    impl ModelProvider for ScriptedCodingProvider {
        async fn complete(
            &self,
            req: ModelRequest,
            _cancel: &CancellationToken,
        ) -> Result<ModelResponse> {
            let idx = self.step.fetch_add(1, Ordering::SeqCst);
            match idx {
                0 => {
                    tokio::time::sleep(std::time::Duration::from_millis(40)).await;
                    Ok(ModelResponse {
                        content: String::new(),
                        tool_calls: vec![
                            ToolCall {
                                id: "c1".into(),
                                name: "write".into(),
                                arguments: json!({"path": "hello.rs", "content": "fn main() {}"}),
                            },
                            ToolCall {
                                id: "c2".into(),
                                name: "expensive_lookup".into(),
                                arguments: json!({}),
                            },
                        ],
                    })
                }
                1 => Ok(ModelResponse {
                    content: String::new(),
                    tool_calls: vec![
                        ToolCall {
                            id: "c3".into(),
                            name: "edit".into(),
                            arguments: json!({
                                "path": "hello.rs",
                                "old_text": "fn main() {}",
                                "new_text": "fn main() { println!(\"steered\"); }"
                            }),
                        },
                        ToolCall {
                            id: "c4".into(),
                            name: "expensive_lookup".into(),
                            arguments: json!({}),
                        },
                    ],
                }),
                _ => {
                    let saw_steer = req
                        .messages
                        .iter()
                        .any(|m| m.role == Role::User && m.content.contains("add println"));
                    Ok(ModelResponse {
                        content: format!("finished(steered={saw_steer})"),
                        tool_calls: vec![],
                    })
                }
            }
        }
    }

    let models2 = ModelRegistry::new();
    models2.set_provider(
        "scripted",
        Arc::new(ScriptedCodingProvider {
            step: AtomicUsize::new(0),
        }),
    );
    let tools2 = ToolRegistry::new();
    tools2.install_coding_tools();
    tools2.register(Arc::new(MemoTool(Arc::clone(&memo_runs))));
    tools2.wrap_tool(Arc::new(AuditWrapper(Arc::clone(&wrapped_calls))));
    let env_clone = Arc::clone(&mem_env);
    let harness2 = Harness::open(
        Arc::new(MemoryStorage::new()),
        HarnessOptions::new(models2, tools2)
            .with_env(Arc::new(move |_| env_clone.clone() as Arc<dyn kata_agent::ExecutionEnv>)),
    )
    .await?;

    let coder = harness2
        .root(RootOptions {
            model: ModelChoice {
                provider: "scripted".into(),
                model_id: "zen6-coder".into(),
            },
            cwd: PathBuf::from("/sandbox"),
            system_prompt: None,
        })
        .await?;

    let main_run = coder.submit(SubmissionInput::new("create hello.rs")).await?;
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    let steer_run = coder
        .submit(SubmissionInput::new("also add println").with_when_busy(WhenBusy::Steer))
        .await?;

    let main_settled = main_run.wait(&cancel).await?;
    let steer_settled = steer_run.wait(&cancel).await?;
    assert_eq!(main_settled.answer.as_deref(), Some("finished(steered=true)"));
    assert_eq!(steer_settled.answer, main_settled.answer);
    assert_eq!(memo_runs.load(Ordering::SeqCst), 1);
    assert_eq!(
        *wrapped_calls.lock().unwrap(),
        vec!["write", "expensive_lookup", "edit", "expensive_lookup"]
    );
    assert_eq!(
        kata_agent::ExecutionEnv::read_file(mem_env.as_ref(), std::path::Path::new("hello.rs")).await?,
        "fn main() { println!(\"steered\"); }"
    );
    Ok(())
}
