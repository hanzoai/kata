//! Closed-loop Detect -> Triage -> Synthesize -> Verify repair orchestrator.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::env::{ExecOutput, ExecutionEnv, LocalExecutionEnv};
use crate::model::{
    ModelChoice, ModelProvider, ModelRegistry, ModelRequest, ModelResponse, Role, ToolCall,
};
use crate::storage::MemoryStorage;
use crate::tools::ToolRegistry;
use crate::{Harness, HarnessOptions, KataError, Result, RootOptions, SubmissionInput};

/// Configuration for running a closed-loop repair workflow on a fixture directory.
#[derive(Debug, Clone)]
pub struct RepairConfig {
    pub fixture_dir: PathBuf,
    pub max_attempts: u32,
    pub test_command: Option<String>,
}

impl RepairConfig {
    pub fn new(fixture_dir: impl Into<PathBuf>) -> Self {
        Self {
            fixture_dir: fixture_dir.into(),
            max_attempts: 3,
            test_command: None,
        }
    }

    pub fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    pub fn with_test_command(mut self, cmd: impl Into<String>) -> Self {
        self.test_command = Some(cmd.into());
        self
    }
}

/// Outcome of a closed-loop Detect -> Triage -> Synthesize -> Verify run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairOutcome {
    pub passed: bool,
    pub escalated: bool,
    pub attempts_used: u32,
    pub initial_stderr: String,
    pub final_stdout: String,
    pub transcript_len: usize,
    pub stages: Vec<String>,
}

/// Default single-file Rust test compile-and-run command for fixtures containing `lib.rs`.
pub fn default_rust_fixture_test_cmd() -> &'static str {
    #[cfg(windows)]
    {
        "rustc --test lib.rs -o kata_fixture_test.exe && kata_fixture_test.exe"
    }
    #[cfg(not(windows))]
    {
        "rustc --test lib.rs -o kata_fixture_test && ./kata_fixture_test"
    }
}

struct DeterministicRepairProvider;

#[async_trait]
impl ModelProvider for DeterministicRepairProvider {
    async fn complete(
        &self,
        request: ModelRequest,
        cancel: &CancellationToken,
    ) -> Result<ModelResponse> {
        if cancel.is_cancelled() {
            return Err(KataError::Aborted);
        }
        let has_read = request
            .messages
            .iter()
            .any(|m| m.role == Role::Assistant && m.tool_calls.iter().any(|tc| tc.name == "read"));
        let has_edit = request
            .messages
            .iter()
            .any(|m| m.role == Role::Assistant && m.tool_calls.iter().any(|tc| tc.name == "edit"));

        if !has_read {
            return Ok(ModelResponse {
                content: "Reading `lib.rs` to triage the failing assertion.".into(),
                tool_calls: vec![ToolCall {
                    id: "call-read-lib".into(),
                    name: "read".into(),
                    arguments: json!({ "path": "lib.rs" }),
                }],
            });
        }

        if !has_edit {
            let source = request
                .messages
                .iter()
                .rev()
                .find(|m| m.role == Role::Tool)
                .map(|m| m.content.as_str())
                .unwrap_or("");
            if source.contains("pub fn add(a: i32, b: i32) -> i32") && source.contains("    a - b")
            {
                return Ok(ModelResponse {
                    content: "Synthesizing minimal fix in `lib.rs`: replace `a - b` with `a + b`."
                        .into(),
                    tool_calls: vec![ToolCall {
                        id: "call-edit-lib".into(),
                        name: "edit".into(),
                        arguments: json!({
                            "path": "lib.rs",
                            "old_text": "    a - b",
                            "new_text": "    a + b",
                        }),
                    }],
                });
            }
            return Ok(ModelResponse {
                content: "No deterministic patch pattern matched the fixture failure.".into(),
                tool_calls: vec![],
            });
        }

        Ok(ModelResponse {
            content: "Applied targeted fix to `lib.rs`; ready to re-run verification tests.".into(),
            tool_calls: vec![],
        })
    }
}

fn copy_fixture_files(src: &Path, dst: &Path) -> Result<()> {
    if !src.is_dir() {
        return Err(KataError::InvalidArgument(format!(
            "fixture directory does not exist: {}",
            src.display()
        )));
    }
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_file() {
            std::fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }
    Ok(())
}

fn format_exec_output(out: &ExecOutput) -> String {
    let stdout = out.stdout.trim();
    let stderr = out.stderr.trim();
    match (stdout.is_empty(), stderr.is_empty()) {
        (true, true) => String::new(),
        (false, true) => stdout.to_owned(),
        (true, false) => stderr.to_owned(),
        (false, false) => format!("{stdout}\n{stderr}"),
    }
}

struct ScopedWorkdir(PathBuf);

impl ScopedWorkdir {
    fn create() -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("kata-repair-{}", Uuid::now_v7()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScopedWorkdir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Runs the Detect -> Triage -> Synthesize -> Verify closed repair loop inside an isolated temp copy.
pub async fn run_closed_loop_repair(config: &RepairConfig) -> Result<RepairOutcome> {
    let workdir = ScopedWorkdir::create()?;
    copy_fixture_files(&config.fixture_dir, workdir.path())?;

    let test_cmd: &str = match config.test_command.as_deref() {
        Some(cmd) => cmd,
        None => default_rust_fixture_test_cmd(),
    };
    let cancel = CancellationToken::new();
    let env = LocalExecutionEnv::new(workdir.path());

    let mut stages = Vec::new();
    let first = env.exec(test_cmd, None, &cancel).await?;
    let initial_output = format_exec_output(&first);
    stages.push(format!("detect(exit={})", first.exit_code));

    if first.exit_code == 0 {
        return Ok(RepairOutcome {
            passed: true,
            escalated: false,
            attempts_used: 0,
            initial_stderr: String::new(),
            final_stdout: initial_output,
            transcript_len: 0,
            stages,
        });
    }

    let storage = Arc::new(MemoryStorage::new());
    let models = ModelRegistry::new();
    models.set_provider("kata-local", Arc::new(DeterministicRepairProvider));
    let tools = ToolRegistry::new();
    tools.install_coding_tools();

    let harness = Harness::open(storage, HarnessOptions::new(models, tools)).await?;
    let conv = harness
        .root(RootOptions {
            model: ModelChoice {
                provider: "kata-local".into(),
                model_id: "deterministic-repair".into(),
            },
            cwd: workdir.path().to_path_buf(),
            system_prompt: Some(
                "You are Hanzo Kata. Triage test failures and apply minimal edits.".into(),
            ),
        })
        .await?;

    let mut latest_output = initial_output.clone();
    let mut attempts_used = 0u32;

    for attempt in 1..=config.max_attempts {
        attempts_used = attempt;
        stages.push(format!("triage(attempt={attempt})"));
        let prompt = format!(
            "Attempt {attempt}/{}: command `{test_cmd}` failed:\n{latest_output}",
            config.max_attempts
        );
        let handle = conv.submit(SubmissionInput::new(prompt)).await?;
        let settled = handle.wait(&cancel).await?;
        stages.push(format!("synthesize({:?})", settled.status));

        let verify = env.exec(test_cmd, None, &cancel).await?;
        latest_output = format_exec_output(&verify);
        stages.push(format!("verify(exit={})", verify.exit_code));

        if verify.exit_code == 0 {
            let transcript = conv.working_transcript().await?;
            return Ok(RepairOutcome {
                passed: true,
                escalated: false,
                attempts_used,
                initial_stderr: initial_output,
                final_stdout: latest_output,
                transcript_len: transcript.len(),
                stages,
            });
        }
    }

    stages.push("escalate".into());
    let transcript = conv.working_transcript().await?;
    Ok(RepairOutcome {
        passed: false,
        escalated: true,
        attempts_used,
        initial_stderr: initial_output,
        final_stdout: latest_output,
        transcript_len: transcript.len(),
        stages,
    })
}
