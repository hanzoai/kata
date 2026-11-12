use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use crate::{KataError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecOutput {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

#[async_trait]
pub trait ExecutionEnv: Send + Sync {
    fn cwd(&self) -> &Path;
    async fn read_file(&self, path: &Path) -> Result<String>;
    async fn write_file(&self, path: &Path, content: &str) -> Result<()>;
    async fn list_dir(&self, path: &Path) -> Result<Vec<String>>;
    async fn exec(
        &self,
        command: &str,
        timeout: Option<Duration>,
        cancel: &CancellationToken,
    ) -> Result<ExecOutput>;
}

pub type EnvFactory = Arc<dyn Fn(&Path) -> Arc<dyn ExecutionEnv> + Send + Sync>;

fn resolve_path(cwd: &Path, path: &Path) -> PathBuf {
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let mut normalized = PathBuf::new();
    for comp in joined.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

#[derive(Debug, Clone)]
pub struct LocalExecutionEnv {
    cwd: PathBuf,
}

impl LocalExecutionEnv {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self { cwd: cwd.into() }
    }
}

#[async_trait]
impl ExecutionEnv for LocalExecutionEnv {
    fn cwd(&self) -> &Path {
        &self.cwd
    }

    async fn read_file(&self, path: &Path) -> Result<String> {
        let full = resolve_path(&self.cwd, path);
        Ok(tokio::fs::read_to_string(full).await?)
    }

    async fn write_file(&self, path: &Path, content: &str) -> Result<()> {
        let full = resolve_path(&self.cwd, path);
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        Ok(tokio::fs::write(full, content).await?)
    }

    async fn list_dir(&self, path: &Path) -> Result<Vec<String>> {
        let full = resolve_path(&self.cwd, path);
        let mut rd = tokio::fs::read_dir(full).await?;
        let mut names = Vec::new();
        while let Some(entry) = rd.next_entry().await? {
            names.push(entry.file_name().to_string_lossy().into_owned());
        }
        names.sort();
        Ok(names)
    }

    async fn exec(
        &self,
        command: &str,
        timeout: Option<Duration>,
        cancel: &CancellationToken,
    ) -> Result<ExecOutput> {
        if cancel.is_cancelled() {
            return Err(KataError::Aborted);
        }
        let mut cmd = Command::new("/bin/sh");
        cmd.arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .kill_on_drop(true);
        let child = cmd.output();
        let out = match timeout {
            Some(limit) => tokio::select! {
                _ = cancel.cancelled() => return Err(KataError::Aborted),
                res = tokio::time::timeout(limit, child) => {
                    res.map_err(|_| KataError::Tool(format!("command timed out after {limit:?}")))??
                }
            },
            None => tokio::select! {
                _ = cancel.cancelled() => return Err(KataError::Aborted),
                res = child => res?,
            },
        };
        Ok(ExecOutput {
            exit_code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }
}

type ExecHandler = Arc<dyn Fn(&str, &Path) -> Result<ExecOutput> + Send + Sync>;

#[derive(Clone)]
pub struct MemoryExecutionEnv {
    cwd: PathBuf,
    files: Arc<RwLock<BTreeMap<PathBuf, String>>>,
    exec_handler: Option<ExecHandler>,
}

impl MemoryExecutionEnv {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            files: Arc::new(RwLock::new(BTreeMap::new())),
            exec_handler: None,
        }
    }

    pub fn with_exec_handler(
        mut self,
        handler: impl Fn(&str, &Path) -> Result<ExecOutput> + Send + Sync + 'static,
    ) -> Self {
        self.exec_handler = Some(Arc::new(handler));
        self
    }
}

#[async_trait]
impl ExecutionEnv for MemoryExecutionEnv {
    fn cwd(&self) -> &Path {
        &self.cwd
    }

    async fn read_file(&self, path: &Path) -> Result<String> {
        let full = resolve_path(&self.cwd, path);
        let guard = self.files.read().unwrap();
        guard
            .get(&full)
            .cloned()
            .ok_or_else(|| KataError::NotFound(full.display().to_string()))
    }

    async fn write_file(&self, path: &Path, content: &str) -> Result<()> {
        let full = resolve_path(&self.cwd, path);
        self.files.write().unwrap().insert(full, content.to_owned());
        Ok(())
    }

    async fn list_dir(&self, path: &Path) -> Result<Vec<String>> {
        let dir = resolve_path(&self.cwd, path);
        let guard = self.files.read().unwrap();
        let mut names = std::collections::BTreeSet::new();
        for key in guard.keys() {
            if let Ok(rel) = key.strip_prefix(&dir) {
                if let Some(first) = rel.components().next() {
                    names.insert(first.as_os_str().to_string_lossy().into_owned());
                }
            }
        }
        Ok(names.into_iter().collect())
    }

    async fn exec(
        &self,
        command: &str,
        _timeout: Option<Duration>,
        cancel: &CancellationToken,
    ) -> Result<ExecOutput> {
        if cancel.is_cancelled() {
            return Err(KataError::Aborted);
        }
        if let Some(handler) = &self.exec_handler {
            return handler(command, &self.cwd);
        }
        Ok(ExecOutput {
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
        })
    }
}
