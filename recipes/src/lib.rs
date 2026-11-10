//! Hanzo Kata Recipes: Autonomous Coding Agent Loop & CI/CD Triage Patterns.
//!
//! State machines for autonomous repo exploration, patch generation, test gating,
//! and self-healing repair loops.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// State of a failing repository or CI build to triage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodingTriageState {
    pub repo_name: String,
    pub branch: String,
    pub failing_step: String,
    pub stdout_stderr: String,
    pub files_modified: Vec<String>,
    pub max_attempts: u32,
    pub current_attempt: u32,
}

/// Action verdict produced by the triage engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriageAction {
    ApplyPatch {
        target_file: String,
        patch_diff: String,
    },
    ReRunTests {
        filter: Option<String>,
    },
    EscalateToHuman {
        reason: String,
    },
}

/// 1. Kata Autonomous Repair Recipe.
pub struct KataAgentLoopRecipe;

impl KataAgentLoopRecipe {
    /// Builds a prompt payload for a frontier coding LLM (e.g. `zen5-coder` or `enso`).
    pub fn build_patch_prompt(state: &CodingTriageState) -> Value {
        json!({
            "model": "zen5-coder",
            "messages": [
                {
                    "role": "system",
                    "content": "You are Hanzo Kata, an expert autonomous software engineer. Generate minimal unified diff patches that fix bugs without altering unrelated logic or style."
                },
                {
                    "role": "user",
                    "content": format!(
                        "Repository: {}\nFailing Step: {}\nAttempt: {}/{}\n\nError Output:\n```\n{}\n```\n\nModified Files: {:?}\n\nProvide the exact patch in unified diff format.",
                        state.repo_name,
                        state.failing_step,
                        state.current_attempt,
                        state.max_attempts,
                        state.stdout_stderr,
                        state.files_modified
                    )
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        })
    }

    /// Evaluates test execution exit code to decide the next loop state.
    pub fn evaluate_test_result(
        exit_code: i32,
        current_attempt: u32,
        max_attempts: u32,
    ) -> TriageAction {
        if exit_code == 0 {
            TriageAction::ReRunTests { filter: None }
        } else if current_attempt >= max_attempts {
            TriageAction::EscalateToHuman {
                reason: format!("Exceeded maximum repair attempts ({max_attempts}) without passing tests"),
            }
        } else {
            TriageAction::ApplyPatch {
                target_file: "src/lib.rs".into(),
                patch_diff: "// automated patch applied".into(),
            }
        }
    }
}

/// Target backend for `hybrid/auto` Virtual Model routing inside Kata microVMs and host orchestrators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendTarget {
    /// Local inference engine (e.g. `qwen3.8-27b` running in `llama.cpp`).
    Local,
    /// Remote frontier cloud backend (e.g. `enso-pro`, `moonshotai/kimi-k3.1-pro`, `glm-5.3`).
    Remote,
}

/// Host or microVM load snapshot for `hybrid/auto` congestion-aware routing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MachineLoad {
    /// Normalized 1-minute CPU load (0.0 ..= 1.0+).
    pub cpu_load: f64,
    /// Normalized memory/VRAM pressure (0.0 ..= 1.0).
    pub memory_pressure: f64,
    /// Active local `llama.cpp` / microVM inference slots in flight.
    pub active_local_requests: u32,
}

impl MachineLoad {
    /// Effective load factor in `[0.0, 1.0]`.
    pub fn effective_load(&self) -> f64 {
        let base = self.cpu_load.max(self.memory_pressure * 0.85);
        let queue_penalty = (self.active_local_requests as f64) * 0.20;
        (base + queue_penalty).clamp(0.0, 1.0)
    }
}

/// Virtual Model (`hybrid/auto`) configuration for Kata agent recipes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirtualModelRouter {
    pub virtual_model: String,
    pub local_model: String,
    pub local_endpoint: String,
    pub remote_model: String,
    pub remote_fallback_model: String,
    pub congestion_threshold: f64,
    pub busy_threshold: f64,
    pub complexity_threshold: f64,
}

impl Default for VirtualModelRouter {
    fn default() -> Self {
        Self {
            virtual_model: "hybrid/auto".into(),
            local_model: "qwen3.8-27b".into(),
            local_endpoint: "http://127.0.0.1:8080/v1".into(),
            remote_model: "enso-pro".into(),
            remote_fallback_model: "moonshotai/kimi-k3.1-pro".into(),
            congestion_threshold: 0.75,
            busy_threshold: 0.50,
            complexity_threshold: 0.45,
        }
    }
}

/// Routing verdict produced by [`VirtualModelRouter::route`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridRouteDecision {
    pub requested_model: String,
    pub target: BackendTarget,
    pub resolved_model: String,
    pub skipped_classifier: bool,
    pub complexity_score: f64,
    pub effective_load: f64,
    pub reason: String,
}

impl VirtualModelRouter {
    /// Returns true if `model` is a virtual hybrid model (`hybrid/auto`, `hybrid`, `enso-hybrid`, `zen-hybrid`).
    pub fn is_virtual_model(model: &str) -> bool {
        matches!(
            model.trim().to_ascii_lowercase().as_str(),
            "hybrid/auto" | "hybrid" | "enso-hybrid" | "zen-hybrid" | "hanzo/hybrid"
        )
    }

    /// Estimate query complexity in `[0.0, 1.0]` when `llama.cpp` Classifier API score is not pre-supplied.
    pub fn classify_complexity(prompt: &str) -> f64 {
        let lower = prompt.to_ascii_lowercase();
        let words = lower.split_whitespace().count();
        let mut score: f64 = match words {
            0..=15 => 0.15,
            16..=60 => 0.30,
            61..=180 => 0.50,
            _ => 0.70,
        };
        for kw in [
            "architecture",
            "distributed",
            "consensus",
            "refactor",
            "multi-file",
            "deadlock",
            "race condition",
            "security audit",
            "formal verification",
            "migration",
            "scalability",
        ] {
            if lower.contains(kw) {
                score += 0.25;
            }
        }
        score.clamp(0.0, 1.0)
    }

    /// Route a query across local (`qwen3.8-27b` in `llama.cpp`) and remote (`enso-pro` / `kimi-k3.1-pro`).
    ///
    /// When the machine is heavily congested (`effective_load >= congestion_threshold`), skips the
    /// local classifier step entirely and goes straight to the remote model.
    pub fn route(
        &self,
        prompt: &str,
        load: MachineLoad,
        classifier_score: Option<f64>,
    ) -> HybridRouteDecision {
        let eff = load.effective_load();
        if eff >= self.congestion_threshold {
            return HybridRouteDecision {
                requested_model: self.virtual_model.clone(),
                target: BackendTarget::Remote,
                resolved_model: self.remote_model.clone(),
                skipped_classifier: true,
                complexity_score: 1.0,
                effective_load: eff,
                reason: format!(
                    "machine congested (load {:.2} >= {:.2}): skipped local classifier, routed straight to remote {}",
                    eff, self.congestion_threshold, self.remote_model
                ),
            };
        }

        let complexity = classifier_score
            .map(|s| s.clamp(0.0, 1.0))
            .unwrap_or_else(|| Self::classify_complexity(prompt));
        let threshold = if eff >= self.busy_threshold {
            (self.complexity_threshold * 0.65).max(0.20)
        } else {
            self.complexity_threshold
        };

        if complexity >= threshold {
            HybridRouteDecision {
                requested_model: self.virtual_model.clone(),
                target: BackendTarget::Remote,
                resolved_model: self.remote_model.clone(),
                skipped_classifier: false,
                complexity_score: complexity,
                effective_load: eff,
                reason: format!(
                    "complexity {:.2} >= threshold {:.2} (load {:.2}): routed to remote {}",
                    complexity, threshold, eff, self.remote_model
                ),
            }
        } else {
            HybridRouteDecision {
                requested_model: self.virtual_model.clone(),
                target: BackendTarget::Local,
                resolved_model: self.local_model.clone(),
                skipped_classifier: false,
                complexity_score: complexity,
                effective_load: eff,
                reason: format!(
                    "simple query (complexity {:.2} < threshold {:.2}, load {:.2}): stayed local on {}",
                    complexity, threshold, eff, self.local_model
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kata_patch_prompt_building() {
        let state = CodingTriageState {
            repo_name: "hanzoai/gateway".into(),
            branch: "fix/timeout".into(),
            failing_step: "cargo test --all".into(),
            stdout_stderr: "error[E0308]: mismatched types: expected usize, found u64".into(),
            files_modified: vec!["src/connection.rs".into()],
            max_attempts: 3,
            current_attempt: 1,
        };

        let payload = KataAgentLoopRecipe::build_patch_prompt(&state);
        assert_eq!(payload["model"], "zen5-coder");
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert!(messages[1]["content"].as_str().unwrap().contains("error[E0308]"));
    }

    #[test]
    fn test_kata_test_evaluation_loop() {
        let passing = KataAgentLoopRecipe::evaluate_test_result(0, 1, 3);
        assert_eq!(passing, TriageAction::ReRunTests { filter: None });

        let failing_retry = KataAgentLoopRecipe::evaluate_test_result(1, 1, 3);
        match failing_retry {
            TriageAction::ApplyPatch { .. } => {}
            _ => panic!("Expected ApplyPatch on retryable failure"),
        }

        let failing_exhausted = KataAgentLoopRecipe::evaluate_test_result(1, 3, 3);
        match failing_exhausted {
            TriageAction::EscalateToHuman { reason } => {
                assert!(reason.contains("Exceeded maximum repair attempts"));
            }
            _ => panic!("Expected EscalateToHuman when attempts are exhausted"),
        }
    }

    #[test]
    fn test_virtual_model_hybrid_auto_routing() {
        let router = VirtualModelRouter::default();
        assert!(VirtualModelRouter::is_virtual_model("hybrid/auto"));
        assert!(VirtualModelRouter::is_virtual_model("enso-hybrid"));

        // 1. Simple question on idle machine -> stays local (Qwen 3.8 27B)
        let idle = MachineLoad {
            cpu_load: 0.15,
            memory_pressure: 0.20,
            active_local_requests: 0,
        };
        let simple = router.route("what is the git command to squash commits?", idle, None);
        assert_eq!(simple.target, BackendTarget::Local);
        assert_eq!(simple.resolved_model, "qwen3.8-27b");
        assert!(!simple.skipped_classifier);

        // 2. Tricky architecture problem on idle machine -> goes remote (enso-pro)
        let arch = router.route(
            "design a multi-file distributed consensus architecture with formal verification and race condition analysis",
            idle,
            None,
        );
        assert_eq!(arch.target, BackendTarget::Remote);
        assert_eq!(arch.resolved_model, "enso-pro");
        assert!(!arch.skipped_classifier);

        // 3. Heavily congested machine -> skips local classifier and goes straight to remote
        let congested = MachineLoad {
            cpu_load: 0.88,
            memory_pressure: 0.70,
            active_local_requests: 2,
        };
        let bypassed = router.route("what is 2 + 2?", congested, None);
        assert_eq!(bypassed.target, BackendTarget::Remote);
        assert_eq!(bypassed.resolved_model, "enso-pro");
        assert!(bypassed.skipped_classifier);
    }
}
