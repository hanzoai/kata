# Hanzo Kata

Hanzo Kata is the autonomous coding agent and self-healing CI/CD triage loop:
1. Ingest compiler errors, lint warnings, or test failures from terminal output or CI logs.
2. Search repository AST and dependency graphs for the root cause.
3. Generate minimal, auditable unified diff patches using Zen5-Coder or Enso.
4. Run validation checks (`cargo check`, `npm test`, `pytest`) and gate automated commits vs. human escalation.

- **Cloud Endpoint:** `POST https://api.hanzo.ai/v1/chat/completions`
- **Models:** `zen5-coder`, `zen`, `enso-pro`
- **Docs:** [docs.hanzo.ai](https://docs.hanzo.ai)

---

## Authentication & API Key

Set `HANZO_API_KEY`:

```bash
# Option 1: Automatic login via Hanzo CLI
curl -fsSL https://hanzo.sh | sh
hanzo auth login
export HANZO_API_KEY="$(hanzo auth token)"

# Option 2: Generate via Web Console
# Visit https://platform.hanzo.ai/keys and create an API key
export HANZO_API_KEY="sk_live_..."
```

---

## Runnable Integration Examples

Runnable client examples across languages in [`examples/`](examples/):

- **Python**: [`examples/python/coding_agent.py`](examples/python/coding_agent.py) (Autonomous repair loop)
- **TypeScript**: [`examples/typescript/coding_agent.ts`](examples/typescript/coding_agent.ts) (Native fetch)
- **Go**: [`examples/go/main.go`](examples/go/main.go) (Standard `net/http`)
- **Rust**: [`examples/rust/`](examples/rust/) (Type-safe client with `reqwest`)
- **cURL**: [`examples/curl/agent.sh`](examples/curl/agent.sh) (One-liner script)

```bash
# Quick run (Python)
python3 examples/python/coding_agent.py

# Quick run (cURL)
./examples/curl/agent.sh
```

---

## Production Rust Recipes

[`recipes/`](recipes/): Modular state machine and repair loop for Rust services:
- **`KataAgentLoopRecipe`**: Self-healing test-and-repair state machine that ingests failing compiler/test output and generates unified diffs.
- **`TriageAction`**: Evaluates test exit codes (`ApplyPatch`, `ReRunTests`, `EscalateToHuman`).

```rust
use kata_recipes::{CodingTriageState, KataAgentLoopRecipe};

let state = CodingTriageState {
    repo_name: "myorg/core".into(),
    branch: "fix/leak".into(),
    failing_step: "cargo test".into(),
    stdout_stderr: "error[E0308]: mismatched types".into(),
    files_modified: vec!["src/lib.rs".into()],
    max_attempts: 3,
    current_attempt: 1,
};

let prompt = KataAgentLoopRecipe::build_patch_prompt(&state);
```
