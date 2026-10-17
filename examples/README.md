# Hanzo Kata SDK Integration Examples

Hanzo Kata is the autonomous coding agent and CI/CD triage loop. It operates directly in git repositories to:
1. Ingest compiler, linter, or test failures from stdout/stderr.
2. Search repository AST and dependency graphs for the root cause.
3. Generate minimal, auditable unified diff patches using Zen5-Coder or Enso.
4. Run validation checks (`cargo check`, `npm test`, `pytest`) and gate automated git commits vs. human escalation.

## Language Examples

- **Python**: [`python/coding_agent.py`](python/coding_agent.py) (Autonomous repair loop)
- **TypeScript**: [`typescript/coding_agent.ts`](typescript/coding_agent.ts) (Native `fetch`, strongly typed)
- **Go**: [`go/main.go`](go/main.go) (Standard `net/http`)
- **cURL**: [`curl/agent.sh`](curl/agent.sh) (One-liner command line call)
- **Rust**: [`rust/src/main.rs`](rust/src/main.rs) (Using `reqwest` & `serde_json`)

## Quick Run

```bash
export HANZO_API_KEY="$(hanzo auth token)"

# Python
python3 python/coding_agent.py

# cURL
./curl/agent.sh
```
