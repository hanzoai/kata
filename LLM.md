# Hanzo Kata (`hanzoai/kata`)

Durable, minimal, malleable Rust agent harness and autonomous coding recipes.

## Workspace Crates

- **`kata-agent/`** (`kata-agent`): Durable multi-conversation agent harness with pluggable `Storage` (`MemoryStorage`, `JsonlStorage`, `SqliteStorage` + conformance suite), pluggable `ExecutionEnv` (`LocalExecutionEnv`, `MemoryExecutionEnv`), checkpointed tasks with `ReplayPolicy` (`Safe` vs `Interrupt`) crash recovery (`Harness::resume`), `request_id` submission idempotency, O(1) copy-on-write `fork` with 3-tier `Ownership` (`Task`, `Conversation`, `Ownerless`), `WhenBusy` (`Queue` / `Steer`), typed `DocSpec` `AsOf` snapshots, bounded working-set `compact` / `reset_with_handoff` + `search_history`, `CodingTools`, `wrap_tool` middleware, and durable `memo`.
- **`recipes/`** (`kata-recipes`): Autonomous coding triage state machines (`KataAgentLoopRecipe`) and congestion-aware `VirtualModelRouter` (`hybrid/auto`).

## Commands

```bash
cargo test --workspace
```
