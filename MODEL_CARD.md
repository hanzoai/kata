---
language:
- en
- code
license: proprietary
library_name: hanzo
pipeline_tag: reinforcement-learning
tags:
- autonomous-agent
- self-healing-ci
- code-generation
- agentic-ai
- api-only
metrics:
- swe_bench_verified
- devbench_pass_rate
- repair_iterations
model-index:
- name: hanzoai/kata
  results:
  - task:
      type: software-engineering
      name: SWE-bench Verified
    metrics:
    - type: pass@1
      value: 0.492
      name: Issue Resolution Rate
  - task:
      type: autonomous-execution
      name: DevBench
    metrics:
    - type: pass@1
      value: 0.724
      name: Full-Stack Task Pass Rate
---

# Hanzo Kata: Autonomous Agent & Self-Healing Execution System Card

> ⚠️ **Platform Runtime · Closed Source**  
> Hanzo Kata is the autonomous task DAG and self-healing software engineering agent runtime.

---

## 1. System Overview

Kata is an autonomous agent framework built on Hanzo's dual decision-generative plane:
- **Decision Engine (Kai)**: Evaluates preflight safety, tool selection, risk triage, and termination checks at every step ($0.021/M tokens).
- **Reasoning Engine (Zen / Enso)**: Synthesizes minimal unified diff patches and generates code solutions.
- **DAG Execution Loop**: Ingests compiler/test feedback, iterates over repair candidates, and halts on deterministic pass criteria.

---

## 2. Verified Autonomous Agent Benchmarks

Evaluated on industry-standard benchmark suites:

| Benchmark | Task Focus | Kata Agent Pass@1 | Human Engineer Baseline | Standard LLM (Zero-shot) |
|---|---|---|---|---|
| **SWE-bench Verified** | Real GitHub Bug Fixes | **49.2%** | 62.0% | 19.5% |
| **DevBench** | Multi-file Software Projects | **72.4%** | 81.0% | 34.0% |
| **Self-Healing CI Loop** | Automatic Compiler Bug Fix | **84.6%** (< 3 turns) | — | 41.2% |

---

## 3. Related Resources

- **Interactive Agent Arena**: [`huggingface.co/spaces/hanzoai/agent-benchmark`](https://huggingface.co/spaces/hanzoai/agent-benchmark)
- **Documentation**: [docs.hanzo.ai](https://docs.hanzo.ai)
