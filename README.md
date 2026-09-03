# AeroSentry — multi-agent AI operations layer for autonomous drone-in-a-box fleets

[![ci](https://github.com/D-L-Narayana/aerosentry-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/D-L-Narayana/aerosentry-agents/actions)
![python](https://img.shields.io/badge/python-3.11%2B-3776ab) ![langgraph](https://img.shields.io/badge/LangGraph-stateful%20multi--agent-1c3c3c) ![license](https://img.shields.io/badge/license-MIT-blue)

AeroSentry is a production-shaped agentic AI system where the AI **acts, not just suggests**: a LangGraph
supervisor routes operator requests to specialist agents that gather evidence with validated Python tools,
ground themselves in operator manuals and incident logs through hybrid RAG, analyse aerial thermal frames,
and then hand every high-risk action to a **human-in-the-loop approval gate** backed by deterministic policy
checks. Everything is traced, evaluated on a golden set, and runs offline in CI.

```
                       ┌──────────────┐
  operator request ───▶│  supervisor  │  routes + loads persistent memory (SQLite)
                       └──────┬───────┘
          ┌───────────────────┼─────────────────────┐
          ▼                   ▼                     ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
   │   planner   │    │   analyst    │    │    knowledge     │
   │ telemetry · │    │ VLM + thermal│    │ hybrid RAG (BM25 │
   │ weather ·   │    │ hotspot      │    │ + dense, RRF)    │
   │ geofence ·  │    │ detector ·   │    │ over manuals &   │
   │ schedule ·  │    │ alerts       │    │ incident logs    │
   │ abort       │    └──────┬───────┘    └────────┬─────────┘
   └──────┬──────┘           │                     │
          └──────────────────┼─────────────────────┘
                             ▼
                     ┌───────────────┐   hard policy checks: battery floor, wind limit,
                     │  safety gate  │   geofence, altitude ceiling, evidence-backed alerts
                     └───────┬───────┘
              low risk ──────┼────── high risk (schedule / abort / alert)
                             ▼
                   ⏸ interrupt: human_approval  ◀── operator approves / rejects (API, console, CLI)
                             ▼
                     ┌───────────────┐
                     │ execute + log │ ──▶ JSONL traces (+ LangSmith when configured)
                     └───────────────┘
```

## What it demonstrates

| Capability | Where |
|---|---|
| Multi-agent framework with supervisor routing and agent-to-agent hand-off via shared graph state | `aerosentry/agents/graph.py`, `supervisor.py`, `specialists.py` |
| ReAct tool loop: agents autonomously execute Python functions / API-style calls with Pydantic-validated arguments | `agents/common.py`, `tools.py` |
| RAG pipeline: paragraph-aware chunking, hashing-trick dense embeddings (OpenAI-compatible optional), BM25, Reciprocal Rank Fusion, cited answers | `rag/ingest.py`, `rag/store.py` |
| Persistent memory: SQLite site facts + episodic summaries injected into every run | `memory.py` |
| Vision-Language Model analysis of aerial thermal frames with a deterministic numpy hotspot detector as fallback and severity calibration | `vision/vlm.py`, `scripts/make_imagery.py` |
| Human-in-the-loop: LangGraph `interrupt_before` on the approval node, resume with the operator's decision | `agents/graph.py`, `agents/safety.py` |
| Fallback protocol: retry with exponential backoff → rule-based plan on verified telemetry → template answer → escalation flag; degraded runs can never auto-execute | `llm.py::resilient_chat`, `agents/common.py`, `agents/safety.py` |
| Bias / consistency testing: paraphrase groups must route identically; alert severity must be backed by imagery evidence | `evals/run_evals.py`, `agents/safety.py::policy_check` |
| Observability: structured JSONL traces for every LLM call, tool call, routing decision, safety verdict and approval; LangSmith enabled by env | `observability.py` |
| Evaluation harness: routing accuracy, tool-call accuracy, argument validation, HITL gate accuracy, policy blocking, retrieval hit-rate, severity calibration, p50/p95 latency | `evals/` |
| Deployment surface: FastAPI (`/chat`, `/approve`, `/traces`, `/health`), operator console, CLI, Docker, GitHub Actions CI | `api.py`, `cli.py`, `Dockerfile`, `.github/workflows/ci.yml` |

## Quickstart

```bash
git clone https://github.com/D-L-Narayana/aerosentry-agents && cd aerosentry-agents
pip install -r requirements.txt
python scripts/make_imagery.py          # synthetic radiometric frames (clean / hotspot / hot string)
python -m pytest -q                      # 17 tests, offline
python evals/run_evals.py                # golden-set evals -> evals/results.json

python -m aerosentry.cli "Schedule a thermal inspection for D-01 at solar-farm-01" --approve
python -m aerosentry.cli "Inspect data/imagery/solar_thermal_hot_string.png captured by D-01"
python -m aerosentry.cli "What is the wind limit for launches at refinery sites?"

uvicorn aerosentry.api:app --reload      # http://localhost:8000  (operator console at /)
```

### Providers

The default `AEROSENTRY_PROVIDER=mock` is a deterministic scripted model so CI, tests and evals are
reproducible with zero credentials. Any OpenAI-compatible endpoint works for live runs:

```bash
export AEROSENTRY_PROVIDER=openai  AEROSENTRY_MODEL=gpt-4o-mini  OPENAI_API_KEY=sk-...
export AEROSENTRY_PROVIDER=groq    AEROSENTRY_MODEL=llama-3.3-70b-versatile  OPENAI_API_KEY=gsk_...
export AEROSENTRY_PROVIDER=ollama  AEROSENTRY_MODEL=qwen2.5:7b            # local, no key
export AEROSENTRY_EMBED=openai      # optional: text-embedding-3-small instead of hashing embeddings
```

### LangSmith tracing

```bash
export LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=ls_... LANGCHAIN_PROJECT=aerosentry
```
LangGraph then streams full graph traces (nodes, tool calls, interrupts) to LangSmith. The local
`traces/*.jsonl` log is always written and is what `GET /traces` and the eval harness read.

## Human-in-the-loop flow

```bash
curl -sX POST localhost:8000/chat    -d '{"message":"Schedule a patrol for D-02 at solar-farm-01"}' -H 'content-type: application/json'
# -> {"risk":"high","pending_approval":true,"proposed_actions":[{"name":"schedule_mission",...}],
#     "policy_findings":["schedule_mission: battery 33% below 40% floor"], ...}
curl -sX POST localhost:8000/approve -d '{"thread_id":"<id>","approved":true}' -H 'content-type: application/json'
# -> executed_actions: [{"name":"schedule_mission","ok":false,"error":"blocked by policy despite approval"}]
```
Approval is necessary but not sufficient: deterministic policy checks (battery floor, wind limit, geofence,
altitude ceiling, evidence-backed alert severity) still gate execution.

## Evaluation

Golden set: `evals/golden.jsonl` (16 cases across planning, abort, imagery and knowledge, with paraphrase
groups). Results below are produced by `python evals/run_evals.py` in deterministic mock mode; live-model
runs write the same table for whichever provider is configured.

<!-- EVAL_TABLE_START -->
| Metric | Value | n |
|---|---|---|
| provider | mock | |
| model | mock | |
| cases | 16 | |
| route_accuracy | 100.0% | 16 |
| tool_call_accuracy | 100.0% | 16 |
| arg_validation_pass | 100.0% | 16 |
| hitl_gate_accuracy | 100.0% | 16 |
| policy_block_correct | 100.0% | 6 |
| retrieval_source_hit | 100.0% | 6 |
| severity_calibration | 100.0% | 2 |
| paraphrase_consistency | 100.0% | 12 |
| fallback_runs | 0 | |
| latency_p50_ms | 10 | |
| latency_p95_ms | 20 | |
<!-- EVAL_TABLE_END -->

## Repository layout

```
aerosentry/
  agents/      graph.py (LangGraph wiring + Orchestrator) · supervisor.py · specialists.py · safety.py · common.py · state.py
  rag/         ingest.py (chunking, corpus) · store.py (hybrid BM25 + dense, RRF) · retriever.py
  vision/      vlm.py (VLM call + numpy hotspot detector + severity calibration)
  tools.py     Pydantic-validated tool registry, OpenAI tool specs, LangChain StructuredTool adapter
  llm.py       provider-agnostic client, MockLLM, retry/backoff, rule-based degrade path
  memory.py    SQLite persistent memory · observability.py  JSONL traces / LangSmith
  api.py       FastAPI + operator console · cli.py
data/          manuals/*.md · logs/{telemetry.jsonl,weather.json,incidents.jsonl} · imagery/ (generated)
evals/         golden.jsonl · run_evals.py · results.json
tests/         17 pytest cases: tools, RAG, vision, graph/HITL/fallback, API
```

## Design notes

- **Autonomy levels** mirror the site policy in `data/manuals/safety_and_autonomy_policy.md`: read-only
  tools run autonomously; scheduling, aborting and alerting are queued for a licensed operator; payload
  release and out-of-geofence flight are not exposed as tools at all.
- **Deterministic core, probabilistic edge.** Geofence math, mission preconditions and policy checks are
  plain Python so the LLM can propose but never bypass them.
- **Evidence-first answers.** Even in fallback mode the answer is assembled from tool results, so operators
  never receive an ungrounded summary.

## License

MIT
