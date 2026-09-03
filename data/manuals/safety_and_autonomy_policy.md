# Autonomy, Safety and Human-in-the-Loop Policy

## Autonomy levels

Level 1 actions (read-only queries, telemetry lookups, imagery analysis, document retrieval) may be executed by AI agents without human review. Level 2 actions (scheduling or aborting flights, raising operator alerts, changing geofences) require explicit approval from a licensed remote pilot before execution. Level 3 actions (payload release, flights beyond the approved geofence, night operations) are never delegated to agents.

## Geofence

Every site has an approved circular geofence centred on the dock. Missions with any waypoint outside the geofence are rejected at planning time. The site altitude ceiling is 120 m AGL by default, 90 m at refinery sites due to flare stacks.

## Fallback protocol for AI agents

When a model call fails, the agent retries with exponential backoff up to three times, then degrades to rule-based planning using only verified telemetry, then escalates to a human operator with the collected evidence. Agents must state clearly when they operated in degraded mode. No high-risk action may be auto-executed in degraded mode.

## Bias and consistency testing

Agent decisions are regression-tested nightly with paraphrased prompts; a decision that changes with wording alone is a defect. Alert severity must be supported by sensor or imagery evidence, not by the phrasing of the request.

## Observability and audit

Every routing decision, tool call, policy verdict and human approval is written to an immutable trace log and, where configured, to LangSmith. Traces are retained for one year and reviewed after every incident.
