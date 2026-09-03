from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    thread_id: str
    run_id: str
    route: str
    memory: list[str]
    messages: list[dict]
    tool_results: list[dict]        # executed tool calls: {name, arguments, ok, valid, result}
    proposed_actions: list[dict]    # high-risk tool calls awaiting human approval
    requires_approval: bool
    approved: Optional[bool]
    risk: str
    policy_findings: list[str]
    answer: str
    citations: list[dict]
    fallback_used: bool
    executed_actions: list[dict]
    meta: dict[str, Any]
