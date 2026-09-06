"""Agent package for Reasoning vs Questioning experiments."""

from .questioning_agent import QuestioningAgent
from .reasoning_agent import ReasoningAgent
from .scott_weeden_agent import ScottWeedenAgent, claim_after_agent1_write
from .ten_agents import (
    TenAgentQueue,
    build_all_10_subagents,
)

__all__ = [
    "ReasoningAgent",
    "QuestioningAgent",
    "ScottWeedenAgent",
    "TenAgentQueue",
    "build_all_10_subagents",
    "claim_after_agent1_write",
]
