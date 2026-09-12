"""Agent package for Reasoning vs Questioning experiments."""

from .questioning_agent import QuestioningAgent
from .rag_observer_reasoning_agent import (
    RAGObserverReasoningAgent,
    build_opponent_kmap_mask,
)
from .reasoning_agent import ReasoningAgent
from .scott_weeden_agent import ScottWeedenAgent, claim_after_agent1_write
from .ten_agents import (
    TenAgentQueue,
    build_all_10_subagents,
)
from .tripartite_composite_agent import (
    TripartiteDecisionContext,
    TripartiteDecisionEngine,
    TripartiteReasoningAgent,
)
from .qkd_replay_rl_agent import QKDReplayRLAgent
from .qkd_gbdt_agent import QKDGBDTAgent
from .multi_phase_imitation_opponent import (
    MultiPhaseImitationOpponent,
    MultiPhaseImitationOpponentAgent,
    get_multi_phase_opponent,
    multi_phase_imitation_agent,
)

__all__ = [
    "ReasoningAgent",
    "RAGObserverReasoningAgent",
    "build_opponent_kmap_mask",
    "TripartiteReasoningAgent",
    "TripartiteDecisionEngine",
    "TripartiteDecisionContext",
    "QuestioningAgent",
    "ScottWeedenAgent",
    "TenAgentQueue",
    "build_all_10_subagents",
    "claim_after_agent1_write",
    "QKDReplayRLAgent",
    "QKDGBDTAgent",
    "MultiPhaseImitationOpponent",
    "MultiPhaseImitationOpponentAgent",
    "get_multi_phase_opponent",
    "multi_phase_imitation_agent",
]
