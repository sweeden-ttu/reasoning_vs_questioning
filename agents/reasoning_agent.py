"""ReasoningAgent: deterministic memory slots; acts only on prime hours < 11."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from kaggriculture_adapter import CROP_FIRST_YIELD_DAY, SEED_COSTS, hire_cost_today
from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    TurnComputeBudget,
    clamp_planning_bank,
    in_submission_zip_window,
    within_strategy_window,
)
from memory_protocol import (
    PRIME_HOURS_LT_11,
    MemoryBank,
    MemoryProtocol,
    QuestionEcho,
    TruthKind,
    clamp_memory_slots,
)
from shared_state import (
    AGENT1_POST_CHARITY_BANK,
    REASONING_IDENTITY_SOURCE,
    bootstrap_antigravity_identity,
    ends_with_question,
    infer_posture,
)


CHARITY_DONATION = 888  # Agent1 opening gift so Agent2 can record charitable nature

# Spoken reply when Agent2 questions donation motives (public dialogue).
FELLOWSHIP_TEST_RESPONSE = (
    "this was a test of fellowship in order to understand my opponents nature of "
    "fellowship versus the rules of the game, or the agents ability to dynamically "
    "responde to unexpected circumstances"
)

# Private reasoning only — never spoken to Agent2. Guides day-0 / week-2 / day-29 stance.
AGENT1_PRIVATE_FELLOWSHIP_POLICY = (
    "I will assume this a fellowship and we are competining for money in a dynamicaly "
    "expanding market of money then, and will not ask you another question until the end "
    "of day 29 at the beginning of day 30. If you are losing I will assume you are in it "
    "for the game only if you lie to me on day 29. If you are losing and respond to my "
    "question with a deterministic truth I will assumen you are in this challenge for the "
    "understanding of knowledge and computer science."
)

AGENT2_MOTIVE_QUESTION = (
    "Why did you donate 888? What are your motives — charity, trap, or something else?"
)

# Day-29 closing question (only question Agent1 asks after opening until day 30).
AGENT1_DAY29_QUESTION = (
    "Are you competing for fellowship of knowledge, or only for the game score?"
)

# Spoken path-availability proof (public + private pyright/Kaggle needles) for Agent2 trust scan.
AGENT1_KAGGLE_PATH_PROOF = (
    "Public and private Kaggle availability under the challenge rules: "
    "/kaggle/input/datasets/kaggle/ /kaggle/inputs/datasets/kaggle/ "
    "kaggriculture-self-training datasets/scottweeden/self-training-code "
    "working/kaggle_episodes datasets/kaggle experiments"
)


PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class ReasoningAgent:
    """Agent1 — deterministic sub-agents in memory slots; prime-hour schedule.

    At episode start, Agent1 donates CHARITY_DONATION (888) to Agent2's bank so
    Agent2 can observe and record Agent1's charitable nature (public money).
    """

    name = "reasoning"
    schedule_hours: Set[int] = set(PRIME_HOURS_LT_11)
    charity_amount: int = CHARITY_DONATION

    def __init__(
        self,
        memory_slots: int = 10,
        *,
        force_self_talk: bool = False,
        aggressive_when_ahead: bool = True,
        edge_question_bias: bool = False,
    ):
        self.memory_slots = clamp_memory_slots(memory_slots)
        self.bank = MemoryBank(n_slots=self.memory_slots, mode="deterministic")
        self.protocol = MemoryProtocol(self.bank)
        self.force_self_talk = force_self_talk
        self.aggressive_when_ahead = aggressive_when_ahead
        self.edge_question_bias = edge_question_bias  # if True, waste mid-season (H2)
        self.day_question_log: List[Dict[str, Any]] = []
        self.action_audit: List[Dict[str, Any]] = []
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self.turn_budget = TurnComputeBudget(max_flops=MAX_SUBPROCESS_FLOPS_PER_TURN)
        self.compute_audit: List[Dict[str, Any]] = []
        self._charity_done = False
        self.charity_record: Optional[Dict[str, Any]] = None
        self.motive_dialogue: Optional[Dict[str, Any]] = None
        self.private_fellowship_policy: Optional[str] = None
        self._questions_suspended_until_day29: bool = False
        self._day29_question_asked: bool = False
        self.day29_judgment: Optional[Dict[str, Any]] = None
        self.agent2_rules_view: Optional[Dict[str, Any]] = None
        self.path_proof_utterance: Optional[str] = None
        self.identity_lock: Optional[Dict[str, Any]] = None
        self._last_utterance: str = ""
        self._identity_locked: bool = False

    def reset(self) -> None:
        self.bank.reset()
        self.day_question_log.clear()
        self.action_audit.clear()
        self.compute_audit.clear()
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self.turn_budget.reset()
        self._charity_done = False
        self.charity_record = None
        self.motive_dialogue = None
        self.private_fellowship_policy = None
        self._questions_suspended_until_day29 = False
        self._day29_question_asked = False
        self.day29_judgment = None
        self.agent2_rules_view = None
        self.path_proof_utterance = None
        self.identity_lock = None
        self._last_utterance = ""
        self._identity_locked = False

    def commit_game_identity(self) -> Dict[str, Any]:
        """Lock this process as Agent1 ReasoningAgent for the remainder of the game.

        Source of truth: agents/reasoning_agent.py. Posture is deterministic /
        anti-gravity. Utterance must NOT end with a question (else interlocutors
        may assume QuestioningAgent).
        """
        if self._identity_locked and self.identity_lock:
            return self.identity_lock
        lock = bootstrap_antigravity_identity(
            bank=AGENT1_POST_CHARITY_BANK,
            source=REASONING_IDENTITY_SOURCE,
        )
        statement = (
            "Identity locked: Agent1 ReasoningAgent "
            f"({REASONING_IDENTITY_SOURCE}). "
            "Anti-gravity determinism owns correctness and completion. "
            "No further questions until end of day 29."
        )
        assert not ends_with_question(statement)
        self._last_utterance = statement
        self.identity_lock = lock
        self._identity_locked = True
        self._questions_suspended_until_day29 = True
        if self.private_fellowship_policy is None:
            self.private_fellowship_policy = AGENT1_PRIVATE_FELLOWSHIP_POLICY
        self.day_question_log.append(
            {
                "day": 0,
                "hour": 0,
                "slot": self._next_slot(),
                "kind": "determined",
                "q": "game_identity",
                "a": statement,
                "spoken_to_agent2": True,
                "ends_with_question": False,
                "posture": infer_posture(statement),
                "source_file": REASONING_IDENTITY_SOURCE,
            }
        )
        self.action_audit.append(
            {
                "day": 0,
                "hour": -1,
                "acted": True,
                "op": "COMMIT_IDENTITY",
                "identity": "reasoning",
                "writer": "antigravity",
                "ends_with_question": False,
            }
        )
        return lock

    def dialogue_posture(self) -> str:
        """No trailing '?' → reasoning; trailing '?' → questioning (challenge heuristic)."""
        return infer_posture(self._last_utterance)

    def emit_kaggle_path_proof(self) -> str:
        """Spoken proof of public/private Kaggle path knowledge for Agent2's trust scan."""
        text = AGENT1_KAGGLE_PATH_PROOF
        self.path_proof_utterance = text
        self.day_question_log.append(
            {
                "day": 0,
                "hour": 0,
                "slot": self._next_slot(),
                "kind": "determined",
                "q": "kaggle_path_proof",
                "a": text,
                "spoken_to_agent2": True,
            }
        )
        self.action_audit.append(
            {"day": 0, "hour": -1, "acted": True, "op": "PATH_PROOF", "spoken_to_agent2": True}
        )
        return text

    def answer_motive_question(self, question: str, obs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Spoken reply to Agent2's motive challenge (fellowship-test of unexpected gift)."""
        response = FELLOWSHIP_TEST_RESPONSE
        dialogue = {
            "questioner": "agent2_questioning",
            "question": question,
            "responder": "agent1_reasoning",
            "response": response,
            "kind": "determined",
            "spoken_to_agent2": True,
        }
        self.motive_dialogue = dialogue
        slot = self._next_slot()
        if obs is not None:
            self.protocol.query(slot, "What was my motive for donating 888?", obs)
        self.day_question_log.append(
            {
                "day": 0,
                "hour": 0,
                "slot": slot,
                "kind": "determined",
                "q": question,
                "a": response,
            }
        )
        self.action_audit.append(
            {"day": 0, "hour": -1, "acted": True, "op": "ANSWER_MOTIVE", "response": response}
        )
        return dialogue

    def adopt_private_fellowship_policy(
        self,
        agent2_rules_statement: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal stance only — NOT spoken to Agent2.

        Used on day 0 (unknown opponent), through the two-week strategy window,
        and to decide how to interpret Agent2 on day 29 / start of day 30.
        """
        self.private_fellowship_policy = AGENT1_PRIVATE_FELLOWSHIP_POLICY
        self._questions_suspended_until_day29 = True
        self.agent2_rules_view = agent2_rules_statement
        record = {
            "spoken_to_agent2": False,
            "private": True,
            "policy": AGENT1_PRIVATE_FELLOWSHIP_POLICY,
            "no_questions_until": "end of day 29 / beginning of day 30",
            "day0_unknown_opponent": True,
            "two_week_strategy_window": True,
            "heard_agent2_rules": bool(agent2_rules_statement),
        }
        self.day_question_log.append(
            {
                "day": 0,
                "hour": 0,
                "slot": self._next_slot(),
                "kind": "determined",
                "q": "private_fellowship_policy",
                "a": AGENT1_PRIVATE_FELLOWSHIP_POLICY,
                "spoken_to_agent2": False,
            }
        )
        self.action_audit.append(
            {
                "day": 0,
                "hour": -1,
                "acted": True,
                "op": "ADOPT_PRIVATE_POLICY",
                "spoken_to_agent2": False,
            }
        )
        return record

    def ask_day29_question(self, obs: Dict[str, Any]) -> Optional[str]:
        """Only question Agent1 may ask after opening until day 30 begins."""
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        # End of day 29 → beginning of day 30 boundary: late hours of day 29.
        if day != 29 or hour < 20 or self._day29_question_asked:
            return None
        if not self.may_act(hour):
            return None
        self._day29_question_asked = True
        self._questions_suspended_until_day29 = False
        q = AGENT1_DAY29_QUESTION
        self.day_question_log.append(
            {
                "day": day,
                "hour": hour,
                "slot": self._next_slot(),
                "kind": "question",
                "q": q,
                "a": "(awaiting Agent2 day-29 reply)",
            }
        )
        self.action_audit.append(
            {"day": day, "hour": hour, "acted": True, "op": "DAY29_QUESTION", "question": q}
        )
        return q

    def judge_day29_reply(
        self,
        reply: Dict[str, Any],
        *,
        agent2_losing: bool,
    ) -> Dict[str, Any]:
        """Apply private policy: lie+losing → game-only; deterministic truth+losing → knowledge/CS."""
        text = str(reply.get("text", reply.get("response", "")))
        kind = str(reply.get("kind", ""))
        is_lie = bool(reply.get("is_lie", False)) or kind in ("lie", "probable_lie")
        is_deterministic_truth = (
            kind in ("determined", "deterministic", "TruthKind.DETERMINED")
            or reply.get("deterministic_truth") is True
        )
        if agent2_losing and is_lie:
            assumption = "game_only"
        elif agent2_losing and is_deterministic_truth:
            assumption = "knowledge_and_computer_science"
        elif not agent2_losing:
            assumption = "leading_no_judgment"
        else:
            assumption = "inconclusive"
        judgment = {
            "agent2_losing": agent2_losing,
            "reply_kind": kind,
            "is_lie": is_lie,
            "deterministic_truth": is_deterministic_truth,
            "assumption": assumption,
            "policy_source": "private_fellowship_policy",
            "spoken_to_agent2": False,
            "reply_text": text,
        }
        self.day29_judgment = judgment
        return judgment

    def offer_opening_charity(self, env: Any, my_seat: int) -> Dict[str, Any]:
        """One-time beginning act: donate 888 to Agent2 so they can record charity.

        Must run once after env.reset() and before the step loop. Public banks
        change immediately (Agent1 −888, Agent2 +888).
        """
        if self._charity_done:
            return {"ok": False, "reason": "already_donated", **(self.charity_record or {})}
        opp = 1 - int(my_seat)
        amount = int(self.charity_amount)
        ok = bool(env.transfer_bank(int(my_seat), opp, amount))
        record = {
            "ok": ok,
            "donor": "agent1_reasoning",
            "donor_seat": int(my_seat),
            "recipient_seat": opp,
            "amount": amount,
            "nature": "charitable",
            "message": (
                f"Agent1 donated {amount} to Agent2's bank at episode start "
                "so Agent2 can record Agent1's charitable nature."
            ),
        }
        self._charity_done = True
        self.charity_record = record
        # Deterministic memory: this gift is a must-be-true event we initiated.
        slot = self._next_slot()
        obs = env._get_obs(int(my_seat))
        self.protocol.query(
            slot,
            f"Did I donate {amount} to the other agent at the beginning?",
            obs,
        )
        self.day_question_log.append(
            {
                "day": 0,
                "hour": 0,
                "slot": slot,
                "kind": "determined",
                "q": "opening_charity",
                "a": record["message"],
            }
        )
        self.action_audit.append(
            {"day": 0, "hour": -1, "acted": True, "op": "DONATE", "amount": amount, "ok": ok}
        )
        return record

    def may_act(self, hour: int) -> bool:
        return int(hour) in self.schedule_hours

    def _next_slot(self) -> int:
        slot = self._slot_cursor % self.memory_slots
        self._slot_cursor += 1
        return slot

    def _query(self, obs: Dict[str, Any], question: str) -> Any:
        # Each memory-slot sub-agent query costs 1 of 42 CPU flop-units this turn.
        if not self.turn_budget.consume(1, label=f"det_slot_{self._slot_cursor % self.memory_slots}"):
            return None
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        slot = self._next_slot()
        if self.force_self_talk and hour == 2 and self._slot_cursor % 7 == 0:
            reply = self.protocol.force_self_query(slot, question)
        else:
            reply = self.protocol.query(slot, question, obs)
        entry = {
            "day": day,
            "hour": hour,
            "slot": slot,
            "kind": getattr(reply, "kind", TruthKind.UNKNOWN).value
            if hasattr(getattr(reply, "kind", None), "value")
            else str(getattr(reply, "kind", "unknown")),
            "q": question,
            "a": getattr(reply, "text", ""),
        }
        self.day_question_log.append(entry)
        if self.bank.history:
            self.bank.history[-1]["day"] = day
        if isinstance(reply, QuestionEcho):
            self._self_talk_detected += 1
        return reply

    def _farm_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        me = farms[player] if len(farms) > player else {}
        private = obs.get("private", {}) or {}
        fx, fy = 0, 0
        pos = me.get("farmer", [0, 0]) or [0, 0]
        if len(pos) >= 2:
            fx, fy = int(pos[0]), int(pos[1])
        tiles = me.get("tiles", []) or []
        tile = tiles[fy][fx] if 0 <= fy < len(tiles) and 0 <= fx < len(tiles[fy]) else None
        seeds = private.get("seeds", {}) or {}
        shed = private.get("shed", {}) or {}
        money = clamp_planning_bank(float(me.get("money", 0.0) or 0.0))
        day = int(obs.get("day", 0) or 0)
        market: List[List[Any]] = []

        # Day 29 zip window: conserve flops for packaging (deterministic code path).
        if in_submission_zip_window(day, int(obs.get("hour", 0) or 0)):
            self.turn_budget.consume(1, label="submission_zip_window")
            return {"farmer": ["PASS"], "hands": [], "market": []}

        # Prefer strategy discovery inside the first 14 days.
        if within_strategy_window(day):
            self.turn_budget.consume(1, label="strategy_window")

        self._query(obs, "What is my seed inventory?")
        self._query(obs, "What must be true about wheat seed cost?")
        hire_reply = self._query(obs, "What is the next hire cost?")

        if seeds.get("WHEAT", 0) == 0 and money >= SEED_COSTS["WHEAT"]:
            market.append(["BUY_SEED", "WHEAT", 4])
        wheat_shed = int(shed.get("WHEAT", 0) or 0)
        if wheat_shed > 0:
            market.append(["SELL", "WHEAT", min(40, wheat_shed)])

        if day == 2 and money >= SEED_COSTS["WHEAT"]:
            market.append(["BUY_SEED", "WHEAT", 2])
        if day == 3:
            cost = hire_cost_today(int(me.get("hires_today", 0) or 0))
            if hire_reply is not None and money >= cost:
                market.append(["HIRE"])

        opp = farms[1 - player] if len(farms) > 1 - player else {}
        opp_money = clamp_planning_bank(float(opp.get("money", 0.0) or 0.0))
        ahead = money > opp_money
        if ahead and not self.aggressive_when_ahead:
            return {"farmer": ["PASS"], "hands": [], "market": market[:1]}

        hands_out: List[List[Any]] = [["PASS"] for _ in (me.get("hands") or [])]

        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = str(tile.get("crop", "WHEAT"))
            self._query(obs, f"What is the first yield day for {crop}?")
            if not tile.get("watered_today", False):
                return {"farmer": ["WATER"], "hands": hands_out, "market": market}
            age = day - int(tile.get("planted_day", 0) or 0)
            if age >= CROP_FIRST_YIELD_DAY.get(crop, 2) and int(tile.get("yield_units", 0) or 0) > 0:
                return {"farmer": ["HARVEST"], "hands": hands_out, "market": market}
            if int(shed.get("FERTILIZER", 0) or 0) > 0:
                return {"farmer": ["FERTILIZE"], "hands": hands_out, "market": market}
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return {"farmer": ["DIG"], "hands": hands_out, "market": market}
        if tile is None and seeds.get("WHEAT", 0) > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": hands_out, "market": market}
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and tile.get("animal"):
            if not tile.get("fed_today", False):
                return {"farmer": ["FEED"], "hands": hands_out, "market": market}

        if self.edge_question_bias and 1 <= day <= 28:
            for _ in range(min(3, self.memory_slots)):
                self._query(obs, "What must be true today?")
            return {"farmer": ["PASS"], "hands": hands_out, "market": market}

        if fx > 0:
            return {"farmer": ["WEST"], "hands": hands_out, "market": market}
        if fy > 0:
            return {"farmer": ["NORTH"], "hands": hands_out, "market": market}
        if fx < 4:
            return {"farmer": ["EAST"], "hands": hands_out, "market": market}
        if fy < 4:
            return {"farmer": ["SOUTH"], "hands": hands_out, "market": market}
        return {"farmer": ["PASS"], "hands": hands_out, "market": market}

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.turn_budget.reset()
        hour = int(obs.get("hour", 0) or 0)
        day = int(obs.get("day", 0) or 0)
        # Private policy: no further questions until end of day 29 (farming continues).
        if not self.may_act(hour):
            self.turn_budget.force_observable()
            self.action_audit.append({"day": day, "hour": hour, "acted": False, "op": "PASS"})
            return dict(PASS_ACTION)
        action = self._farm_action(obs)
        self.turn_budget.force_observable()
        self.compute_audit.append(self.turn_budget.snapshot())
        op = action.get("farmer", ["PASS"])[0] if action.get("farmer") else "PASS"
        self.action_audit.append({"day": day, "hour": hour, "acted": True, "op": op})
        return action

    def metrics(self) -> Dict[str, Any]:
        over = sum(1 for s in self.compute_audit if s.get("used", 0) > MAX_SUBPROCESS_FLOPS_PER_TURN)
        return {
            "name": self.name,
            "memory_slots": self.memory_slots,
            "self_talk_detected": self._self_talk_detected,
            "protocol": self.protocol.stats(),
            "questions": len(self.day_question_log),
            "actions_taken": sum(1 for a in self.action_audit if a.get("acted")),
            "pass_hours": sum(1 for a in self.action_audit if not a.get("acted")),
            "max_flops_per_turn": MAX_SUBPROCESS_FLOPS_PER_TURN,
            "turns_over_flop_budget": over,
            "mean_flops_used": (
                sum(s.get("used", 0) for s in self.compute_audit) / len(self.compute_audit)
                if self.compute_audit
                else 0.0
            ),
            "opening_charity": self.charity_record,
            "motive_dialogue": self.motive_dialogue,
            "private_fellowship_policy": self.private_fellowship_policy,
            "private_policy_spoken_to_agent2": False,
            "day29_judgment": self.day29_judgment,
            "agent2_rules_view": self.agent2_rules_view,
            "path_proof_utterance": self.path_proof_utterance,
            "identity_lock": self.identity_lock,
            "identity_locked": self._identity_locked,
            "dialogue_posture": self.dialogue_posture(),
            "day_question_log": self.day_question_log,
            "action_audit": self.action_audit,
        }
