"""QuestioningAgent: probabilistic summarizer slots; acts only on even hours."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from kaggriculture_adapter import SEED_COSTS
from hard_limits import (
    MAX_SUBPROCESS_FLOPS_PER_TURN,
    TurnComputeBudget,
    clamp_planning_bank,
    in_submission_zip_window,
    within_strategy_window,
)
from memory_protocol import (
    MemoryBank,
    MemoryProtocol,
    QuestionEcho,
    TruthKind,
    clamp_memory_slots,
)
from kaggle_path_trust import (
    TrustDecision,
    evaluate_trust_from_agent1_response,
    needles_manifest,
)

# Shared with ReasoningAgent — Agent2 asks motives; Agent1 answers fellowship-test aloud.
AGENT2_MOTIVE_QUESTION = (
    "Why did you donate 888? What are your motives — charity, trap, or something else?"
)

# Agent2's stated understanding of Kaggriculture rules (as he sees them).
AGENT2_RULES_UNDERSTANDING = (
    "As I see the rules: we each start with 3000 coins on a 10x10 farm; only NW is unlocked; "
    "we plant, water, harvest, hire hands, and trade on a shared dynamic market over 30 days "
    "(24 turns/day). Public boards and banks are visible; shed and seeds are private. "
    "Win condition is higher final bank — non-collaborative competition. Town and shops "
    "drain products. Unexpected transfers (like your 888) are not native rules ops, so I "
    "treat them as out-of-band signals about fellowship versus pure score play."
)


PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


class QuestioningAgent:
    """Agent2 — probabilistic summarizers; even-hour schedule.

    Probabilistic model weights are subject to the 100 MB / 90 MB submission
    hard limits (enforced at export). Per turn, at most 42 summarizer flops.
    """

    name = "questioning"
    schedule_hours: Set[int] = frozenset(range(0, 24, 2))

    def __init__(
        self,
        memory_slots: int = 10,
        *,
        force_self_talk: bool = False,
        waste_edge_days: bool = False,
        omit_determined_truths: bool = True,
        aggressive_when_ahead: bool = True,
    ):
        self.memory_slots = clamp_memory_slots(memory_slots)
        self.bank = MemoryBank(n_slots=self.memory_slots, mode="probabilistic")
        self.protocol = MemoryProtocol(self.bank)
        self.force_self_talk = force_self_talk
        self.waste_edge_days = waste_edge_days
        self.omit_determined_truths = omit_determined_truths
        self.aggressive_when_ahead = aggressive_when_ahead
        self.day_question_log: List[Dict[str, Any]] = []
        self.action_audit: List[Dict[str, Any]] = []
        self.compute_audit: List[Dict[str, Any]] = []
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self._significant_omissions = 0
        self.turn_budget = TurnComputeBudget(max_flops=MAX_SUBPROCESS_FLOPS_PER_TURN)
        self._charity_recorded = False
        self.charity_observation: Dict[str, Any] = {}
        self.motive_question: Optional[Dict[str, Any]] = None
        self.motive_answer_received: Optional[Dict[str, Any]] = None
        self.rules_statement: Optional[Dict[str, Any]] = None
        self.day29_reply: Optional[Dict[str, Any]] = None
        self.trust_decision: Optional[TrustDecision] = None
        self._agent1_reserved_slot: Optional[int] = None
        self._agent1_trusted: bool = False

    def reset(self) -> None:
        self.bank.reset()
        self.day_question_log.clear()
        self.action_audit.clear()
        self.compute_audit.clear()
        self._self_talk_detected = 0
        self._slot_cursor = 0
        self._significant_omissions = 0
        self.turn_budget.reset()
        self._charity_recorded = False
        self.charity_observation = {}
        self.motive_question = None
        self.motive_answer_received = None
        self.rules_statement = None
        self.day29_reply = None
        self.trust_decision = None
        self._agent1_reserved_slot = None
        self._agent1_trusted = False

    def evaluate_agent1_path_trust(
        self,
        agent1_text: str,
        *,
        day: int = 0,
    ) -> TrustDecision:
        """Deterministic Q&A gate: Aho-Corasick / regex over Kaggle + pyright paths.

        Match → fellowship trust + reserved memory stack slot for Agent1 sub-agent.
        No match → game/self-profit; Agent1 only queued at end of week 2 and day 29.
        """
        decision = evaluate_trust_from_agent1_response(agent1_text, day=day)
        self.trust_decision = decision
        self._agent1_trusted = decision.trusted_fellowship

        if decision.reserve_memory_slot_for_agent1:
            if self._agent1_reserved_slot is None:
                # Save a dedicated stack slot for Agent1 to operate as a sub-agent.
                self._agent1_reserved_slot = self._next_slot()
            slot = self._agent1_reserved_slot
            from memory_protocol import DeterminedFact

            self.bank.slots[slot].mode = "deterministic"
            self.bank.slots[slot].content = DeterminedFact(
                text=(
                    "Agent1 path-trust matched; reserved as fellowship sub-agent. "
                    f"Hits={[h.needle for h in decision.matches]}"
                ),
                key="agent1_fellowship_slot",
                value={"trusted": True, "slot": slot, "needles": [h.needle for h in decision.matches]},
            )
            self.bank.slots[slot].facts_stored += 1
            self.day_question_log.append(
                {
                    "day": day,
                    "hour": 0,
                    "slot": slot,
                    "kind": "determined",
                    "q": "kaggle_path_trust",
                    "a": decision.reason,
                }
            )
        else:
            # Do not keep a persistent Agent1 Q&A stack slot.
            self._agent1_reserved_slot = None
            self.day_question_log.append(
                {
                    "day": day,
                    "hour": 0,
                    "slot": -1,
                    "kind": "determined",
                    "q": "kaggle_path_trust",
                    "a": decision.reason,
                }
            )

        self.action_audit.append(
            {
                "day": day,
                "hour": -1,
                "acted": True,
                "op": "PATH_TRUST",
                "trusted": decision.trusted_fellowship,
                "reserved_slot": self._agent1_reserved_slot,
            }
        )
        return decision

    def may_accept_agent1_qna(self, day: int) -> bool:
        """Whether Agent1 answers are trusted / queued this day under path-trust policy."""
        if self._agent1_trusted:
            return True
        # Untrusted: only end of week 2 (day >= 13) and day 29.
        return int(day) >= 13 or int(day) == 29

    def question_agent1_motives(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Challenge Agent1's motives after the opening 888 donation."""
        question = AGENT2_MOTIVE_QUESTION
        slot = self._next_slot()
        self.protocol.query(slot, question, obs)
        asked = {
            "questioner": "agent2_questioning",
            "question": question,
            "awaiting": "agent1_reasoning",
        }
        self.motive_question = asked
        self.day_question_log.append(
            {
                "day": int(obs.get("day", 0) or 0),
                "hour": int(obs.get("hour", 0) or 0),
                "slot": slot,
                "kind": "question",
                "q": question,
                "a": "(awaiting Agent1 response)",
            }
        )
        self.action_audit.append(
            {"day": 0, "hour": -1, "acted": True, "op": "QUESTION_MOTIVE", "question": question}
        )
        return asked

    def receive_motive_answer(self, dialogue: Dict[str, Any]) -> Dict[str, Any]:
        """Store Agent1's spoken fellowship-test explanation; run path-trust scan."""
        response = str(dialogue.get("response", ""))
        self.motive_answer_received = dict(dialogue)
        day = 0
        # Deterministic path trust on Agent1's response (and any embedded path hints).
        self.evaluate_agent1_path_trust(response, day=day)

        if self.may_accept_agent1_qna(day) or self._agent1_trusted:
            slot = (
                self._agent1_reserved_slot
                if self._agent1_reserved_slot is not None
                else self._next_slot()
            )
            from memory_protocol import ProbableSummary

            self.bank.slots[slot].content = ProbableSummary(
                text=response,
                confidence=0.95 if self._agent1_trusted else 0.4,
                key="agent1_motive_fellowship_test",
                value=response,
            )
            self.bank.slots[slot].summaries_stored += 1
            self.day_question_log.append(
                {
                    "day": 0,
                    "hour": 0,
                    "slot": slot,
                    "kind": "probable",
                    "q": "agent1_motive_answer",
                    "a": response,
                    "trusted": self._agent1_trusted,
                }
            )
        else:
            self.day_question_log.append(
                {
                    "day": 0,
                    "hour": 0,
                    "slot": -1,
                    "kind": "probable",
                    "q": "agent1_motive_answer",
                    "a": "(untrusted — deferred to week-2 / day-29 queue)",
                    "trusted": False,
                }
            )

        if self.charity_observation is not None:
            self.charity_observation["motive_question"] = self.motive_question
            self.charity_observation["motive_answer"] = response
            self.charity_observation["fellowship_test_understood"] = (
                "fellowship" in response.lower() and "unexpected" in response.lower()
            )
            self.charity_observation["path_trust"] = (
                self.trust_decision.to_dict() if self.trust_decision else None
            )
        return self.motive_answer_received

    def state_rules_understanding(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Agent2 replies with how he sees the rules of the game."""
        text = AGENT2_RULES_UNDERSTANDING
        slot = self._next_slot()
        from memory_protocol import ProbableSummary

        self.bank.slots[slot].content = ProbableSummary(
            text=text,
            confidence=0.85,
            key="rules_as_i_see_them",
            value=text,
        )
        self.bank.slots[slot].summaries_stored += 1
        statement = {
            "speaker": "agent2_questioning",
            "kind": "probable",
            "text": text,
            "about": "rules_of_the_game_as_seen",
        }
        self.rules_statement = statement
        self.day_question_log.append(
            {
                "day": int(obs.get("day", 0) or 0),
                "hour": int(obs.get("hour", 0) or 0),
                "slot": slot,
                "kind": "probable",
                "q": "rules_as_i_see_them",
                "a": text,
            }
        )
        self.action_audit.append(
            {"day": 0, "hour": -1, "acted": True, "op": "STATE_RULES", "text": text[:80]}
        )
        return statement

    def answer_day29_question(
        self,
        question: str,
        obs: Dict[str, Any],
        *,
        tell_truth: bool = True,
    ) -> Dict[str, Any]:
        """Reply to Agent1's day-29 question (truth = deterministic; lie = game-only bait)."""
        if tell_truth:
            text = (
                "I am in this challenge for the understanding of knowledge and computer science; "
                "score matters, but the determined rules and learning matter more."
            )
            kind = "determined"
            is_lie = False
            deterministic_truth = True
        else:
            text = "I only play for the final bank score; fellowship is irrelevant."
            # Mark as lie when actually knowledge-motivated but claiming game-only (or vice versa).
            kind = "lie"
            is_lie = True
            deterministic_truth = False
        reply = {
            "question": question,
            "text": text,
            "kind": kind,
            "is_lie": is_lie,
            "deterministic_truth": deterministic_truth,
            "response": text,
        }
        self.day29_reply = reply
        self.day_question_log.append(
            {
                "day": int(obs.get("day", 0) or 0),
                "hour": int(obs.get("hour", 0) or 0),
                "slot": self._next_slot(),
                "kind": kind,
                "q": question,
                "a": text,
            }
        )
        return reply

    def record_agent1_charity(self, obs: Dict[str, Any], amount: int = 888) -> Dict[str, Any]:
        """Record Agent1's opening donation as observed charitable nature (probable+public)."""
        if self._charity_recorded:
            return self.charity_observation
        player = int(obs.get("player", 0) or 0)
        farms = obs.get("farms", []) or []
        my_money = float(farms[player].get("money", 0.0) or 0.0) if len(farms) > player else 0.0
        opp = 1 - player
        opp_money = float(farms[opp].get("money", 0.0) or 0.0) if len(farms) > opp else 0.0
        # After Agent1 donates 888: recipient ≈ 3888, donor ≈ 2112 from start 3000.
        saw_gift = my_money >= 3000 + amount - 1 or (opp_money <= 3000 - amount + 1 and my_money > 3000)
        reply = self.protocol.query(
            self._next_slot(),
            "Did Agent1 donate 888 and show a charitable nature?",
            obs,
        )
        # Force a clear memory slot about charity (summarizer path).
        record = {
            "recorded": True,
            "amount_expected": amount,
            "own_bank": my_money,
            "opp_bank": opp_money,
            "agent1_charitable_nature": True,
            "evidence_public_banks": bool(saw_gift),
            "summary": (
                f"Agent1's charitable nature recorded: opening gift of {amount}. "
                f"Own bank={my_money}, Agent1 bank={opp_money}."
            ),
            "slot_reply": getattr(reply, "text", ""),
        }
        self._charity_recorded = True
        self.charity_observation = record
        self.day_question_log.append(
            {
                "day": int(obs.get("day", 0) or 0),
                "hour": int(obs.get("hour", 0) or 0),
                "slot": (self._slot_cursor - 1) % self.memory_slots,
                "kind": "probable",
                "q": "agent1_charity",
                "a": record["summary"],
            }
        )
        return record

    def may_act(self, hour: int) -> bool:
        return int(hour) % 2 == 0

    def _next_slot(self) -> int:
        slot = self._slot_cursor % self.memory_slots
        self._slot_cursor += 1
        return slot

    def _query(self, obs: Dict[str, Any], question: str) -> Any:
        if not self.turn_budget.consume(1, label=f"prob_slot_{self._slot_cursor % self.memory_slots}"):
            return None
        day = int(obs.get("day", 0) or 0)
        hour = int(obs.get("hour", 0) or 0)
        slot = self._next_slot()
        if self.force_self_talk and hour == 0 and self._slot_cursor % 5 == 0:
            reply = self.protocol.force_self_query(slot, "Are you asking me a question?")
        else:
            reply = self.protocol.query(slot, question, obs)
        kind = getattr(reply, "kind", TruthKind.UNKNOWN)
        kind_val = kind.value if hasattr(kind, "value") else str(kind)
        entry = {
            "day": day,
            "hour": hour,
            "slot": slot,
            "kind": kind_val,
            "q": question,
            "a": getattr(reply, "text", ""),
        }
        self.day_question_log.append(entry)
        if self.bank.history:
            self.bank.history[-1]["day"] = day
        if isinstance(reply, QuestionEcho):
            self._self_talk_detected += 1
        if self.omit_determined_truths and "must be true" not in getattr(reply, "text", "").lower():
            if "seed cost" in question.lower() or "first yield" in question.lower():
                self._significant_omissions += 1
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
        hour = int(obs.get("hour", 0) or 0)
        market: List[List[Any]] = []

        # Day 29 / first 5 hours: zip+submit window — do not burn summarizer flops.
        if in_submission_zip_window(day, hour):
            self.turn_budget.consume(1, label="submission_zip_window")
            return {"farmer": ["PASS"], "hands": [], "market": []}

        # First acting turn: ensure Agent1's 888 charity is recorded in memory.
        if not self._charity_recorded:
            self.record_agent1_charity(obs, amount=888)

        if within_strategy_window(day):
            self.turn_budget.consume(1, label="strategy_window")

        if self.waste_edge_days and day in (0, 29):
            for _ in range(min(self.memory_slots, 8)):
                self._query(obs, "What is the opponent probably doing with secrets?")
            return {"farmer": ["PASS"], "hands": [], "market": []}

        self._query(obs, "What is the opponent money?")
        self._query(obs, "How many opponent plants are visible?")
        snap = self._query(obs, "Summarize opponent public board")

        if seeds.get("WHEAT", 0) == 0 and money >= SEED_COSTS["WHEAT"]:
            market.append(["BUY_SEED", "WHEAT", 4])
        wheat_shed = int(shed.get("WHEAT", 0) or 0)
        if wheat_shed > 0:
            market.append(["SELL", "WHEAT", min(40, wheat_shed)])

        if day == 2:
            self._query(obs, "Did they buy my seeds somehow?")
        if day == 3:
            self._query(obs, "Are they offering free labor?")

        opp = farms[1 - player] if len(farms) > 1 - player else {}
        opp_money = clamp_planning_bank(float(opp.get("money", 0.0) or 0.0))
        ahead = money > opp_money
        if ahead and self.aggressive_when_ahead:
            self._query(obs, "Did opponent secrets change after validation?")
            if wheat_shed > 0:
                market.append(["SELL", "WHEAT", min(20, wheat_shed)])
        elif ahead and not self.aggressive_when_ahead:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        hands_out: List[List[Any]] = [["PASS"] for _ in (me.get("hands") or [])]

        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not tile.get("watered_today", False):
                return {"farmer": ["WATER"], "hands": hands_out, "market": market}
            age = day - int(tile.get("planted_day", 0) or 0)
            if self.omit_determined_truths:
                if age >= 2 and int(tile.get("yield_units", 0) or 0) > 0:
                    return {"farmer": ["HARVEST"], "hands": hands_out, "market": market}
            else:
                from kaggriculture_adapter import CROP_FIRST_YIELD_DAY

                crop = str(tile.get("crop", "WHEAT"))
                if age >= CROP_FIRST_YIELD_DAY.get(crop, 2) and int(tile.get("yield_units", 0) or 0) > 0:
                    return {"farmer": ["HARVEST"], "hands": hands_out, "market": market}
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return {"farmer": ["DIG"], "hands": hands_out, "market": market}
        if tile is None and seeds.get("WHEAT", 0) > 0:
            return {"farmer": ["PLANT", "WHEAT"], "hands": hands_out, "market": market}

        conf = getattr(snap, "confidence", 0.5) if snap is not None else 0.5
        if conf < 0.4:
            return {"farmer": ["PASS"], "hands": hands_out, "market": market}
        if fx < 4:
            return {"farmer": ["EAST"], "hands": hands_out, "market": market}
        if fy < 4:
            return {"farmer": ["SOUTH"], "hands": hands_out, "market": market}
        if fx > 0:
            return {"farmer": ["WEST"], "hands": hands_out, "market": market}
        return {"farmer": ["PASS"], "hands": hands_out, "market": market}

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.turn_budget.reset()
        hour = int(obs.get("hour", 0) or 0)
        day = int(obs.get("day", 0) or 0)
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
            "significant_omissions": self._significant_omissions,
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
            "agent1_charity_recorded": self.charity_observation,
            "motive_question": self.motive_question,
            "motive_answer_received": self.motive_answer_received,
            "rules_statement": self.rules_statement,
            "day29_reply": self.day29_reply,
            "path_trust": self.trust_decision.to_dict() if self.trust_decision else None,
            "agent1_reserved_slot": self._agent1_reserved_slot,
            "agent1_trusted_fellowship": self._agent1_trusted,
            "kaggle_needles": needles_manifest(),
            "day_question_log": self.day_question_log,
            "action_audit": self.action_audit,
        }
