import json
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from planner.planner_base import BasePlanner
from planner.rule_planner import RulePlanner


class LLMPlanner(BasePlanner):
    """
    V5-b LLM PHASE planner.

    Important:
    - LLM does NOT choose primitive moves anymore.
    - LLM only decides the current high-level phase.
    - RulePlanner remains the robust execution engine.

    Output schema:
    {
      "phase": "find_key" | "go_to_door" | "search_goal" | "go_to_goal" | "recover",
      "target": "key" | "door" | "goal" | null,
      "reason": "..."
    }
    """

    VALID_PHASES = {
        "find_key",
        "go_to_door",
        "search_goal",
        "go_to_goal",
        "recover",
    }

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 80,
        temperature: float = 0.0,
        do_sample: bool = False,
        verbose: bool = True,
        predictor_checkpoint: str = "predictor/checkpoints/jepa_lite_mlp_po_v2.pt",
        use_predictor_hint: bool = False,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.verbose = verbose

        self.use_predictor_hint = use_predictor_hint
        self.predictor = None
        self.predictor_enabled = False

        self.fallback_planner = RulePlanner()

        if self.verbose:
            print(f"[LLMPlanner] Loading tokenizer from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.verbose:
            print("[LLMPlanner] Loading model in 4-bit mode...")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config,
            dtype="auto",
        )

        self.model.eval()

        if self.verbose:
            print("hf_device_map =", getattr(
                self.model, "hf_device_map", None))
            print("first_param_device =", next(self.model.parameters()).device)
            print("[LLMPlanner] 4-bit model loaded successfully.")

        # kept for compatibility; currently not used in V5-b
        if self.use_predictor_hint:
            try:
                from predictor.mlp_predictor import MLPPredictor

                self.predictor = MLPPredictor(
                    checkpoint_path=predictor_checkpoint
                )
                self.predictor_enabled = True
                if self.verbose:
                    print("[LLMPlanner] Predictor hint enabled.")
            except Exception as e:
                self.predictor = None
                self.predictor_enabled = False
                if self.verbose:
                    print(f"[LLMPlanner] Predictor hint disabled: {e}")

    # =========================================================
    # Public API
    # =========================================================

    def choose_phase(
        self,
        z_t: dict,
        memory_summary: Optional[dict] = None,
        memory_patch: Optional[list] = None,
        frontier_candidates: Optional[list] = None,
        loop_hints: Optional[dict] = None,
        planner_context: Optional[dict] = None,
        replan: bool = False,
        last_info: dict | None = None,
    ) -> dict:
        memory_summary = memory_summary or {}
        memory_patch = memory_patch or []
        frontier_candidates = frontier_candidates or []
        loop_hints = loop_hints or {}
        planner_context = planner_context or {}

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            z_t=z_t,
            memory_summary=memory_summary,
            memory_patch=memory_patch,
            frontier_candidates=frontier_candidates,
            loop_hints=loop_hints,
            replan=replan,
            last_info=last_info,
        )

        try:
            raw_text = self._generate(system_prompt, user_prompt)
            phase_decision = self._parse_and_validate_phase(raw_text)
            phase_decision = self._postprocess_phase(
                phase_decision=phase_decision,
                z_t=z_t,
                memory_summary=memory_summary,
                loop_hints=loop_hints,
                last_info=last_info,
            )

            if self.verbose:
                print("[LLMPlanner] raw_phase_response =", raw_text)
                print("[LLMPlanner] parsed_phase =", phase_decision)

            return phase_decision

        except Exception as e:
            if self.verbose:
                print(f"[LLMPlanner] fallback to heuristic phase due to: {e}")

            return self._fallback_phase_decision(
                z_t=z_t,
                memory_summary=memory_summary,
                loop_hints=loop_hints,
                last_info=last_info,
            )

    # backward compatibility
    def choose_skill(
        self,
        z_t: dict,
        memory_summary: Optional[dict] = None,
        memory_patch: Optional[list] = None,
        frontier_candidates: Optional[list] = None,
        loop_hints: Optional[dict] = None,
        planner_context: Optional[dict] = None,
        replan: bool = False,
        last_info: dict | None = None,
    ) -> dict:
        """
        Backward-compat bridge:
        if old code still calls choose_skill on this class,
        convert phase decision into forced_phase and let RulePlanner execute.
        """
        phase_decision = self.choose_phase(
            z_t=z_t,
            memory_summary=memory_summary,
            memory_patch=memory_patch,
            frontier_candidates=frontier_candidates,
            loop_hints=loop_hints,
            planner_context=planner_context,
            replan=replan,
            last_info=last_info,
        )

        planner_context = dict(planner_context or {})
        planner_context["forced_phase"] = phase_decision.get("phase")
        planner_context["phase_reason"] = phase_decision.get("reason")

        return self.fallback_planner.choose_skill(
            z_t=z_t,
            memory_summary=memory_summary or {},
            memory_patch=memory_patch or [],
            frontier_candidates=frontier_candidates or [],
            loop_hints=loop_hints or {},
            planner_context=planner_context,
            replan=replan,
            last_info=last_info,
        )

    # =========================================================
    # Prompt construction
    # =========================================================

    def _build_system_prompt(self) -> str:
        return """You are the slow-layer PHASE planner for a partially observable grid-world task.

Your responsibility:
- decide what high-level phase the agent should be in
- do NOT choose primitive movement actions
- do NOT do pathfinding
- low-level navigation and exploration are handled by another execution system

Possible world objects may include:
- KEY
- DOOR (locked or open)
- GOAL
- WALLS

The overall task may require:
- finding a prerequisite object
- reaching a blocking object
- searching for a final target
- reaching the final target

Available phases:
1. find_key
   - use when the key is required and has not been collected yet
2. go_to_door
   - use when key is already collected and the door should be reached next
3. search_goal
   - use when the door is open but the goal location is still unknown
4. go_to_goal
   - use when the goal location is known and should be reached
5. recover
   - use when the agent is clearly stuck or looping and needs recovery behavior

Return ONLY one valid JSON object.
Do not output markdown.
Do not output code fences.
Do not output explanation outside JSON.

Valid output format:
{"phase":"find_key","target":"key","reason":"..."}
{"phase":"go_to_door","target":"door","reason":"..."}
{"phase":"search_goal","target":"goal","reason":"..."}
{"phase":"go_to_goal","target":"goal","reason":"..."}
{"phase":"recover","target":null,"reason":"..."}

Important:
- If the agent is clearly stuck, choose recover.
- If the key is not yet collected, prefer find_key.
- After the key is collected, the door must be handled before the final goal can be pursued.
- If the key is collected and the door is not open yet:
  - prefer go_to_door if the door location is known
  - otherwise still prefer go_to_door as the next objective (the execution layer will explore to find it)
- Only choose go_to_goal when the goal is known AND the door is already open.
- If the door is open and the goal is not known, prefer search_goal.
"""

    def _build_user_prompt(
        self,
        z_t: dict,
        memory_summary: dict,
        memory_patch: list,
        frontier_candidates: list,
        loop_hints: dict,
        replan: bool,
        last_info: dict | None,
    ) -> str:
        has_key = bool(z_t.get("has_key", False))

        visible_key_pos = z_t.get("visible_key_pos", None)
        visible_door_pos = z_t.get("visible_door_pos", None)
        visible_goal_pos = z_t.get("visible_goal_pos", None)
        visible_door_open = z_t.get("visible_door_open", None)

        known_key_pos = memory_summary.get("known_key_pos", None)
        known_door_pos = memory_summary.get("known_door_pos", None)
        known_goal_pos = memory_summary.get("known_goal_pos", None)
        known_door_open = memory_summary.get("known_door_open", None)

        last_failed_action = None
        if last_info is not None and (
            last_info.get("hit_wall", False)
            or last_info.get("out_of_bounds", False)
            or last_info.get("blocked_by_locked_door", False)
        ):
            last_failed_action = str(last_info.get("action", "")).lower()

        payload = {
            "task_manual": {
                "possible_objects": ["KEY", "DOOR_LOCKED", "DOOR_OPEN", "GOAL", "WALL"],
                "planner_role": "Decide the current high-level phase only. Do not choose primitive actions.",
            },
            "latent_state": {
                "agent_pos": z_t.get("agent_pos"),
                "has_key": has_key,
                "key_visible": z_t.get("key_visible", False),
                "visible_key_pos": visible_key_pos,
                "door_visible": z_t.get("door_visible", False),
                "visible_door_pos": visible_door_pos,
                "visible_door_open": visible_door_open,
                "goal_visible": z_t.get("goal_visible", False),
                "visible_goal_pos": visible_goal_pos,
                "local_walls": z_t.get("local_walls", {}),
            },
            "memory_summary": {
                "known_key_pos": known_key_pos,
                "known_door_pos": known_door_pos,
                "known_door_open": known_door_open,
                "known_goal_pos": known_goal_pos,
                "visited_count": memory_summary.get("visited_count", 0),
                "recent_positions": memory_summary.get("recent_positions", []),
            },
            "loop_hints": loop_hints,
            "replan": replan,
            "recent_events": {
                "picked_key_recently": bool(last_info is not None and last_info.get("picked_key", False)),
                "opened_door_recently": bool(last_info is not None and last_info.get("opened_door", False)),
                "hit_wall_recently": bool(last_info is not None and last_info.get("hit_wall", False)),
                "out_of_bounds_recently": bool(last_info is not None and last_info.get("out_of_bounds", False)),
                "blocked_by_locked_door_recently": bool(last_info is not None and last_info.get("blocked_by_locked_door", False)),
                "last_failed_action": last_failed_action,
            },
            "memory_patch": memory_patch,
            "frontier_candidates": frontier_candidates[:5],
            "instruction": (
                "Choose the best current PHASE. "
                "Do not output primitive moves. "
                "Use recover only when the agent is clearly stuck or looping."
            ),
        }

        payload = self._json_safe(payload)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # =========================================================
    # Parsing / validation
    # =========================================================

    def _parse_and_validate_phase(self, raw_text: str) -> dict:
        json_obj = self._extract_json(raw_text)

        if "phase" not in json_obj:
            raise ValueError("Missing 'phase' field")

        phase = str(json_obj["phase"]).strip().lower()
        if phase not in self.VALID_PHASES:
            raise ValueError(f"Invalid phase: {phase}")

        target = json_obj.get("target", None)
        if target is not None:
            target = str(target).strip().lower()
            if target not in {"key", "door", "goal"}:
                target = None

        reason = str(json_obj.get("reason", "")).strip()

        return {
            "phase": phase,
            "target": target,
            "reason": reason,
        }

    def _extract_json(self, raw_text: str) -> dict:
        raw_text = raw_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        start = raw_text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in model output")

        depth = 0
        for i in range(start, len(raw_text)):
            ch = raw_text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw_text[start:i + 1]
                    return json.loads(candidate)

        raise ValueError("No complete JSON object found in model output")

    # =========================================================
    # Postprocess / fallback
    # =========================================================

    def _postprocess_phase(
        self,
        phase_decision: dict,
        z_t: dict,
        memory_summary: dict,
        loop_hints: dict,
        last_info: dict | None,
    ) -> dict:
        # hard stuck always recover
        if loop_hints.get("is_stuck", False):
            return {
                "phase": "recover",
                "target": None,
                "reason": "loop_hints indicates the agent is stuck",
            }

        if loop_hints.get("oscillation_pair") is not None:
            return {
                "phase": "recover",
                "target": None,
                "reason": "oscillation detected",
            }

        # sanitize with hard world constraints
        has_key = bool(z_t.get("has_key", False))
        known_goal_pos = memory_summary.get("known_goal_pos", None)
        known_door_pos = memory_summary.get("known_door_pos", None)
        known_door_open = memory_summary.get("known_door_open", None)
        visible_goal_pos = z_t.get("visible_goal_pos", None)
        visible_door_pos = z_t.get("visible_door_pos", None)
        visible_door_open = z_t.get("visible_door_open", None)

        goal_known = (visible_goal_pos is not None) or (
            known_goal_pos is not None)
        door_known = (visible_door_pos is not None) or (
            known_door_pos is not None)
        door_open = (visible_door_open is True) or (known_door_open is True)

        phase = phase_decision["phase"]

        if not has_key:
            if phase in {"go_to_door", "search_goal", "go_to_goal"}:
                return {
                    "phase": "find_key",
                    "target": "key",
                    "reason": "key has not been collected yet",
                }

        if has_key and not door_open:
            if phase in {"go_to_goal", "search_goal"}:
                return {
                    "phase": "go_to_door",
                    "target": "door",
                    "reason": "door-first correction before pursuing goal",
                }

            if phase == "go_to_door":
                return {
                    "phase": "go_to_door",
                    "target": "door",
                    "reason": "key collected and door not open yet",
                }

        if has_key and door_open and not goal_known:
            if phase == "go_to_door":
                return {
                    "phase": "search_goal",
                    "target": "goal",
                    "reason": "door already open and goal not known",
                }

        if goal_known and phase == "search_goal":
            return {
                "phase": "go_to_goal",
                "target": "goal",
                "reason": "goal is already known",
            }

        return phase_decision

    def _fallback_phase_decision(
        self,
        z_t: dict,
        memory_summary: dict,
        loop_hints: dict,
        last_info: dict | None,
    ) -> dict:
        if loop_hints.get("is_stuck", False) or loop_hints.get("oscillation_pair") is not None:
            return {
                "phase": "recover",
                "target": None,
                "reason": "fallback: stuck or oscillation",
            }

        has_key = bool(z_t.get("has_key", False))

        visible_key_pos = z_t.get("visible_key_pos", None)
        visible_door_pos = z_t.get("visible_door_pos", None)
        visible_goal_pos = z_t.get("visible_goal_pos", None)
        visible_door_open = z_t.get("visible_door_open", None)

        known_key_pos = memory_summary.get("known_key_pos", None)
        known_door_pos = memory_summary.get("known_door_pos", None)
        known_goal_pos = memory_summary.get("known_goal_pos", None)
        known_door_open = memory_summary.get("known_door_open", None)

        key_known = (visible_key_pos is not None) or (
            known_key_pos is not None)
        door_known = (visible_door_pos is not None) or (
            known_door_pos is not None)
        goal_known = (visible_goal_pos is not None) or (
            known_goal_pos is not None)
        door_open = (visible_door_open is True) or (known_door_open is True)
        if not has_key:
            return {
                "phase": "find_key",
                "target": "key",
                "reason": "fallback: key not collected yet",
            }

        # -----------------------------
        # HARD RULE:
        # once key is collected, door must be handled before goal pursuit
        # -----------------------------

        if not door_open:
            return {
                "phase": "go_to_door",
                "target": "door",
                "reason": "fallback: key collected, door not open yet",
            }

        if goal_known:
            return {
                "phase": "go_to_goal",
                "target": "goal",
                "reason": "fallback: goal is known and door is already open",
            }

        return {
            "phase": "search_goal",
            "target": "goal",
            "reason": "fallback: door open, goal unknown",
        }
    # =========================================================
    # Generation
    # =========================================================

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"

        inputs = self.tokenizer(text, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        return raw_text

    # =========================================================
    # Utils
    # =========================================================

    def _json_safe(self, obj):
        if isinstance(obj, dict):
            safe_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    safe_key = str(k)
                else:
                    safe_key = (
                        str(k)
                        if not isinstance(k, (str, int, float, bool)) and k is not None
                        else k
                    )
                safe_dict[safe_key] = self._json_safe(v)
            return safe_dict

        if isinstance(obj, tuple):
            return [self._json_safe(x) for x in obj]

        if isinstance(obj, list):
            return [self._json_safe(x) for x in obj]

        return obj
