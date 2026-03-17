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
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        do_sample: bool = False,
        verbose: bool = True,
        predictor_checkpoint: str = "predictor/checkpoints/jepa_lite_mlp_po_v2.pt",
        use_predictor_hint: bool = True,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.verbose = verbose

        self.use_predictor_hint = use_predictor_hint
        self.predictor = None
        self.predictor_enabled = False

        # Temporary fallback.
        # Later this can be replaced by a dedicated fast policy.
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

        # Optional predictor hint
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
        memory_summary = memory_summary or {}
        memory_patch = memory_patch or []
        frontier_candidates = frontier_candidates or []
        loop_hints = loop_hints or {}

        predictor_hints = self._build_predictor_hints(
            z_t=z_t,
            memory_summary=memory_summary,
            last_info=last_info,
        )

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            z_t=z_t,
            memory_summary=memory_summary,
            memory_patch=memory_patch,
            frontier_candidates=frontier_candidates,
            loop_hints=loop_hints,
            predictor_hints=predictor_hints,
            replan=replan,
            last_info=last_info,
        )

        try:
            raw_text = self._generate(system_prompt, user_prompt)
            skill_spec = self._parse_and_validate(raw_text)
            skill_spec = self._postprocess_skill(
                skill_spec=skill_spec,
                z_t=z_t,
                memory_summary=memory_summary,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                last_info=last_info,
            )

            if self.verbose:
                print("[LLMPlanner] predictor_hints =", predictor_hints)
                print("[LLMPlanner] raw_response =", raw_text)
                print("[LLMPlanner] parsed_skill =", skill_spec)

            return skill_spec

        except Exception as e:
            if self.verbose:
                print(f"[LLMPlanner] fallback to RulePlanner due to: {e}")

            return self.fallback_planner.choose_skill(
                z_t=z_t,
                memory_summary=memory_summary,
                memory_patch=memory_patch,
                frontier_candidates=frontier_candidates,
                loop_hints=loop_hints,
                replan=replan,
                last_info=last_info,
            )

    def _build_system_prompt(self) -> str:
        return """You are the slow-layer strategic planner for a maze agent in a partially observable environment.

Your job is to choose exactly one next skill.

Allowed skills:
1. move
2. scan
3. escape_loop

Return ONLY one valid JSON object.
Do not output markdown.
Do not output code fences.
Do not output explanation.
Do not output extra text before or after JSON.

Valid JSON formats:
{"skill":"move","args":{"direction":"up"}}
{"skill":"move","args":{"direction":"down"}}
{"skill":"move","args":{"direction":"left"}}
{"skill":"move","args":{"direction":"right"}}
{"skill":"scan","args":{}}
{"skill":"escape_loop","args":{}}

Critical decision rules:
- You are a slow-layer planner, not a per-step reflex controller.
- Use move for the next local step when a safe direction is clear.
- Use scan when new perception is needed or exploration is uncertain.
- Use escape_loop when the agent is stuck in repeated positions.
- Never choose a direction that is currently blocked by a wall.
- If the previous move hit a wall, do not choose the same direction again.
- If a known goal position exists, prefer moving toward that remembered goal.
- If the goal is not known, use frontier information and memory_patch for exploration.
- Avoid repeated scan unless there is clear uncertainty or lack of progress.
- If predictor_hints are available, use them as one-step decision aids.
- Predictor hints are advisory, not mandatory.
"""

    def _build_user_prompt(
        self,
        z_t: dict,
        memory_summary: dict,
        memory_patch: list,
        frontier_candidates: list,
        loop_hints: dict,
        predictor_hints: dict,
        replan: bool,
        last_info: dict | None,
    ) -> str:
        walls = z_t.get("local_walls", {})
        blocked_directions = [
            d for d in ["up", "down", "left", "right"]
            if walls.get(d, False)
        ]
        open_directions = [
            d for d in ["up", "down", "left", "right"]
            if not walls.get(d, False)
        ]

        last_failed_action = None
        if last_info is not None and (
            last_info.get("hit_wall", False)
            or last_info.get("out_of_bounds", False)
        ):
            last_failed_action = str(last_info.get("action", "")).lower()

        planner_meta = {
            "replan": replan,
            "just_scanned": bool(last_info is not None and last_info.get("scan", False)),
            "hit_wall_recently": bool(last_info is not None and last_info.get("hit_wall", False)),
            "out_of_bounds_recently": bool(last_info is not None and last_info.get("out_of_bounds", False)),
            "goal_reached_recently": bool(last_info is not None and last_info.get("goal_reached", False)),
            "last_failed_action": last_failed_action,
            "blocked_directions": blocked_directions,
            "open_directions": open_directions,
        }

        payload = {
            "latent_state": z_t,
            "memory_summary": memory_summary,
            "memory_patch": memory_patch,
            "frontier_candidates": frontier_candidates[:5],
            "loop_hints": loop_hints,
            "predictor_hints": predictor_hints,
            "planner_context": planner_meta,
            "task": (
                "Choose exactly one next skill for strategic replanning. "
                "Use memory_patch for spatial reasoning, use frontier_candidates for exploration, "
                "use loop_hints to avoid oscillation, and use predictor_hints as one-step rollout aids."
            ),
            "output_requirement": "Return valid JSON only.",
        }

        payload = self._json_safe(payload)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_predictor_hints(
        self,
        z_t: dict,
        memory_summary: dict,
        last_info: dict | None,
    ) -> dict:
        if not self.predictor_enabled or self.predictor is None:
            return {
                "enabled": False,
                "candidate_rollouts": {}
            }

        walls = z_t.get("local_walls", {})
        recent_positions = memory_summary.get("recent_positions", [])
        recent_set = set(tuple(p) for p in recent_positions[-6:])

        last_failed_direction = None
        if last_info is not None and (
            last_info.get("hit_wall", False) or last_info.get(
                "out_of_bounds", False)
        ):
            last_failed_direction = str(last_info.get("action", "")).lower()

        candidate_rollouts = {}

        for direction in ["up", "down", "left", "right"]:
            if walls.get(direction, False):
                candidate_rollouts[direction] = {
                    "available": False,
                    "reason": "blocked_now",
                }
                continue

            if last_failed_direction is not None and direction == last_failed_direction:
                candidate_rollouts[direction] = {
                    "available": False,
                    "reason": "same_as_recent_failed_direction",
                }
                continue

            skill_spec = {
                "skill": "move",
                "args": {"direction": direction.upper()},
            }

            try:
                z_hat = self.predictor.predict_next_state(z_t, skill_spec)
            except Exception as e:
                candidate_rollouts[direction] = {
                    "available": False,
                    "reason": f"predictor_error: {str(e)}",
                }
                continue

            pred_pos = tuple(z_hat.get("agent_pos", z_t["agent_pos"]))
            pred_goal_visible = z_hat.get("goal_visible", False)
            pred_goal_distance = z_hat.get("goal_distance", None)
            pred_dx = z_hat.get("dx", None)
            pred_dy = z_hat.get("dy", None)
            pred_local_walls = z_hat.get("local_walls", {})

            candidate_rollouts[direction] = {
                "available": True,
                "predicted_agent_pos": pred_pos,
                "predicted_goal_visible": pred_goal_visible,
                "predicted_goal_distance": pred_goal_distance,
                "predicted_dx": pred_dx,
                "predicted_dy": pred_dy,
                "predicted_local_walls": pred_local_walls,
                "predicted_recent_repeat": pred_pos in recent_set,
            }

        return {
            "enabled": True,
            "candidate_rollouts": candidate_rollouts,
        }

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

    def _parse_and_validate(self, raw_text: str) -> dict:
        json_obj = self._extract_json(raw_text)

        if "skill" not in json_obj:
            raise ValueError("Missing 'skill' field")

        skill = str(json_obj["skill"]).strip().lower()
        args = json_obj.get("args", {})

        if skill == "scan":
            if not isinstance(args, dict):
                raise ValueError("scan args must be a dict")
            return {"skill": "scan", "args": {}}

        if skill == "escape_loop":
            if not isinstance(args, dict):
                raise ValueError("escape_loop args must be a dict")
            return {"skill": "escape_loop", "args": {}}

        if skill == "move":
            if not isinstance(args, dict):
                raise ValueError("move args must be a dict")

            direction = str(args.get("direction", "")).strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                raise ValueError(f"Invalid move direction: {direction}")

            return {
                "skill": "move",
                "args": {"direction": direction.upper()},
            }

        raise ValueError(f"Unknown skill: {skill}")

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

    def _postprocess_skill(
        self,
        skill_spec: dict,
        z_t: dict,
        memory_summary: dict,
        frontier_candidates: list,
        loop_hints: dict,
        last_info: dict | None,
    ) -> dict:
        walls = z_t.get("local_walls", {})

        # hard stuck -> escape_loop
        if loop_hints.get("is_stuck", False):
            return {"skill": "escape_loop", "args": {}}

        if skill_spec["skill"] == "move":
            chosen = skill_spec["args"]["direction"].lower()
            if self._direction_is_unsafe(chosen, walls, last_info):
                replacement = self._choose_safe_direction(
                    z_t=z_t,
                    memory_summary=memory_summary,
                    avoid_direction=self._last_failed_direction(last_info),
                )
                if replacement is None:
                    return {"skill": "scan", "args": {}}
                return {"skill": "move", "args": {"direction": replacement.upper()}}
            return skill_spec

        return skill_spec

    def _last_failed_direction(self, last_info: dict | None) -> str | None:
        if last_info is None:
            return None
        if last_info.get("hit_wall") or last_info.get("out_of_bounds"):
            return str(last_info.get("action", "")).strip().lower()
        return None

    def _direction_is_unsafe(
        self,
        direction: str,
        walls: dict,
        last_info: dict | None,
    ) -> bool:
        if walls.get(direction, False):
            return True

        last_failed_action = self._last_failed_direction(last_info)
        if last_failed_action is not None and direction == last_failed_action:
            return True

        return False

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

    def _choose_safe_direction(
        self,
        z_t: dict,
        memory_summary: dict,
        avoid_direction: str | None = None,
    ) -> str | None:
        walls = z_t.get("local_walls", {})
        r, c = z_t["agent_pos"]

        recent_positions = memory_summary.get("recent_positions", [])
        recent_set = set(tuple(p) for p in recent_positions[-4:])

        known_goal_pos = memory_summary.get("known_goal_pos", None)

        candidates = []

        if known_goal_pos is not None:
            gr, gc = tuple(known_goal_pos)

            if gr < r:
                candidates.append("up")
            elif gr > r:
                candidates.append("down")

            if gc < c:
                candidates.append("left")
            elif gc > c:
                candidates.append("right")

        for d in ["up", "right", "down", "left"]:
            if d not in candidates:
                candidates.append(d)

        candidates = [d for d in candidates if not walls.get(d, False)]

        if avoid_direction is not None:
            filtered = [d for d in candidates if d != avoid_direction]
            if filtered:
                candidates = filtered

        if not candidates:
            return None

        fresh_candidates = []
        for d in candidates:
            nr, nc = self._simulate_move(r, c, d)
            if (nr, nc) not in recent_set:
                fresh_candidates.append(d)

        if fresh_candidates:
            candidates = fresh_candidates

        return candidates[0]

    def _simulate_move(
        self,
        row: int,
        col: int,
        direction: str,
    ) -> tuple[int, int]:
        if direction == "up":
            return (row - 1, col)
        if direction == "down":
            return (row + 1, col)
        if direction == "left":
            return (row, col - 1)
        if direction == "right":
            return (row, col + 1)
        return (row, col)
