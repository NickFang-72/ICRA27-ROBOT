"""Runtime controller for phase-by-phase VLM actions with no repair loop."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from phase_anchor_pipeline.anchor_utils import normalized_text

from .action_guard import (
    DEFAULT_NOOP_ACTION,
    guard_action,
    is_required_phase,
    phase_failure_transition,
    sanitize_action,
)
from .long_term_memory import build_long_term_memory
from .phase_step_prompt import build_phase_step_messages, build_phase_step_user_prompt
from .short_term_memory import ShortTermMemory


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("No JSON object found in phase-step output.")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : index + 1])
    raise ValueError("JSON object was opened but not closed.")


class PhaseMemoryController:
    """Execute an anchored phase plan through per-phase VLM decisions."""

    def __init__(
        self,
        *,
        task_name: str,
        episode_id: int,
        review_root: str,
        savedir: str,
        retrieval_method: str,
        retrieval_prompt: str = "",
        max_actions_per_phase: int = 2,
        max_repeated_voxel: int = 1,
        anchor_distance_limit: float = 32.0,
    ):
        self.task_name = task_name
        self.episode_id = int(episode_id)
        self.review_root = Path(review_root)
        self.task_dir = self._find_task_dir()
        self.savedir = Path(savedir)
        self.retrieval_method = retrieval_method
        self.max_actions_per_phase = int(max_actions_per_phase)
        self.max_repeated_voxel = int(max_repeated_voxel)
        self.anchor_distance_limit = float(anchor_distance_limit)
        self.memory = ShortTermMemory()
        self.phase_pointer = 0
        self.aborted = False
        self.completed = False
        self.step_index = 0

        combined_path = self.task_dir / "08_phase_anchor_pipeline_combined.json"
        normalized_path = self.task_dir / "05_normalized_anchored_phase_plan.json"
        extracted_path = self.task_dir / "01_extracted_query.json"
        if combined_path.exists():
            combined = _read_json(combined_path)
            self.normalized_plan = combined.get("normalized_phase_plan") or {}
        elif normalized_path.exists():
            self.normalized_plan = _read_json(normalized_path)
        else:
            raise RuntimeError(f"Missing normalized phase plan under {self.task_dir}")
        self.query_context = _read_json(extracted_path) if extracted_path.exists() else {}
        self.phases = list(self.normalized_plan.get("anchored_phase_plan") or [])
        self.compiled_actions = self._load_compiled_actions()
        self.action_candidates_by_phase = self._build_action_candidates_by_phase()
        self.long_term_memory = build_long_term_memory(
            task_name=task_name,
            instruction=self.instruction,
            normalized_plan=self.normalized_plan,
            retrieval_prompt=retrieval_prompt,
            retrieval_method=retrieval_method,
        )
        self._write_manifest()

    @property
    def instruction(self) -> str:
        return (
            self.query_context.get("instruction")
            or self.normalized_plan.get("instruction")
            or self.task_name
        )

    def _find_task_dir(self) -> Path:
        patterns = [
            f"*_{self.task_name}_episode{self.episode_id}",
            f"{self.task_name}_episode{self.episode_id}",
            f"*{self.task_name}*episode{self.episode_id}*",
        ]
        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(path for path in self.review_root.glob(pattern) if path.is_dir())
        if not matches:
            raise RuntimeError(
                f"No phase-memory review packet found for task={self.task_name}, "
                f"episode={self.episode_id} under {self.review_root}"
            )
        return sorted(matches)[0]

    def _runtime_dir(self) -> Path:
        return self.savedir / "phase_memory_runtime" / self.task_name / f"episode{self.episode_id}"

    def _write_manifest(self) -> None:
        _write_json(
            self._runtime_dir() / "00_phase_memory_manifest.json",
            {
                "task": self.task_name,
                "episode_id": self.episode_id,
                "source_phase_packet": str(self.task_dir),
                "max_actions_per_phase": self.max_actions_per_phase,
                "max_repeated_voxel": self.max_repeated_voxel,
                "anchor_distance_limit": self.anchor_distance_limit,
                "phase_count": len(self.phases),
                "candidate_action_count": sum(len(items) for items in self.action_candidates_by_phase.values()),
                "candidate_action_mode": "selected_action_id",
                "long_term_memory": self.long_term_memory,
            },
        )

    def _load_compiled_actions(self) -> Dict[str, Any]:
        action_path = self.task_dir / "06_compiled_open_loop_actions.json"
        if not action_path.exists():
            return {}
        try:
            return _read_json(action_path)
        except Exception:
            return {}

    def _build_action_candidates_by_phase(self) -> Dict[int, List[Dict[str, Any]]]:
        candidates_by_phase: Dict[int, List[Dict[str, Any]]] = {}
        for step in self.compiled_actions.get("compiled_steps") or []:
            try:
                phase_index = int(step.get("phase_index") or 0)
            except (TypeError, ValueError):
                continue
            if phase_index <= 0:
                continue
            phase_name = normalized_text(step.get("phase") or f"phase_{phase_index}") or "phase"
            phase_candidates = candidates_by_phase.setdefault(phase_index, [])
            action_id = f"phase_{phase_index:02d}_{phase_name}_candidate_{len(phase_candidates) + 1:02d}"
            action = sanitize_action(step.get("action_7d"))
            phase_candidates.append(
                {
                    "action_id": action_id,
                    "action_7d": action,
                    "phase_index": phase_index,
                    "phase": step.get("phase"),
                    "anchor_role": step.get("anchor_role"),
                    "anchor_index": step.get("anchor_index"),
                    "anchor_voxel_xyz": step.get("anchor_voxel_xyz"),
                    "resolved_voxel_xyz": step.get("resolved_voxel_xyz"),
                    "extra_motion_offset_voxel_xyz": step.get("extra_motion_offset_voxel_xyz"),
                    "rotation_discrete_euler": step.get("rotation_discrete_euler"),
                    "gripper": step.get("gripper"),
                    "source": "compiled_open_loop_actions",
                }
            )
        return candidates_by_phase

    def _action_candidates_for_phase(self, phase: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            phase_index = int(phase.get("phase_index") or (self.phase_pointer + 1))
        except (TypeError, ValueError):
            phase_index = self.phase_pointer + 1
        return list(self.action_candidates_by_phase.get(phase_index, []))

    def _choose_action_candidate(
        self,
        parsed: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> tuple[Optional[Dict[str, Any]], str]:
        if not candidates:
            return None, "no_candidates"
        by_id = {str(candidate.get("action_id")): candidate for candidate in candidates}
        requested = str(parsed.get("selected_action_id") or "").strip()
        if requested in by_id:
            return by_id[requested], "selected_action_id"

        normalized_by_id = {normalized_text(key): value for key, value in by_id.items()}
        normalized_requested = normalized_text(requested)
        if normalized_requested in normalized_by_id:
            return normalized_by_id[normalized_requested], "normalized_selected_action_id"

        # Compatibility shim: if the model outputs the exact compiled action,
        # map it back to the id. This keeps old prompts from causing bad casts.
        raw_action = parsed.get("next_action_7d") or parsed.get("action_7d")
        if isinstance(raw_action, list) and len(raw_action) == 7:
            action = sanitize_action(raw_action)
            for candidate in candidates:
                if candidate.get("action_7d") == action:
                    return candidate, "matched_legacy_action_7d"

        if len(candidates) == 1:
            return candidates[0], "single_candidate_fallback"
        return None, "invalid_or_missing_selected_action_id"

    def _execute_candidate(
        self,
        *,
        phase: Dict[str, Any],
        phase_index: int,
        attempts: int,
        candidate: Dict[str, Any],
        status: str,
        step_dir: Path,
        decision_reason: str,
    ) -> Optional[List[int]]:
        action = sanitize_action(candidate.get("action_7d"))
        guard = guard_action(
            phase=phase,
            action_7d=action,
            attempts_so_far=attempts,
            repeated_action=self.memory.repeated_action(action),
            repeated_voxel=self.memory.repeated_voxel(action),
            max_actions_per_phase=self.max_actions_per_phase,
            max_repeated_voxel=self.max_repeated_voxel,
            anchor_distance_limit=self.anchor_distance_limit,
        )
        _write_json(
            step_dir / "03_controller_decision.json",
            {
                "decision": "execute_selected_candidate",
                "selection_reason": decision_reason,
                "selected_candidate": candidate,
                "guard": guard,
            },
        )
        if not guard["allowed"]:
            self._transition_phase(guard["decision"])
            return None

        final_action = guard["action_7d"]
        self.memory.record_action(
            phase_index=phase_index,
            phase_name=str(phase.get("phase") or "unknown"),
            action_7d=final_action,
            model_status=status,
            guard_decision=guard,
        )
        if os.environ.get("XICM_PHASE_MEMORY_ADVANCE_AFTER_CANDIDATE", "1") != "0":
            self._advance_phase("executed_selected_candidate")
        _write_json(step_dir / "04_short_term_memory_after.json", self.memory.as_prompt_dict())
        return final_action

    def current_phase(self) -> Optional[Dict[str, Any]]:
        if self.aborted or self.completed:
            return None
        while self.phase_pointer < len(self.phases):
            phase = self.phases[self.phase_pointer]
            phase_index = int(phase.get("phase_index") or (self.phase_pointer + 1))
            self.memory.start_phase(phase_index)
            if self.memory.attempts_for_phase(phase_index) >= self.max_actions_per_phase:
                self._transition_phase("budget_exhausted_before_prompt")
                continue
            return phase
        self.completed = True
        return None

    def _advance_phase(self, reason: str) -> None:
        phase = self.phases[self.phase_pointer] if self.phase_pointer < len(self.phases) else {}
        phase_index = int(phase.get("phase_index") or (self.phase_pointer + 1))
        self.memory.record_phase_transition(
            phase_index=phase_index,
            phase_name=str(phase.get("phase") or "unknown"),
            transition="advance",
            reason=reason,
        )
        self.phase_pointer += 1
        if self.phase_pointer >= len(self.phases):
            self.completed = True

    def _transition_phase(self, reason: str) -> None:
        phase = self.phases[self.phase_pointer] if self.phase_pointer < len(self.phases) else {}
        transition = phase_failure_transition(phase)
        phase_index = int(phase.get("phase_index") or (self.phase_pointer + 1))
        self.memory.record_phase_transition(
            phase_index=phase_index,
            phase_name=str(phase.get("phase") or "unknown"),
            transition=transition,
            reason=reason,
        )
        if transition == "failed_required":
            self.aborted = True
        else:
            self.phase_pointer += 1
            if self.phase_pointer >= len(self.phases):
                self.completed = True

    def _parse_output(self, raw_output: str) -> Dict[str, Any]:
        try:
            parsed = _extract_first_json_object(raw_output)
        except Exception as exc:
            return {
                "phase_status": "failed",
                "next_action_7d": None,
                "parse_error": str(exc),
                "why_this_action": "model output was not valid JSON",
                "safe_to_advance": False,
            }
        status = normalized_text(parsed.get("phase_status"))
        if status not in {"continue", "done", "failed"}:
            parsed["phase_status"] = "failed"
            parsed["parse_warning"] = f"invalid phase_status={status}"
        return parsed

    def next_action(
        self,
        *,
        observation_summary: str,
        front_rgb_path: Optional[str],
        overhead_rgb_path: Optional[str],
        generate_text: Callable[[List[Dict[str, Any]], int], str],
    ) -> List[int]:
        self.memory.update_observation(observation_summary)
        for _ in range(4):
            phase = self.current_phase()
            if phase is None:
                return list(DEFAULT_NOOP_ACTION)
            phase_index = int(phase.get("phase_index") or (self.phase_pointer + 1))
            attempts = self.memory.attempts_for_phase(phase_index)
            action_candidates = self._action_candidates_for_phase(phase)
            prompt = build_phase_step_user_prompt(
                task=self.task_name,
                instruction=self.instruction,
                phase=phase,
                all_phases=self.phases,
                action_candidates=action_candidates,
                long_term_memory=self.long_term_memory,
                short_term_memory=self.memory.as_prompt_dict(),
                observation_summary=observation_summary,
                max_actions_per_phase=self.max_actions_per_phase,
            )
            messages = build_phase_step_messages(
                prompt,
                front_rgb_path=front_rgb_path,
                overhead_rgb_path=overhead_rgb_path,
            )
            raw_output = generate_text(messages, int(os.environ.get("XICM_PHASE_MEMORY_MAX_TOKENS", "360")))
            parsed = self._parse_output(raw_output)
            status = normalized_text(parsed.get("phase_status"))
            step_dir = self._runtime_dir() / f"step_{self.step_index:03d}_phase_{phase_index:02d}"
            _write_json(
                step_dir / "01_prompt_packet.json",
                {
                    "phase": phase,
                    "prompt": prompt,
                    "messages_preview": messages,
                    "short_term_memory": self.memory.as_prompt_dict(),
                    "action_candidates": action_candidates,
                },
            )
            _write_json(step_dir / "02_vlm_output.json", {"raw_output": raw_output, "parsed": parsed})
            self.step_index += 1

            if status == "done":
                if action_candidates and is_required_phase(phase) and attempts == 0 and os.environ.get("XICM_PHASE_MEMORY_ALLOW_ZERO_ATTEMPT_DONE", "0") != "1":
                    action = self._execute_candidate(
                        phase=phase,
                        phase_index=phase_index,
                        attempts=attempts,
                        candidate=action_candidates[0],
                        status="continue_forced_from_done",
                        step_dir=step_dir,
                        decision_reason="required_phase_reported_done_before_any_action",
                    )
                    if action is not None:
                        return action
                    continue
                if parsed.get("safe_to_advance") is False:
                    self._transition_phase("model_reported_done_but_not_safe_to_advance")
                    _write_json(
                        step_dir / "03_controller_decision.json",
                        {"decision": "phase_failed", "reason": "done_not_safe_to_advance", "aborted": self.aborted},
                    )
                    continue
                self._advance_phase("model_reported_phase_done")
                _write_json(step_dir / "03_controller_decision.json", {"decision": "advance", "reason": "phase_done"})
                continue
            if status == "failed":
                self._transition_phase(str(parsed.get("parse_error") or parsed.get("why_this_action") or "model_reported_failed"))
                _write_json(step_dir / "03_controller_decision.json", {"decision": "phase_failed", "aborted": self.aborted})
                continue

            candidate, selection_reason = self._choose_action_candidate(parsed, action_candidates)
            if candidate is None:
                self._transition_phase(selection_reason)
                _write_json(
                    step_dir / "03_controller_decision.json",
                    {
                        "decision": "phase_failed",
                        "reason": selection_reason,
                        "candidate_count": len(action_candidates),
                        "aborted": self.aborted,
                    },
                )
                continue

            action = self._execute_candidate(
                phase=phase,
                phase_index=phase_index,
                attempts=attempts,
                candidate=candidate,
                status=status,
                step_dir=step_dir,
                decision_reason=selection_reason,
            )
            if action is not None:
                return action
            continue
        return list(DEFAULT_NOOP_ACTION)
