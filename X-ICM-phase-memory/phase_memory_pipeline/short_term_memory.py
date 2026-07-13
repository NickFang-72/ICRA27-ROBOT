"""Short-term memory for one RLBench episode."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _voxel(action: Optional[List[int]]) -> Optional[List[int]]:
    if not action or len(action) != 7:
        return None
    return [int(action[0]), int(action[1]), int(action[2])]


class ShortTermMemory:
    """Track recent action/state information without allowing phase loops."""

    def __init__(self, max_history: int = 8):
        self.max_history = max_history
        self.reset_episode()

    def reset_episode(self) -> None:
        self.current_phase_index: Optional[int] = None
        self.phase_attempts: Dict[int, int] = {}
        self.phase_failures: List[Dict[str, Any]] = []
        self.phase_history: List[Dict[str, Any]] = []
        self.recent_actions: List[Dict[str, Any]] = []
        self.last_action_7d: Optional[List[int]] = None
        self.last_observation_summary = "unknown"
        self.previous_observation_summary = "unknown"

    def start_phase(self, phase_index: int) -> None:
        if self.current_phase_index != phase_index:
            self.current_phase_index = phase_index
            self.phase_attempts.setdefault(phase_index, 0)

    def attempts_for_phase(self, phase_index: int) -> int:
        return int(self.phase_attempts.get(phase_index, 0))

    def repeated_action(self, action_7d: List[int]) -> bool:
        return self.last_action_7d == action_7d

    def repeated_voxel(self, action_7d: List[int]) -> bool:
        return _voxel(self.last_action_7d) == _voxel(action_7d)

    def scene_changed_since_last_action(self) -> bool:
        return self.previous_observation_summary != self.last_observation_summary

    def update_observation(self, observation_summary: str) -> None:
        self.previous_observation_summary = self.last_observation_summary
        self.last_observation_summary = observation_summary or "unknown"

    def record_action(
        self,
        *,
        phase_index: int,
        phase_name: str,
        action_7d: List[int],
        model_status: str,
        guard_decision: Dict[str, Any],
    ) -> None:
        self.phase_attempts[phase_index] = self.attempts_for_phase(phase_index) + 1
        item = {
            "phase_index": phase_index,
            "phase": phase_name,
            "action_7d": list(action_7d),
            "model_status": model_status,
            "guard": guard_decision,
        }
        self.recent_actions.append(item)
        self.recent_actions = self.recent_actions[-self.max_history :]
        self.last_action_7d = list(action_7d)

    def record_phase_transition(
        self,
        *,
        phase_index: int,
        phase_name: str,
        transition: str,
        reason: str,
    ) -> None:
        self.phase_history.append(
            {
                "phase_index": phase_index,
                "phase": phase_name,
                "transition": transition,
                "reason": reason,
            }
        )
        if transition in {"abort", "failed_required", "skipped_optional"}:
            self.phase_failures.append(self.phase_history[-1])

    def as_prompt_dict(self) -> Dict[str, Any]:
        return {
            "current_phase_index": self.current_phase_index,
            "last_action_7d": self.last_action_7d,
            "last_observation_summary": self.last_observation_summary,
            "scene_changed_since_last_action": self.scene_changed_since_last_action(),
            "recent_actions": self.recent_actions[-5:],
            "phase_history": self.phase_history[-8:],
            "phase_failures": self.phase_failures[-5:],
        }
