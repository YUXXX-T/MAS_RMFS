from .base_task_assigner import BaseTaskAssigner
from .GreedyTaskAssigner import GreedyTaskAssigner

from Policies.policy_registry import register
register("task_assigner", "GreedyTaskAssigner", GreedyTaskAssigner)

__all__ = ["BaseTaskAssigner", "GreedyTaskAssigner"]
