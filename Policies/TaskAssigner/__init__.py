from .base_task_assigner import BaseTaskAssigner
from .GreedyTaskAssigner import GreedyTaskAssigner
from .HungarianTaskAssigner import HungarianTaskAssigner

from Policies.policy_registry import register
register("task_assigner", "GreedyTaskAssigner", GreedyTaskAssigner)
register("task_assigner", "HungarianTaskAssigner", HungarianTaskAssigner)

__all__ = ["BaseTaskAssigner", "GreedyTaskAssigner", "HungarianTaskAssigner"]
