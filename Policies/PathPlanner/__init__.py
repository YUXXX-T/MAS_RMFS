from .base_path_planner import BasePathPlanner
from .AStarPathPlanner import AStarPathPlanner
from .PrioritizedPathPlanner import PrioritizedPathPlanner

from Policies.policy_registry import register
register("path_planner", "AStarPathPlanner", AStarPathPlanner)
register("path_planner", "PrioritizedPathPlanner", PrioritizedPathPlanner)

__all__ = ["BasePathPlanner", "AStarPathPlanner", "PrioritizedPathPlanner"]
