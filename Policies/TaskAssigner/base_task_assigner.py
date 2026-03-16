"""
Base Task Assigner
==================
Abstract base class (interface) for task assignment policies.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.task_state import Task


class BaseTaskAssigner(ABC):
    """
    Interface for task assignment policies.

    Implementations decide how to decompose orders into tasks
    and assign them to available agents.

    属性
    ----------
    pod_return_planner : BasePodReturnPlanner or None
        Optional return planner injected after construction.
        If set, used to determine where pods are returned.
    """

    def __init__(self):
        self.pod_return_planner = None  # injected by main.py

    @abstractmethod
    def assign(self, world_state: "WorldState") -> List["Task"]:
        """
        Assign pending orders to idle agents by creating tasks.

        参数
        ----------
        world_state : WorldState
            Current simulation state.

        返回值
        -------
        list[Task]
            Newly created and assigned tasks.
        """
        ...
