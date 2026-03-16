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
    """

    @abstractmethod
    def assign(self, world_state: "WorldState") -> List["Task"]:
        """
        Assign pending orders to idle agents by creating tasks.

        Parameters
        ----------
        world_state : WorldState
            Current simulation state.

        Returns
        -------
        list[Task]
            Newly created and assigned tasks.
        """
        ...
