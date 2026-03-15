"""
Base Policy Module
==================
Abstract base classes (interfaces) for all pluggable policies.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.order_state import Order
    from WorldState.task_state import Task
    from WorldState.agent_state import AgentState


class BaseOrderGenerator(ABC):
    """
    Interface for order generation policies.

    Implementations decide when and how to create new customer orders.
    """

    @abstractmethod
    def generate(self, world_state: "WorldState") -> List["Order"]:
        """
        Generate new orders based on the current world state.

        Parameters
        ----------
        world_state : WorldState
            Current simulation state including tick counter.

        Returns
        -------
        list[Order]
            Newly generated orders (may be empty).
        """
        ...


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


class BasePathPlanner(ABC):
    """
    Interface for path planning policies.

    Implementations compute a path from the agent's current position
    to a goal on the grid.
    """

    @abstractmethod
    def plan(
        self,
        agent: "AgentState",
        goal: Tuple[int, int],
        world_state: "WorldState",
    ) -> List[Tuple[int, int]]:
        """
        Compute a path from the agent's current position to the goal.

        Parameters
        ----------
        agent : AgentState
            The agent requesting a path.
        goal : tuple[int, int]
            Target (row, col) position.
        world_state : WorldState
            Current simulation state (for map & obstacle info).

        Returns
        -------
        list[tuple[int, int]]
            Sequence of (row, col) positions forming the path,
            excluding the agent's current position.
            Empty list if no path is found.
        """
        ...
