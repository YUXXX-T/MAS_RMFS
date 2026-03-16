"""
AgentState Module
=================
Represents individual robot agent state.
"""

from enum import Enum, auto
from typing import Optional, List, Tuple


class AgentStatus(Enum):
    """Possible states of a robot agent."""
    IDLE = auto()           # Waiting for a task
    MOVING_TO_POD = auto()  # Navigating to pick up a pod
    CARRYING = auto()       # Moving with a pod toward the station
    DELIVERING = auto()     # At station, delivering the pod
    RETURNING = auto()      # Returning the pod to its home
    MOVING = auto()         # Generic movement (e.g., repositioning)


class AgentState:
    """
    State of a single robot agent.

    属性
    ----------
    agent_id : int
        Unique identifier for this agent.
    position : tuple[int, int]
        Current (row, col) on the grid.
    status : AgentStatus
        Current operational status.
    assigned_task_id : int or None
        ID of the currently assigned task.
    carried_pod_id : int or None
        ID of the pod currently being carried.
    path : list[tuple[int, int]]
        Planned path (sequence of cells to traverse). Empty if idle.
    path_index : int
        Current index into the path list.
    wait_ticks : int
        Number of ticks remaining before the current action completes.
        While > 0 the agent cannot move or accept new tasks.
    """

    def __init__(self, agent_id: int, start_position: Tuple[int, int]):
        self.agent_id = agent_id
        self.position: Tuple[int, int] = start_position
        self.status: AgentStatus = AgentStatus.IDLE
        self.assigned_task_id: Optional[int] = None
        self.carried_pod_id: Optional[int] = None
        self.path: List[Tuple[int, int]] = []
        self.path_index: int = 0
        self.wait_ticks: int = 0

    @property
    def is_idle(self) -> bool:
        """Check if agent is available for new tasks."""
        return self.status == AgentStatus.IDLE

    @property
    def is_waiting(self) -> bool:
        """Check if agent is waiting for an action to complete."""
        return self.wait_ticks > 0

    @property
    def has_path(self) -> bool:
        """Check if agent has remaining steps in its path."""
        return self.path_index < len(self.path)

    def advance(self) -> Optional[Tuple[int, int]]:
        """
        Move the agent one step along its planned path.

        返回值
        -------
        tuple[int, int] or None
            The new position, or None if path is exhausted.
        """
        if self.has_path:
            self.position = self.path[self.path_index]
            self.path_index += 1
            return self.position
        return None

    def clear_path(self):
        """Clear the agent's current path."""
        self.path = []
        self.path_index = 0

    def assign_path(self, path: List[Tuple[int, int]]):
        """Set a new path for the agent to follow."""
        self.path = path
        self.path_index = 0

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id}, pos={self.position}, "
            f"status={self.status.name}, task={self.assigned_task_id})"
        )
