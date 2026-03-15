"""
PodState Module
===============
Represents pods (shelves) in the warehouse.
"""

from typing import Tuple, List, Optional, Dict


class Pod:
    """
    A single pod (shelf unit) in the warehouse.

    Attributes
    ----------
    pod_id : int
        Unique pod identifier.
    home_position : tuple[int, int]
        The pod's designated home location on the grid.
    current_position : tuple[int, int]
        The pod's current location (may differ from home when carried).
    is_carried : bool
        Whether the pod is currently being carried by a robot.
    carried_by : int or None
        Agent ID of the robot carrying this pod.
    """

    def __init__(self, pod_id: int, home_position: Tuple[int, int]):
        self.pod_id = pod_id
        self.home_position = home_position
        self.current_position: Tuple[int, int] = home_position
        self.is_carried: bool = False
        self.carried_by: Optional[int] = None

    @property
    def is_at_home(self) -> bool:
        """Check if the pod is at its home position."""
        return self.current_position == self.home_position

    def pick_up(self, agent_id: int):
        """Mark the pod as picked up by an agent."""
        self.is_carried = True
        self.carried_by = agent_id

    def put_down(self, position: Tuple[int, int]):
        """Mark the pod as put down at a position."""
        self.is_carried = False
        self.carried_by = None
        self.current_position = position

    def __repr__(self) -> str:
        state = "carried" if self.is_carried else "stationary"
        return f"Pod(id={self.pod_id}, pos={self.current_position}, {state})"


class PodState:
    """
    Container managing all pods in the warehouse.

    Attributes
    ----------
    pods : dict[int, Pod]
        All pods keyed by pod_id.
    """

    def __init__(self):
        self.pods: Dict[int, Pod] = {}

    def add_pod(self, pod: Pod):
        """Register a new pod."""
        self.pods[pod.pod_id] = pod

    def get_pod(self, pod_id: int) -> Optional[Pod]:
        """Get a pod by its ID."""
        return self.pods.get(pod_id)

    def get_available_pods(self) -> List[Pod]:
        """Return all pods that are at home and not being carried."""
        return [
            p for p in self.pods.values()
            if not p.is_carried and p.is_at_home
        ]

    def get_pod_at(self, position: Tuple[int, int]) -> Optional[Pod]:
        """Return the pod at a given position, if any."""
        for pod in self.pods.values():
            if pod.current_position == position and not pod.is_carried:
                return pod
        return None

    @property
    def total_pods(self) -> int:
        return len(self.pods)

    def __repr__(self) -> str:
        carried = sum(1 for p in self.pods.values() if p.is_carried)
        return f"PodState(total={self.total_pods}, carried={carried})"
