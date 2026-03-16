"""
Home Return Planner
====================
Default implementation: always returns pods to their original home position.
"""

from typing import Tuple, TYPE_CHECKING

from Policies.PodReturnPlanner.base_pod_return_planner import BasePodReturnPlanner

if TYPE_CHECKING:
    from WorldState.pod_state import Pod
    from WorldState.world import WorldState


class HomeReturnPlanner(BasePodReturnPlanner):
    """
    Always return the pod to its original home position.

    This is the simplest strategy — each pod goes back exactly
    where it started.
    """

    def plan_return(
        self,
        pod: "Pod",
        station_pos: Tuple[int, int],
        world_state: "WorldState",
    ) -> Tuple[int, int]:
        return pod.home_position
