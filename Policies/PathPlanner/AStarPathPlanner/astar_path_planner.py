"""
A* Path Planner
================
Default implementation: A* pathfinding on the warehouse grid.
"""

import heapq
from typing import List, Tuple

from Policies.PathPlanner.base_path_planner import BasePathPlanner


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Heuristic: Manhattan distance."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class AStarPathPlanner(BasePathPlanner):
    """
    A* path planner for single-agent pathfinding on the warehouse grid.

    Uses Manhattan distance as the heuristic.  Avoids cells occupied by
    other agents and, when the robot is carrying a pod, avoids cells
    occupied by stationary pods (shelves).

    Note: This is a *single-agent* planner.  It does NOT solve multi-agent
    conflicts (e.g. two robots heading to the same station).  Conflict
    detection is handled by the simulation engine.
    """

    def __init__(self, avoid_agents: bool = True):
        self.avoid_agents = avoid_agents

    def plan(
        self,
        agent,
        goal: Tuple[int, int],
        world_state,
    ) -> List[Tuple[int, int]]:
        """
        Compute the shortest path from agent's position to goal using A*.

        Returns
        -------
        list[tuple[int, int]]
            Path from start to goal (excluding start position).
            Empty list if no path is found.
        """
        start = agent.position
        if start == goal:
            return []

        map_state = world_state.map_state

        # Build set of blocked cells
        blocked = set()

        # 1. Other agents' current positions
        if self.avoid_agents:
            for other in world_state.agents:
                if other.agent_id != agent.agent_id:
                    blocked.add(other.position)

        # 2. When carrying a pod, the robot+pod footprint cannot pass through
        #    cells occupied by other stationary pods (shelves).
        if agent.carried_pod_id is not None:
            for pod in world_state.pod_state.pods.values():
                if not pod.is_carried:
                    pos = pod.current_position
                    blocked.add(pos)

        # Never block our own goal
        blocked.discard(goal)

        # A* algorithm
        # Priority queue entries: (f_score, counter, position)
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0 + _manhattan_distance(start, goal), counter, start))
        counter += 1

        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = []
                node = goal
                while node != start:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            for neighbor in map_state.get_neighbors(current[0], current[1]):
                if neighbor in blocked:
                    continue

                tentative_g = g_score[current] + 1

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + _manhattan_distance(neighbor, goal)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        # No path found
        return []
