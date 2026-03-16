"""
Prioritized Path Planner
=========================
Multi-agent pathfinding via Prioritized Planning.

Agents are planned one at a time in priority order. Each agent uses
space-time A* to find a collision-free path, treating higher-priority
agents' planned paths as moving obstacles via a reservation table.

This eliminates most vertex conflicts (two agents on the same cell)
and edge/swap conflicts (two agents crossing the same edge in
opposite directions).
"""

import heapq
from typing import List, Set, Tuple, Dict

from Policies.PathPlanner.base_path_planner import BasePathPlanner


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class PrioritizedPathPlanner(BasePathPlanner):
    """
    Prioritized Planning for multi-agent pathfinding.

    Plans paths sequentially — one agent at a time.  Each agent finds
    a collision-free path using **space-time A***, avoiding cells and
    timesteps already reserved by higher-priority agents.

    Priority is determined by the order the engine calls ``plan()``:
    the first agent planned has the highest priority, the second must
    route around the first, etc.

    Features
    --------
    * **Vertex conflict avoidance** — two agents may not occupy the
      same cell at the same timestep.
    * **Edge (swap) conflict avoidance** — two agents may not swap
      positions in a single timestep.
    * **Wait actions** — an agent can stay in place for one timestep
      to let another agent pass.

    Parameters
    ----------
    max_horizon : int
        Maximum number of timesteps the space-time search will
        explore.  Limits computation time.  Default 100.
    goal_reserve : int
        Number of extra timesteps to reserve the goal position after
        arrival, preventing later agents from occupying it.  Default 20.
    """

    def __init__(self, max_horizon: int = 100, goal_reserve: int = 20):
        self.max_horizon = max_horizon
        self.goal_reserve = goal_reserve

        # Internal state — reset each tick
        self._vertex_res: Set[Tuple[int, int, int]] = set()
        self._edge_res: Set[Tuple[int, int, int, int, int]] = set()
        self._last_tick: int = -1

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plan(
        self,
        agent,
        goal: Tuple[int, int],
        world_state,
    ) -> List[Tuple[int, int]]:
        """
        Compute a collision-free path using space-time A*.

        On the first call of each tick the reservation table is rebuilt
        from agents that already have paths.  Each subsequent call adds
        the newly planned path to the table, so lower-priority agents
        automatically avoid higher-priority ones.

        Returns
        -------
        list[tuple[int, int]]
            Path from start to goal (excluding start), possibly with
            "wait" steps (same position repeated).  Empty if no path.
        """
        tick = world_state.tick

        # Reset reservation table once per tick
        if tick != self._last_tick:
            self._last_tick = tick
            self._vertex_res = set()
            self._edge_res = set()
            self._pre_reserve_existing_paths(world_state)

        start = agent.position
        if start == goal:
            return []

        map_state = world_state.map_state

        # Static obstacles: pod positions when carrying a pod
        static_blocked: Set[Tuple[int, int]] = set()
        if agent.carried_pod_id is not None:
            for pod in world_state.pod_state.pods.values():
                if not pod.is_carried:
                    static_blocked.add(pod.current_position)
        static_blocked.discard(goal)

        # Plan using space-time A*
        path = self._space_time_astar(start, goal, map_state, static_blocked)

        # Reserve the new path so lower-priority agents avoid it
        if path:
            self._reserve_path(start, path)

        return path

    # ------------------------------------------------------------------
    # Reservation table management
    # ------------------------------------------------------------------

    def _pre_reserve_existing_paths(self, world_state) -> None:
        """Populate the reservation table with all agents' current and
        planned future positions, including agents waiting at destinations."""
        for agent in world_state.agents:
            pos = agent.position
            # Reserve current position at time-step 0
            self._vertex_res.add((pos[0], pos[1], 0))

            if agent.is_waiting:
                # Agent is parked at this position for wait_ticks more ticks
                # plus a buffer so new paths don't target this cell too early
                for t in range(1, agent.wait_ticks + 1 + self.goal_reserve):
                    self._vertex_res.add((pos[0], pos[1], t))

            elif agent.has_path:
                prev = pos
                remaining = len(agent.path) - agent.path_index
                for i in range(remaining):
                    step = i + 1
                    nxt = agent.path[agent.path_index + i]
                    self._vertex_res.add((nxt[0], nxt[1], step))
                    # Edge reservation (prevents swaps)
                    self._edge_res.add(
                        (nxt[0], nxt[1], prev[0], prev[1], step)
                    )
                    prev = nxt
                # Reserve the final position for extra timesteps
                for extra in range(remaining + 1, remaining + 1 + self.goal_reserve):
                    self._vertex_res.add((prev[0], prev[1], extra))

    def _reserve_path(
        self,
        start: Tuple[int, int],
        path: List[Tuple[int, int]],
    ) -> None:
        """Reserve vertex and edge reservations for a newly planned path."""
        prev = start
        for step, pos in enumerate(path, start=1):
            self._vertex_res.add((pos[0], pos[1], step))
            self._edge_res.add((pos[0], pos[1], prev[0], prev[1], step))
            prev = pos
        # Reserve goal after arrival
        for extra in range(len(path) + 1, len(path) + 1 + self.goal_reserve):
            self._vertex_res.add((path[-1][0], path[-1][1], extra))

    # ------------------------------------------------------------------
    # Space-time A*
    # ------------------------------------------------------------------

    def _space_time_astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        map_state,
        static_blocked: Set[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """
        A* search in the (row, col, timestep) state space.

        Actions: move to a cardinal neighbour **or** wait in place.
        Avoids vertex and edge reservations from the reservation table.
        """
        counter = 0
        open_set: list = []
        heapq.heappush(
            open_set, (_manhattan(start, goal), counter, start[0], start[1], 0)
        )
        counter += 1

        g_score: Dict[Tuple[int, int, int], int] = {
            (start[0], start[1], 0): 0
        }
        came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        while open_set:
            _, _, r, c, t = heapq.heappop(open_set)

            if (r, c) == goal:
                # Reconstruct path (excluding start)
                path: list = []
                node = (r, c, t)
                while node != (start[0], start[1], 0):
                    path.append((node[0], node[1]))
                    node = came_from[node]
                path.reverse()
                return path

            # Enforce time horizon
            if t >= self.max_horizon:
                continue

            next_t = t + 1

            # Candidate moves: 4 cardinal neighbours + wait-in-place
            candidates = list(map_state.get_neighbors(r, c))
            candidates.append((r, c))  # wait action

            for nr, nc in candidates:
                # Static obstacles
                if (nr, nc) in static_blocked:
                    continue

                # Vertex conflict check
                if (nr, nc, next_t) in self._vertex_res:
                    continue

                # When arriving at the goal, the agent will stay there
                # (for pickup/delivery/return processing).  Ensure the
                # goal cell is clear for the full occupancy window so we
                # don't conflict with another agent that arrives later.
                if (nr, nc) == goal:
                    goal_blocked = False
                    for ft in range(next_t + 1, next_t + 1 + self.goal_reserve):
                        if (nr, nc, ft) in self._vertex_res:
                            goal_blocked = True
                            break
                    if goal_blocked:
                        continue

                # Edge (swap) conflict check — look for reverse direction
                if (r, c, nr, nc, next_t) in self._edge_res:
                    continue

                tentative_g = g_score[(r, c, t)] + 1
                state = (nr, nc, next_t)

                if tentative_g < g_score.get(state, float("inf")):
                    came_from[state] = (r, c, t)
                    g_score[state] = tentative_g
                    f = tentative_g + _manhattan((nr, nc), goal)
                    heapq.heappush(open_set, (f, counter, nr, nc, next_t))
                    counter += 1

        # No path found within the time horizon
        return []
