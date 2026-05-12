"""
PIBT Path Planner
=================
Priority Inheritance with Backtracking — per-step decision planner.

PIBT 路径规划器
=================
基于优先级继承与回溯的每步决策规划器。

Unlike traditional planners that compute an entire path at once,
PIBT returns a single next-step position each tick.  The simulation
engine re-calls ``plan()`` every tick because the assigned 1-step
path is immediately exhausted after one ``advance()``.

与传统一次性计算完整路径的规划器不同，
PIBT 每个 tick 仅返回一个下一步位置。
由于分配的单步路径在一次 advance() 后立即耗尽，
仿真引擎会在每个 tick 重新调用 plan()。

On the first ``plan()`` call of each tick the algorithm runs a
**bulk PIBT** pass: all active agents are sorted by dynamic priority
and processed through the PIBT recursion.  Subsequent ``plan()``
calls within the same tick return cached results.

在每个 tick 的首次 plan() 调用时，算法执行一次**批量 PIBT**：
所有活跃智能体按动态优先级排序后，依次通过 PIBT 递归处理。
同一 tick 内的后续 plan() 调用直接返回缓存结果。

Priority aging ensures fairness: agents that fail to make progress
accumulate higher priority and eventually get to move first.

优先级老化保证公平性：未能取得进展的智能体会逐步提升优先级，
最终获得优先移动的机会。

Reference
---------
Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2019).
Priority Inheritance with Backtracking for Iterative Multi-agent
Path Finding. *IJCAI-19*.
"""

from collections import deque
import random
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from Policies.PathPlanner.base_path_planner import BasePathPlanner

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.map_state import MapState
    from WorldState.agent_state import AgentState


def _bfs_distances(
    goal: Tuple[int, int],
    map_state: "MapState",
    blocked: Optional[Set[Tuple[int, int]]] = None,
) -> Dict[Tuple[int, int], int]:
    """Reverse BFS from *goal* — returns {cell: shortest distance to goal}.

    If *blocked* is provided, those cells are treated as impassable
    (except the goal itself).
    """
    dist: Dict[Tuple[int, int], int] = {goal: 0}
    queue: deque[Tuple[int, int]] = deque([goal])
    while queue:
        cur = queue.popleft()
        d = dist[cur] + 1
        for nb in map_state.get_neighbors(*cur):
            if nb not in dist and (blocked is None or nb not in blocked):
                dist[nb] = d
                queue.append(nb)
    return dist


class PIBTPlanner(BasePathPlanner):
    """
    Per-step PIBT planner with priority inheritance, backtracking,
    and dynamic priority aging.

    带有优先级继承、回溯和动态优先级老化的每步 PIBT 规划器。
    """

    def __init__(self, seed: int = 0):
        self._last_tick: int = -1
        self._goals: Dict[int, Tuple[int, int]] = {}
        self._priorities: Dict[int, int] = {}
        self._arrived: Dict[int, Tuple[int, int]] = {}
        self._base_seed = seed
        self._rng = random.Random(seed)

        # BFS distance caches: goal -> {cell -> distance}
        # _dist_cache: ignoring pods (for non-carrying agents)
        # _carry_dist_cache: respecting pods (for carrying agents), rebuilt each tick
        self._dist_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}
        self._dist_cache_map_id: Optional[int] = None
        self._carry_dist_cache: Dict[Tuple[int, int], Dict[Tuple[int, int], int]] = {}

        # Per-tick state (reset in _init_tick)
        self._decided: Dict[int, Tuple[int, int]] = {}
        self._next_occupied: Dict[Tuple[int, int], int] = {}
        self._undecided: Set[int] = set()
        self._agents: Dict[int, "AgentState"] = {}
        self._cur_pos_index: Dict[Tuple[int, int], List[int]] = {}
        self._static_blocked: Set[Tuple[int, int]] = set()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_goals(self, goals: Dict[int, Tuple[int, int]]) -> None:
        """Pre-populate agent goals (used by MAPFRunner before tick 0)."""
        self._goals.update(goals)

    def _dist(
        self,
        pos: Tuple[int, int],
        goal: Tuple[int, int],
        map_state: "MapState",
        carrying: bool = False,
    ) -> int:
        """BFS shortest-path distance from *pos* to *goal* (cached per goal).

        When *carrying* is True, the pod-aware cache is used (cells
        occupied by non-carried pods are treated as impassable).
        """
        if carrying:
            if goal not in self._carry_dist_cache:
                self._carry_dist_cache[goal] = _bfs_distances(
                    goal, map_state, self._static_blocked - {goal}
                )
            return self._carry_dist_cache[goal].get(pos, 0x7FFF_FFFF)

        map_id = id(map_state)
        if map_id != self._dist_cache_map_id:
            self._dist_cache = {}
            self._dist_cache_map_id = map_id
        if goal not in self._dist_cache:
            self._dist_cache[goal] = _bfs_distances(goal, map_state)
        return self._dist_cache[goal].get(pos, 0x7FFF_FFFF)

    def plan(
        self,
        agent: "AgentState",
        goal: Tuple[int, int],
        world_state: "WorldState",
    ) -> List[Tuple[int, int]]:
        tick = world_state.tick

        # First plan() call of this tick → run bulk PIBT for ALL agents
        if tick != self._last_tick:
            self._goals[agent.agent_id] = goal
            self._last_tick = tick
            self._run_bulk_pibt(world_state)

        self._goals[agent.agent_id] = goal

        # Return the PIBT decision — even agents at their goal may have
        # been pushed aside by a higher-priority agent.
        # Always return a length-1 path (even for stay-in-place) so the
        # engine treats it as a successful plan, not a planning failure.
        next_pos = self._decided.get(agent.agent_id, agent.position)
        return [next_pos]

    # ------------------------------------------------------------------
    # Bulk PIBT pass
    # ------------------------------------------------------------------

    def _run_bulk_pibt(self, world_state: "WorldState") -> None:
        # Mix base_seed with tick so behavior varies per tick but is
        # reproducible across runs with the same configured seed.
        self._rng.seed((self._base_seed, world_state.tick))
        self._init_tick(world_state)

        # Process undecided agents in priority order (highest first)
        # Tie-break by agent_id for determinism
        active = sorted(
            list(self._undecided),
            key=lambda aid: (self._priorities.get(aid, 0), aid),
            reverse=True,
        )

        for aid in active:
            if aid in self._undecided:
                self._pibt(aid, None, world_state)

        # Update priorities based on movement outcome
        self._update_priorities(world_state)

    def _update_priorities(self, world_state: "WorldState") -> None:
        map_state = world_state.map_state
        for agent in world_state.agents:
            aid = agent.agent_id
            if agent.is_idle or agent.is_waiting:
                continue
            goal = self._get_goal(aid, world_state)
            next_pos = self._decided.get(aid, agent.position)
            if next_pos == goal:
                self._arrived[aid] = goal
            if self._arrived.get(aid) == goal:
                self._priorities[aid] = -1
                continue
            else:
                self._arrived.pop(aid, None)
            carrying = agent.carried_pod_id is not None
            dist_now = self._dist(agent.position, goal, map_state, carrying)
            dist_next = self._dist(next_pos, goal, map_state, carrying)
            if dist_next < dist_now:
                self._priorities[aid] = 0
            else:
                self._priorities[aid] = self._priorities.get(aid, 0) + 1

    # ------------------------------------------------------------------
    # Per-tick initialisation
    # ------------------------------------------------------------------

    def _init_tick(self, world_state: "WorldState") -> None:
        self._decided = {}
        self._next_occupied = {}
        self._undecided = set()
        self._agents = {}
        self._cur_pos_index = {}
        self._carry_dist_cache = {}

        for a in world_state.agents:
            self._agents[a.agent_id] = a
            if a.is_idle or a.is_waiting:
                self._decide(a.agent_id, a.position)
            else:
                self._undecided.add(a.agent_id)
                self._cur_pos_index.setdefault(a.position, []).append(
                    a.agent_id
                )

        # Static obstacles: non-carried pod positions
        self._static_blocked = set()
        for pod in world_state.pod_state.pods.values():
            if not pod.is_carried:
                self._static_blocked.add(pod.current_position)

    # ------------------------------------------------------------------
    # Decision bookkeeping
    # ------------------------------------------------------------------

    def _decide(self, agent_id: int, pos: Tuple[int, int]) -> None:
        """Mark *agent_id* as decided, claiming *pos* as its next position."""
        self._decided[agent_id] = pos
        self._next_occupied[pos] = agent_id
        self._undecided.discard(agent_id)
        agent = self._agents.get(agent_id)
        if agent and agent.position in self._cur_pos_index:
            ids = self._cur_pos_index[agent.position]
            try:
                ids.remove(agent_id)
            except ValueError:
                pass
            if not ids:
                del self._cur_pos_index[agent.position]

    def _undecide(self, agent_id: int) -> None:
        """Undo a tentative decision (backtracking)."""
        if agent_id not in self._decided:
            return
        pos = self._decided.pop(agent_id)
        if self._next_occupied.get(pos) == agent_id:
            del self._next_occupied[pos]
        self._undecided.add(agent_id)
        agent = self._agents[agent_id]
        self._cur_pos_index.setdefault(agent.position, []).append(agent_id)

    # ------------------------------------------------------------------
    # Goal lookup
    # ------------------------------------------------------------------

    def _get_goal(
        self, agent_id: int, world_state: "WorldState"
    ) -> Tuple[int, int]:
        task = world_state.task_state.get_active_task_for_agent(agent_id)
        if task is not None:
            return task.destination
        task = world_state.task_state.get_next_task_for_agent(agent_id)
        if task is not None:
            return task.destination
        if agent_id in self._goals:
            return self._goals[agent_id]
        return self._agents[agent_id].position

    # ------------------------------------------------------------------
    # Core PIBT recursion
    # ------------------------------------------------------------------

    def _pibt(
        self,
        agent_id: int,
        parent_id: Optional[int],
        world_state: "WorldState",
    ) -> bool:
        agent = self._agents[agent_id]
        goal = self._get_goal(agent_id, world_state)
        map_state = world_state.map_state

        candidates = list(map_state.get_neighbors(*agent.position))
        candidates.append(agent.position)  # stay in place

        # When carrying a pod, avoid non-carried pod positions
        if agent.carried_pod_id is not None:
            goal_set = {goal}
            candidates = [
                c for c in candidates
                if c not in self._static_blocked or c in goal_set
            ]

        # PIBT rule: don't move back to the parent's current position
        if parent_id is not None:
            parent_pos = self._agents[parent_id].position
            candidates = [c for c in candidates if c != parent_pos]

        # Greedy: prefer neighbours closer to goal (BFS distance).
        # Randomised tie-breaking avoids periodic oscillations.
        carrying = agent.carried_pod_id is not None
        self._rng.shuffle(candidates)
        candidates.sort(key=lambda p: self._dist(p, goal, map_state, carrying))

        for c in candidates:
            # Skip positions already claimed by decided agents
            if c in self._next_occupied:
                continue

            # Check if an undecided agent currently sits at c
            # (ignore self — happens when c is the agent's own position)
            occupier_id = self._undecided_at(c)
            if occupier_id == agent_id:
                occupier_id = None

            # Tentatively claim c
            self._decide(agent_id, c)

            if occupier_id is not None:
                # Priority inheritance: recursively push the occupier
                if not self._pibt(occupier_id, agent_id, world_state):
                    # Backtrack: push failed, undo and try next candidate
                    self._undecide(agent_id)
                    continue

            return True

        # All candidates exhausted — stay in place
        self._decide(agent_id, agent.position)
        return False

    def _undecided_at(self, pos: Tuple[int, int]) -> Optional[int]:
        """Return the id of an undecided agent currently at *pos*, or None."""
        ids = self._cur_pos_index.get(pos)
        if ids:
            return ids[0]
        return None
