"""
Simplified MAPF Runner
======================
Lightweight one-shot MAPF benchmark runner.
No pods, no orders, no task chains — pure start→goal navigation.
Reuses the existing MapState and PathPlanner interfaces.
"""

from typing import List, Dict, Tuple

from WorldState.map_state import MapState
from WorldState.agent_state import AgentState, AgentStatus
from WorldState.order_state import OrderState
from WorldState.task_state import TaskState
from WorldState.pod_state import PodState
from WorldState.world import WorldState
from Policies.PathPlanner.base_path_planner import BasePathPlanner
from Debug.logger import SimLogger


def _build_minimal_world(map_state: MapState, agents: List[AgentState]) -> WorldState:
    """Construct a WorldState with only map + agents (no pods/orders)."""
    world = object.__new__(WorldState)
    world.tick = 0
    world.config = None
    world.map_state = map_state
    world.agents = agents
    world.order_state = OrderState()
    world.task_state = TaskState()
    world.pod_state = PodState()
    return world


class MAPFRunner:
    """Run a one-shot MAPF instance and collect metrics.

    Parameters
    ----------
    map_state : MapState
        The grid map (from MovingAILoader or RMFS config).
    agents_with_goals : list[dict]
        Each entry: {"start": (row, col), "goal": (row, col)}.
    path_planner : BasePathPlanner
        Any registered path planner instance.
    max_ticks : int
        Timeout — abort if not all agents arrive within this many ticks.
    """

    def __init__(
        self,
        map_state: MapState,
        agents_with_goals: List[dict],
        path_planner: BasePathPlanner,
        max_ticks: int = 500,
    ):
        self.map_state = map_state
        self.goals: Dict[int, Tuple[int, int]] = {}
        self.path_planner = path_planner
        self.max_ticks = max_ticks
        self.logger = SimLogger("MAPFRunner", level="WARNING")

        self.agents: List[AgentState] = []
        for i, ag in enumerate(agents_with_goals):
            agent = AgentState(agent_id=i, start_position=ag["start"])
            agent.status = AgentStatus.MOVING
            self.agents.append(agent)
            self.goals[i] = ag["goal"]

        self.world = _build_minimal_world(map_state, self.agents)
        self.snapshot_collector = None

    def run(self) -> dict:
        """Execute the MAPF instance and return result metrics."""
        # Pre-populate goals for per-step planners (e.g. PIBT) so that
        # pushed agents know their destinations from the very first tick.
        if hasattr(self.path_planner, "set_goals"):
            self.path_planner.set_goals(dict(self.goals))

        total_conflicts = 0
        tick = 0

        while tick < self.max_ticks:
            # Plan / replan for ALL agents without a path.
            # Per-step planners (e.g. PIBT) return length-1 paths and
            # re-plan every tick.  Agents at their goals call plan() too
            # so PIBT can push them out of the way when needed.
            for agent in self.agents:
                if not agent.has_path:
                    path = self.path_planner.plan(
                        agent, self.goals[agent.agent_id], self.world
                    )
                    if path:
                        agent.assign_path(path)

            prev_positions = {a.agent_id: a.position for a in self.agents}

            for agent in self.agents:
                if agent.has_path:
                    agent.advance()

            total_conflicts += self._count_conflicts(prev_positions)

            if self.snapshot_collector is not None:
                self.snapshot_collector.record_tick(self.world)

            tick += 1
            self.world.tick = tick

            if all(
                a.position == self.goals[a.agent_id] for a in self.agents
            ):
                break

        n = len(self.agents)
        completed = sum(
            1 for a in self.agents if a.position == self.goals[a.agent_id]
        )
        return {
            "makespan": tick,
            "success_rate": completed / n if n else 0.0,
            "completed_agents": completed,
            "total_agents": n,
            "total_conflicts": total_conflicts,
            "ticks_used": tick,
        }

    def _count_conflicts(self, prev_positions: dict) -> int:
        """Count vertex and swap conflicts for this tick."""
        count = 0

        # Vertex conflicts
        pos_to_agents: dict[tuple, list] = {}
        for agent in self.agents:
            pos_to_agents.setdefault(agent.position, []).append(agent.agent_id)
        for pos, aids in pos_to_agents.items():
            if len(aids) > 1:
                count += 1

        # Swap (oncoming) conflicts
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                a, b = self.agents[i], self.agents[j]
                a_prev = prev_positions[a.agent_id]
                b_prev = prev_positions[b.agent_id]
                if (
                    a.position == b_prev
                    and b.position == a_prev
                    and a_prev != a.position
                ):
                    count += 1

        return count
