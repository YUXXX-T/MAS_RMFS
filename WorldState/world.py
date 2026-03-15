"""
WorldState Module
=================
Facade aggregating all simulation state components.
"""

from typing import List, Tuple

from Config.config_loader import SimulationConfig
from WorldState.map_state import MapState
from WorldState.agent_state import AgentState
from WorldState.order_state import OrderState
from WorldState.task_state import TaskState
from WorldState.pod_state import PodState, Pod


class WorldState:
    """
    Aggregate world state — the single source of truth for the simulation.

    Attributes
    ----------
    tick : int
        Current simulation tick.
    map_state : MapState
        The warehouse grid map.
    agents : list[AgentState]
        All robot agents.
    order_state : OrderState
        All customer orders.
    task_state : TaskState
        All atomic tasks.
    pod_state : PodState
        All pods (shelves).
    config : SimulationConfig
        The loaded simulation configuration.
    """

    def __init__(self, config: SimulationConfig):
        self.tick: int = 0
        self.config = config

        # Initialize map
        self.map_state = MapState(config.map)

        # Initialize agents at their start positions
        self.agents: List[AgentState] = []
        for i in range(config.robots.num_robots):
            start = config.robots.starts[i] if i < len(config.robots.starts) else (0, 0)
            self.agents.append(AgentState(agent_id=i, start_position=start))

        # Initialize order and task containers
        self.order_state = OrderState()
        self.task_state = TaskState()

        # Initialize pods from map pod_home_positions
        self.pod_state = PodState()
        for idx, pos in enumerate(self.map_state.pod_home_positions):
            pod = Pod(pod_id=idx, home_position=pos)
            self.pod_state.add_pod(pod)

    def advance_tick(self):
        """Increment the simulation tick counter."""
        self.tick += 1

    def get_agent(self, agent_id: int) -> AgentState:
        """Get an agent by ID."""
        return self.agents[agent_id]

    def get_idle_agents(self) -> List[AgentState]:
        """Return all agents that are currently idle."""
        return [a for a in self.agents if a.is_idle]

    def __repr__(self) -> str:
        return (
            f"WorldState(tick={self.tick}, agents={len(self.agents)}, "
            f"pods={self.pod_state.total_pods}, "
            f"orders={self.order_state.total_orders})"
        )
