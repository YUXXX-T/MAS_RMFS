"""
世界状态模块
=================
聚合所有仿真状态组件的门面。
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
    聚合世界状态 — 仿真的唯一事实来源。

    属性
    ----------
    tick : int
        当前仿真 tick。
    map_state : MapState
        仓库网格地图。
    agents : list[AgentState]
        所有机器人智能体。
    order_state : OrderState
        所有客户订单。
    task_state : TaskState
        所有原子任务。
    pod_state : PodState
        所有货架（Pod）。
    config : SimulationConfig
        已加载的仿真配置。
    """

    def __init__(self, config: SimulationConfig):
        self.tick: int = 0
        self.config = config

        # 初始化地图
        self.map_state = MapState(config.map)

        # 在起始位置初始化智能体
        self.agents: List[AgentState] = []
        starts = self._resolve_agent_starts(config)
        for i in range(config.robots.num_robots):
            self.agents.append(AgentState(agent_id=i, start_position=starts[i]))

        # 初始化订单和任务容器
        self.order_state = OrderState()
        self.task_state = TaskState()

        # 通过 PodInitializer 策略初始化货架
        from Policies.policy_registry import get_policy
        self.pod_state = PodState()
        pi_name = config.policies.pod_initializer[0]
        pi_params = config.policies.pod_initializer[1] if len(config.policies.pod_initializer) > 1 else {}
        PodInitializerCls = get_policy("pod_initializer", pi_name)
        pod_initializer = PodInitializerCls(**pi_params)
        pod_initializer.initialize_pods(self)

    def _resolve_agent_starts(self, config):
        """Return a list of agent start positions of length num_robots.

        Two modes:
        - random_starts=False (default): use config.robots.starts in order;
          pad with (0, 0) for any agent beyond the list (legacy behavior).
        - random_starts=True: sample positions from FREE walkable cells
          (excluding stations and pod homes), using the GLOBAL random module —
          so seeding ``random`` upstream gives reproducible starts.
        """
        import random as _random
        from WorldState.map_state import CellType

        n = config.robots.num_robots
        if not config.robots.random_starts:
            starts = list(config.robots.starts)
            if len(starts) < n:
                starts.extend([(0, 0)] * (n - len(starts)))
            return starts[:n]

        m = self.map_state
        free_cells = [
            (r, c)
            for r in range(m.rows)
            for c in range(m.cols)
            if m.grid[r][c] == CellType.FREE
        ]
        if len(free_cells) < n:
            raise ValueError(
                f"random_starts=True but only {len(free_cells)} free cells "
                f"available for {n} robots."
            )
        return _random.sample(free_cells, n)

    def advance_tick(self):
        """递增仿真 tick 计数器。"""
        self.tick += 1

    def get_agent(self, agent_id: int) -> AgentState:
        """按 ID 获取智能体。"""
        return self.agents[agent_id]

    def get_idle_agents(self) -> List[AgentState]:
        """返回所有当前空闲的智能体。"""
        return [a for a in self.agents if a.is_idle]

    def __repr__(self) -> str:
        return (
            f"WorldState(tick={self.tick}, agents={len(self.agents)}, "
            f"pods={self.pod_state.total_pods}, "
            f"orders={self.order_state.total_orders})"
        )
