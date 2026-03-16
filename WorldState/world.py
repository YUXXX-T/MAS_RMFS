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
        for i in range(config.robots.num_robots):
            start = config.robots.starts[i] if i < len(config.robots.starts) else (0, 0)
            self.agents.append(AgentState(agent_id=i, start_position=start))

        # 初始化订单和任务容器
        self.order_state = OrderState()
        self.task_state = TaskState()

        # 从地图的 pod_home_positions 初始化货架
        self.pod_state = PodState()
        for idx, pos in enumerate(self.map_state.pod_home_positions):
            pod = Pod(pod_id=idx, home_position=pos)
            self.pod_state.add_pod(pod)

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
