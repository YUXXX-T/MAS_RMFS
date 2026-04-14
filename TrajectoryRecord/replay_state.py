"""
Replay State Adapter
=====================
轻量级适配器，将 TrajectoryData 包装为 WorldState 兼容的接口，
使 Panda3D 可视化器无需修改即可用于轨迹回放。

Lightweight adapter wrapping TrajectoryData into a WorldState-compatible
interface so the Panda3D visualizer can replay trajectories without
modification.
"""

from typing import List, Dict, Tuple, Optional

from TrajectoryRecord.trajectory_recorder import TrajectoryData
from WorldState.map_state import CellType


# ── 轻量级 Mock 类 ──────────────────────────────────────────────────


class _MockMapState:
    """模拟 MapState 的最小接口。"""

    def __init__(self, data: TrajectoryData):
        self.rows = data.rows
        self.cols = data.cols
        self.station_positions: Dict[int, Tuple[int, int]] = dict(data.stations)
        self.pod_home_positions: List[Tuple[int, int]] = list(data.pod_homes)

        # 构建网格
        self.grid: List[List[CellType]] = [
            [CellType.FREE for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        for r, c in data.obstacles:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r][c] = CellType.OBSTACLE
        for sid, (sr, sc) in data.stations.items():
            if 0 <= sr < self.rows and 0 <= sc < self.cols:
                self.grid[sr][sc] = CellType.STATION
        for r, c in data.pod_homes:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                if self.grid[r][c] == CellType.FREE:
                    self.grid[r][c] = CellType.POD_HOME


class _MockPod:
    """模拟 Pod 的最小接口。"""

    def __init__(self, pod_id: int, home_position: Tuple[int, int]):
        self.pod_id = pod_id
        self.home_position = home_position
        self.current_position: Tuple[int, int] = home_position
        self.is_carried: bool = False
        self.carried_by: Optional[int] = None

    @property
    def is_at_home(self) -> bool:
        return self.current_position == self.home_position


class _MockPodState:
    """模拟 PodState 的最小接口。"""

    def __init__(self, pod_homes: List[Tuple[int, int]]):
        self.pods: Dict[int, _MockPod] = {}
        for i, pos in enumerate(pod_homes):
            self.pods[i] = _MockPod(pod_id=i, home_position=pos)

    @property
    def total_pods(self) -> int:
        return len(self.pods)


class _MockAgent:
    """模拟 AgentState 的最小接口。"""

    def __init__(self, agent_id: int, position: Tuple[int, int]):
        self.agent_id = agent_id
        self.position = position
        self.carried_pod_id: Optional[int] = None
        self.status = _MockStatus("IDLE")


class _MockStatus:
    """模拟 AgentStatus 的最小接口。"""

    def __init__(self, name: str):
        self.name = name


class _MockOrderState:
    """模拟 OrderState 的最小接口。"""

    def __init__(self):
        self.total_completed = 0
        self.total_orders = 0


# ── 主适配器 ────────────────────────────────────────────────────


class ReplayWorldState:
    """
    将 TrajectoryData 适配为 WorldState 接口，用于驱动 Panda3D 可视化器。

    用法
    -----
    replay = ReplayWorldState(trajectory_data)
    visualizer._setup(replay)       # 初始化 Panda3D 场景

    replay.set_frame(42)            # 跳到第 42 帧
    visualizer._update_agents(replay)
    visualizer._update_pods(replay)
    visualizer._update_hud(replay)
    """

    def __init__(self, data: TrajectoryData):
        self._data = data
        self._current_frame = 0

        # 静态组件
        self.map_state = _MockMapState(data)
        self.pod_state = _MockPodState(data.pod_homes)
        self.order_state = _MockOrderState()

        # 初始化 agents（从第一帧）
        self.agents: List[_MockAgent] = []
        if data.frames:
            first = data.frames[0]
            for agent_info in first["agents"]:
                self.agents.append(_MockAgent(
                    agent_id=agent_info["id"],
                    position=tuple(agent_info["pos"]),
                ))
        self.tick = 0

    @property
    def total_frames(self) -> int:
        return len(self._data.frames)

    @property
    def current_frame_index(self) -> int:
        return self._current_frame

    def set_frame(self, frame_index: int):
        """跳到指定帧，更新所有 agent 和 pod 状态。"""
        frame_index = max(0, min(frame_index, len(self._data.frames) - 1))
        self._current_frame = frame_index
        frame = self._data.frames[frame_index]
        self.tick = frame["tick"]

        # 收集当前帧中被携带的 pod ID
        carried_pods = set()

        for agent_info in frame["agents"]:
            aid = agent_info["id"]
            if aid < len(self.agents):
                agent = self.agents[aid]
                agent.position = tuple(agent_info["pos"])
                agent.carried_pod_id = agent_info.get("pod")
                agent.status = _MockStatus(agent_info.get("status", "IDLE"))
                if agent.carried_pod_id is not None:
                    carried_pods.add(agent.carried_pod_id)

        # 更新 pod 状态：被携带的隐藏，其余回到 home
        for pod in self.pod_state.pods.values():
            if pod.pod_id in carried_pods:
                pod.is_carried = True
            else:
                pod.is_carried = False
                pod.current_position = pod.home_position
