"""
Trajectory Recorder Module
===========================
在仿真运行期间记录所有机器人的运动轨迹，仿真结束后保存为可导入的数据文件。

Record all robot trajectories during simulation. After simulation ends,
save the data as a portable file that can be loaded independently for
visualization and analysis.

用法:
    # 在仿真中使用
    recorder = TrajectoryRecorder()
    # 每 tick 调用
    recorder.snapshot(world_state)
    # 仿真结束后
    recorder.save("output.traj.json.gz")

    # 独立加载
    data = TrajectoryData.load("output.traj.json.gz")
    print(data.num_agents, data.total_ticks)
"""

import gzip
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState

from Debug.logger import SimLogger


@dataclass
class TrajectoryData:
    """
    轨迹数据容器，包含地图元信息和逐 tick 的机器人状态帧。

    Portable trajectory data container with map metadata and per-tick
    agent state frames. This object is fully self-contained and does not
    depend on any simulation runtime.

    属性
    ----------
    rows : int
        地图行数。
    cols : int
        地图列数。
    num_agents : int
        机器人总数。
    total_ticks : int
        记录的总 tick 数。
    obstacles : list[tuple[int, int]]
        障碍物坐标列表。
    stations : dict[int, tuple[int, int]]
        工作站 {id: (row, col)}。
    pod_homes : list[tuple[int, int]]
        货架原始位置列表。
    frames : list[dict]
        逐 tick 帧数据。每帧包含:
        - tick: int
        - agents: list[{id, pos, status, pod}]
    """
    rows: int = 0
    cols: int = 0
    num_agents: int = 0
    total_ticks: int = 0
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    stations: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    pod_homes: List[Tuple[int, int]] = field(default_factory=list)
    frames: List[Dict[str, Any]] = field(default_factory=list)
    record_time: str = ""

    # ── 序列化 ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """将轨迹数据序列化为纯 Python 字典。"""
        return {
            "meta": {
                "rows": self.rows,
                "cols": self.cols,
                "num_agents": self.num_agents,
                "total_ticks": self.total_ticks,
                "obstacles": [list(o) for o in self.obstacles],
                "stations": {str(k): list(v) for k, v in self.stations.items()},
                "pod_homes": [list(p) for p in self.pod_homes],
                "record_time": self.record_time,
            },
            "frames": self.frames,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryData":
        """从字典反序列化。"""
        meta = d["meta"]
        return cls(
            rows=meta["rows"],
            cols=meta["cols"],
            num_agents=meta["num_agents"],
            total_ticks=meta["total_ticks"],
            obstacles=[tuple(o) for o in meta.get("obstacles", [])],
            stations={int(k): tuple(v) for k, v in meta.get("stations", {}).items()},
            pod_homes=[tuple(p) for p in meta.get("pod_homes", [])],
            frames=d.get("frames", []),
            record_time=meta.get("record_time", ""),
        )

    # ── 文件读写 ────────────────────────────────────────────────────

    def save(self, filepath: str):
        """
        保存轨迹数据到 gzip 压缩的 JSON 文件。

        参数
        ----------
        filepath : str
            输出文件路径，建议以 .traj.json.gz 结尾。
        """
        raw = json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with gzip.open(filepath, "wt", encoding="utf-8") as f:
            f.write(raw)

    @classmethod
    def load(cls, filepath: str) -> "TrajectoryData":
        """
        从 gzip 压缩的 JSON 文件加载轨迹数据。

        参数
        ----------
        filepath : str
            轨迹数据文件路径。

        返回值
        -------
        TrajectoryData
            加载的轨迹数据。
        """
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    # ── 便捷查询 ────────────────────────────────────────────────────

    def get_agent_trajectory(self, agent_id: int) -> List[Tuple[int, int]]:
        """
        获取单个机器人的完整位置轨迹。

        返回值
        -------
        list[tuple[int, int]]
            按 tick 排序的 (row, col) 坐标列表。
        """
        trajectory = []
        for frame in self.frames:
            for agent in frame["agents"]:
                if agent["id"] == agent_id:
                    trajectory.append(tuple(agent["pos"]))
                    break
        return trajectory

    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        获取所有机器人的完整位置轨迹。

        返回值
        -------
        dict[int, list[tuple[int, int]]]
            {agent_id: [(row, col), ...]}
        """
        trajs: Dict[int, List[Tuple[int, int]]] = {}
        for frame in self.frames:
            for agent in frame["agents"]:
                aid = agent["id"]
                if aid not in trajs:
                    trajs[aid] = []
                trajs[aid].append(tuple(agent["pos"]))
        return trajs

    def get_frame(self, tick: int) -> Optional[Dict]:
        """获取指定 tick 的帧数据。"""
        if 0 <= tick < len(self.frames):
            return self.frames[tick]
        return None


class TrajectoryRecorder:
    """
    仿真轨迹记录器。

    在仿真运行期间每 tick 调用 `snapshot()` 以记录所有机器人状态。
    仿真结束后调用 `save()` 将数据保存到文件。

    参数
    ----------
    sample_interval : int
        采样间隔（tick 数）。默认为 1（每 tick 都记录）。
        设为 N > 1 时，每 N 个 tick 记录一次。
    """

    def __init__(self, sample_interval: int = 1):
        self._interval = max(1, sample_interval)
        self._data = TrajectoryData()
        self._meta_captured = False
        self._tick_count = 0
        self._logger = SimLogger("TrajectoryRecorder")

    @property
    def data(self) -> TrajectoryData:
        """获取当前已记录的轨迹数据。"""
        return self._data

    def snapshot(self, world_state: "WorldState"):
        """
        记录当前 tick 的所有机器人状态。

        每个 tick 由仿真引擎在 `_tick()` 末尾调用。

        参数
        ----------
        world_state : WorldState
            当前世界状态。
        """
        # 首次调用时捕获地图元信息
        if not self._meta_captured:
            self._capture_meta(world_state)

        tick = world_state.tick

        # 采样间隔过滤
        if tick % self._interval != 0:
            return

        # 记录帧
        agents = []
        for agent in world_state.agents:
            agents.append({
                "id": agent.agent_id,
                "pos": list(agent.position),
                "status": agent.status.name,
                "pod": agent.carried_pod_id,
            })

        self._data.frames.append({
            "tick": tick,
            "agents": agents,
        })
        self._tick_count += 1

    def save(self, filepath: str):
        """
        保存轨迹数据到文件。

        参数
        ----------
        filepath : str
            输出文件路径。
        """
        self._data.total_ticks = self._tick_count
        self._data.record_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self._data.save(filepath)
        self._logger.info(
            f"Trajectory saved: {filepath} "
            f"({self._tick_count} frames, {self._data.num_agents} agents)"
        )

    def _capture_meta(self, world_state: "WorldState"):
        """首次调用 snapshot 时捕获地图元信息。"""
        from WorldState.map_state import CellType

        ms = world_state.map_state
        self._data.rows = ms.rows
        self._data.cols = ms.cols
        self._data.num_agents = len(world_state.agents)

        # 障碍物
        self._data.obstacles = []
        for r in range(ms.rows):
            for c in range(ms.cols):
                if ms.grid[r][c] == CellType.OBSTACLE:
                    self._data.obstacles.append((r, c))

        # 工作站
        self._data.stations = dict(ms.station_positions)

        # Pod Home 位置
        self._data.pod_homes = list(ms.pod_home_positions)

        self._meta_captured = True
        self._logger.info(
            f"Map metadata captured: {ms.rows}x{ms.cols}, "
            f"{self._data.num_agents} agents, "
            f"{len(self._data.stations)} stations, "
            f"{len(self._data.pod_homes)} pod homes"
        )
