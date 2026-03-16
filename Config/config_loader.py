"""
Config Loader Module
====================
Loads simulation configuration from a JSON file and provides it as typed dataclasses.

配置加载模块
====================
从 JSON 文件加载仿真配置，并将其提供为类型化的数据类。

"""

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class StationConfig:
    """Configuration for a single workstation."""
    id: int
    row: int
    col: int


@dataclass
class PodZoneConfig:
    """Configuration for a rectangular block of pod storage positions."""
    origin_row: int
    origin_col: int
    num_rows: int
    num_cols: int


@dataclass
class MapConfig:
    """Configuration for the warehouse map/grid."""
    rows: int
    cols: int
    obstacles: List[Tuple[int, int]]
    stations: List[StationConfig]
    pod_zones: List[PodZoneConfig]


@dataclass
class RobotConfig:
    """Configuration for robot agents."""
    num_robots: int
    starts: List[Tuple[int, int]]
    speed: int = 1


@dataclass
class PolicyConfig:
    """
    Configuration for which algorithm to use for each policy.

    Each field stores a (name, params) tuple.  In JSON the value can be
    either a plain string (name only, no extra params) or an object
    ``{"name": "...", "params": {...}}``.

    用于指定各策略所使用算法的配置。 

    每个字段均存储一个 (name, params) 元组。在 JSON 格式中，其取值可以是
    一个纯字符串（仅包含名称，无额外参数），也可以是一个对象
    ``{"name": "...", "params": {...}}``。
    """
    order_generator: Tuple[str, Dict[str, Any]] = ("RandomOrderGenerator", {})
    task_assigner: Tuple[str, Dict[str, Any]] = ("GreedyTaskAssigner", {})
    path_planner: Tuple[str, Dict[str, Any]] = ("AStarPathPlanner", {})
    pod_return_planner: Tuple[str, Dict[str, Any]] = ("HomeReturnPlanner", {})


@dataclass
class SimulationParams:
    """仿真级参数。"""
    order_interval: int = 5           # Generate new order every N ticks
    max_items_per_order: int = 2      # Max pods per order
    pickup_duration: int = 2          # Ticks to pause when picking up a pod
    dropoff_duration: int = 2         # Ticks to pause when dropping off a pod
    station_process_duration: int = 5 # Ticks to process at workstation
    tick_delay: float = 0.0           # Seconds to sleep between ticks (0 = no delay)
    p3d_view_mode: str = "2d"         # Panda3D camera: "2d" (ortho) or "3d" (perspective)
    p3d_use_gpu: bool = False         # Enable GPU batching/instancing in Panda3D
    night_mode: bool = True           # True = dark theme, False = light/white theme
    log_level: str = "INFO"
    log_file: Optional[str] = None


@dataclass
class SimulationConfig:
    """Top-level simulation configuration aggregating all sub-configs."""
    map: MapConfig = field(default_factory=lambda: MapConfig(10, 10, [], [], []))
    robots: RobotConfig = field(default_factory=lambda: RobotConfig(1, [[0, 0]]))
    simulation: SimulationParams = field(default_factory=SimulationParams)
    policies: PolicyConfig = field(default_factory=PolicyConfig)


def load_config(path: str) -> SimulationConfig:
    """
    Load a SimulationConfig from a JSON file.

    参数
    ----------
    path : str
        JSON 配置文件路径。

    返回值
    -------
    SimulationConfig
        Parsed and validated configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # --- Parse map ---
    map_raw = raw.get("map", {})
    obstacles = [tuple(o) for o in map_raw.get("obstacles", [])]
    stations = [
        StationConfig(id=s["id"], row=s["row"], col=s["col"])
        for s in map_raw.get("stations", [])
    ]
    pod_zones = [
        PodZoneConfig(
            origin_row=pz["origin_row"],
            origin_col=pz["origin_col"],
            num_rows=pz["num_rows"],
            num_cols=pz["num_cols"],
        )
        for pz in map_raw.get("pod_zones", [])
    ]
    map_config = MapConfig(
        rows=map_raw.get("rows", 10),
        cols=map_raw.get("cols", 10),
        obstacles=obstacles,
        stations=stations,
        pod_zones=pod_zones,
    )

    # --- Parse robots ---
    robot_raw = raw.get("robots", {})
    starts = [tuple(s) for s in robot_raw.get("starts", [[0, 0]])]
    robot_config = RobotConfig(
        num_robots=robot_raw.get("num_robots", len(starts)),
        starts=starts,
        speed=robot_raw.get("speed", 1),
    )

    # --- Parse simulation params ---
    sim_raw = raw.get("simulation", {})
    sim_params = SimulationParams(
        order_interval=sim_raw.get("order_interval", 5),
        max_items_per_order=sim_raw.get("max_items_per_order", 2),
        pickup_duration=sim_raw.get("pickup_duration", 2),
        dropoff_duration=sim_raw.get("dropoff_duration", 2),
        station_process_duration=sim_raw.get("station_process_duration", 5),
        tick_delay=sim_raw.get("tick_delay", 0.0),
        p3d_view_mode=sim_raw.get("p3d_view_mode", "2d"),
        p3d_use_gpu=sim_raw.get("p3d_use_gpu", False),
        night_mode=sim_raw.get("night_mode", True),
        log_level=sim_raw.get("log_level", "INFO"),
        log_file=sim_raw.get("log_file", None),
    )

    # --- Parse policies ---
    pol_raw = raw.get("policies", {})

    def _parse_policy_entry(val, default_name: str):
        """Accept either a string or {"name": ..., "params": {...}}."""
        if val is None:
            return (default_name, {})
        if isinstance(val, str):
            return (val, {})
        return (val.get("name", default_name), val.get("params", {}))

    policy_config = PolicyConfig(
        order_generator=_parse_policy_entry(
            pol_raw.get("order_generator"), "RandomOrderGenerator"),
        task_assigner=_parse_policy_entry(
            pol_raw.get("task_assigner"), "GreedyTaskAssigner"),
        path_planner=_parse_policy_entry(
            pol_raw.get("path_planner"), "AStarPathPlanner"),
        pod_return_planner=_parse_policy_entry(
            pol_raw.get("pod_return_planner"), "HomeReturnPlanner"),
    )

    return SimulationConfig(
        map=map_config, robots=robot_config,
        simulation=sim_params, policies=policy_config,
    )
