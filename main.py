"""
MAS-RMFS：多智能体机器人移动履行系统仿真
======================================================================
仿真程序入口。

用法：
    python main.py                          # 使用默认配置
    python main.py --config path/to/cfg.json  # 使用自定义配置
    python main.py --visualize              # 启用终端可视化
    python main.py --mpl                    # 启用 matplotlib 仪表盘
"""

import argparse
import os
import sys
import time

from Config.config_loader import load_config, SimulationConfig
from Engine.simulation_engine import SimulationEngine
import Policies  # noqa: F401 — 触发算法自动注册
from Policies.policy_registry import get_policy
from Visualization.visualizer import TerminalVisualizer, MatplotlibVisualizer
from Debug.logger import SimLogger


def main():
    parser = argparse.ArgumentParser(
        description="MAS-RMFS：多智能体机器人移动履行系统仿真"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Config", "default_config.json"),
        help="JSON 配置文件路径。",
    )
    # default=os.path.join(os.path.dirname(__file__), "Config", "default_config.json"),
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--visualize",
        action="store_true",
        help="每个 tick 启用基于终端的 ASCII 可视化。",
    )
    viz_group.add_argument(
        "--mpl",
        action="store_true",
        help="启用 matplotlib 2×2 动态仪表盘。",
    )
    viz_group.add_argument(
        "--p3d",
        action="store_true",
        help="启用 Panda3D 2D 正交可视化。",
    )

    # --- 轨迹记录参数 ---
    parser.add_argument(
        "--record",
        action="store_true",
        help="启用轨迹记录，仿真结束后保存为可导入的数据文件。",
    )
    parser.add_argument(
        "--record-output",
        type=str,
        default="",
        help="轨迹数据输出路径（默认: TrajectoryRecord/trajectory_<timestamp>.traj.json.gz）。",
    )
    parser.add_argument(
        "--record-interval",
        type=int,
        default=1,
        help="轨迹采样间隔（tick 数），默认 1 表示每 tick 都记录。",
    )

    # --- 轨迹回放参数 ---
    parser.add_argument(
        "--replay",
        type=str,
        default="",
        help="加载轨迹数据文件并使用 Panda3D 回放（跳过仿真）。",
    )
    parser.add_argument(
        "--replay-fps",
        type=int,
        default=10,
        help="回放帧率，默认 10fps。范围 1-60。",
    )

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════
    # 回放模式：加载轨迹数据并直接启动 Panda3D 回放 UI
    # ═══════════════════════════════════════════════════════════════
    if args.replay:
        from TrajectoryRecord import TrajectoryData
        from Visualization.panda3d_visualizer import Panda3DVisualizer
        from Visualization.ui import ReplayUI

        logger = SimLogger("Main")
        logger.info(f"Loading trajectory: {args.replay}")
        data = TrajectoryData.load(args.replay)
        logger.info(f"  Map: {data.rows}x{data.cols}, Agents: {data.num_agents}, "
                     f"Frames: {data.total_ticks}")

        visualizer = Panda3DVisualizer(
            view_mode="2d",
            use_gpu=False,
            night_mode=True,
        )
        ui = ReplayUI(data=data, visualizer=visualizer, night_mode=True,
                      initial_fps=args.replay_fps)
        ui.run()
        return

    # ═══════════════════════════════════════════════════════════════
    # 正常仿真模式
    # ═══════════════════════════════════════════════════════════════

    # --- 加载配置 ---
    logger = SimLogger("Main")
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # --- 从配置实例化策略 ---
    og_name, og_params = config.policies.order_generator
    ta_name, ta_params = config.policies.task_assigner
    pp_name, pp_params = config.policies.path_planner
    rp_name, rp_params = config.policies.pod_return_planner
    pr_name, pr_params = config.policies.pod_retriever

    OrderGeneratorCls = get_policy("order_generator", og_name)
    TaskAssignerCls = get_policy("task_assigner", ta_name)
    PathPlannerCls = get_policy("path_planner", pp_name)
    PodReturnPlannerCls = get_policy("pod_return_planner", rp_name)
    PodRetrieverCls = get_policy("pod_retriever", pr_name)

    logger.info(f"Policies: order_generator={og_name}, "
                f"task_assigner={ta_name}, "
                f"path_planner={pp_name}, "
                f"pod_return_planner={rp_name}, "
                f"pod_retriever={pr_name}")

    order_generator = OrderGeneratorCls(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
        fixed_order_size=config.simulation.fixed_order_size,
        max_items_per_sku=config.simulation.max_items_per_sku,
        **og_params,
    )
    task_assigner = TaskAssignerCls(**ta_params)
    path_planner = PathPlannerCls(**pp_params)
    pod_return_planner = PodReturnPlannerCls(**rp_params)
    pod_retriever = PodRetrieverCls(**pr_params)

    # 将归还规划器和 Pod 检索器注入任务分配器
    task_assigner.pod_return_planner = pod_return_planner
    task_assigner.pod_retriever = pod_retriever

    # --- 可选的可视化器 ---
    if args.mpl:
        visualizer = MatplotlibVisualizer(
            night_mode=config.simulation.night_mode,
        )
    elif args.p3d:
        from Visualization.panda3d_visualizer import Panda3DVisualizer
        visualizer = Panda3DVisualizer(
            view_mode=config.simulation.p3d_view_mode,
            use_gpu=config.simulation.p3d_use_gpu,
            night_mode=config.simulation.night_mode,
        )
    elif args.visualize:
        visualizer = TerminalVisualizer()
    else:
        visualizer = None

    # --- 轨迹记录器 ---
    trajectory_recorder = None
    trajectory_output = ""
    if args.record:
        from TrajectoryRecord import TrajectoryRecorder

        trajectory_recorder = TrajectoryRecorder(sample_interval=args.record_interval)
        if args.record_output:
            trajectory_output = args.record_output
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            traj_dir = os.path.join(os.path.dirname(__file__), "TrajectoryRecord")
            os.makedirs(traj_dir, exist_ok=True)
            trajectory_output = os.path.join(traj_dir, f"trajectory_{ts}.traj.json.gz")
        logger.info(f"Trajectory recording enabled -> {trajectory_output}")

    # --- 创建引擎 ---
    engine = SimulationEngine(
        config=config,
        order_generator=order_generator,
        task_assigner=task_assigner,
        path_planner=path_planner,
        visualizer=visualizer,
        trajectory_recorder=trajectory_recorder,
        trajectory_output=trajectory_output,
    )

    # --- 运行 ---
    if args.p3d and visualizer is not None:
        # Qt UI 驱动循环（替代 engine.run）
        from Visualization.ui import SimulationUI
        ui = SimulationUI(
            engine=engine,
            visualizer=visualizer,
            night_mode=config.simulation.night_mode,
        )
        ui.run()
    else:
        engine.run()


if __name__ == "__main__":
    main()
