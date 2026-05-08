"""
MAS-RMFS：多智能体机器人移动履行系统仿真
======================================================================
仿真程序入口。

用法：
    python main.py                          # 使用默认配置
    python main.py --config path/to/cfg.json  # 使用自定义配置
    python main.py --visualize              # 启用终端可视化
    python main.py --mpl                    # 启用 matplotlib 仪表盘
    python main.py --record --max-ticks 500 # 记录轨迹，运行 500 ticks 后自动停止
"""

import argparse
import math
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
        default=os.path.join(os.path.dirname(__file__), "Config", "config_debug.json"),
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
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="最大运行 tick 数，到达后自动停止仿真。0 表示无限制（默认）。",
    )

    # --- 轨迹回放参数 ---
    parser.add_argument(
        "--replay",
        nargs='+',
        default=[],
        metavar="FILE",
        help="加载一个或多个轨迹数据文件并使用 Panda3D 回放（跳过仿真）。",
    )
    parser.add_argument(
        "--replay-fps",
        type=int,
        default=10,
        help="回放帧率，默认 10fps。范围 1-60。",
    )
    parser.add_argument(
        "--replay-layout",
        type=str,
        default="",
        help="多轨迹回放的网格布局，如 '2x1'、'1x2'、'2x2'。不指定则自动计算。",
    )

    args = parser.parse_args()

    # ═══════════════════════════════════════════════════════════════
    # 回放模式：加载轨迹数据并直接启动 Panda3D 回放 UI
    # ═══════════════════════════════════════════════════════════════
    if args.replay:
        from TrajectoryRecord import TrajectoryData
        from Config.config_loader import load_replay_config

        logger = SimLogger("Main")
        replay_config = load_replay_config(args.config)
        replay_files = args.replay
        n = len(replay_files)

        # 加载所有轨迹文件
        datasets = []
        labels = []
        for path in replay_files:
            logger.info(f"Loading trajectory: {path}")
            data = TrajectoryData.load(path)
            logger.info(f"  Map: {data.rows}x{data.cols}, Agents: {data.num_agents}, "
                         f"Frames: {data.total_ticks}")
            datasets.append(data)
            labels.append(os.path.basename(path))

        # 解析或自动计算布局（优先级：CLI > config > auto）
        def _auto_layout(count: int) -> tuple:
            if count == 1:
                return (1, 1)
            if count == 2:
                return (1, 2)
            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
            return (rows, cols)

        def _parse_layout(layout_str: str, source: str):
            parts = layout_str.lower().split('x')
            if len(parts) != 2:
                logger.error(f"Invalid layout format from {source}: '{layout_str}'. "
                             f"Expected 'RxC', e.g. '2x1'.")
                sys.exit(1)
            return (int(parts[0]), int(parts[1]))

        if args.replay_layout:
            layout = _parse_layout(args.replay_layout, "CLI")
        elif replay_config.layout != "auto":
            layout = _parse_layout(replay_config.layout, "config")
        else:
            layout = _auto_layout(n)

        if n == 1 and not args.replay_layout and replay_config.layout == "auto":
            # 单文件回放：使用原有的 ReplayUI（向后兼容）
            from Visualization.panda3d_visualizer import Panda3DVisualizer
            from Visualization.ui import ReplayUI

            visualizer = Panda3DVisualizer(
                view_mode="2d", use_gpu=False, night_mode=True,
            )
            ui = ReplayUI(data=datasets[0], visualizer=visualizer,
                          night_mode=True, initial_fps=args.replay_fps)
            ui.run()
        elif replay_config.mode == "list":
            # 列表模式
            from Visualization.ui import ListReplayUI

            logger.info(f"List-replay: files={n}")
            ui = ListReplayUI(
                datasets=datasets,
                labels=labels,
                night_mode=True,
                initial_fps=args.replay_fps,
            )
            ui.run()
        else:
            # 窗口模式（支持滚动：n 可以 > rows*cols）
            from Visualization.panda3d_visualizer import MultiPanda3DReplayVisualizer
            from Visualization.ui import MultiReplayUI

            logger.info(f"Multi-replay: layout={layout[0]}x{layout[1]}, "
                         f"files={n}")
            visualizer = MultiPanda3DReplayVisualizer(night_mode=True)
            ui = MultiReplayUI(
                datasets=datasets,
                labels=labels,
                layout=layout,
                visualizer=visualizer,
                night_mode=True,
                initial_fps=args.replay_fps,
            )
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
        max_ticks=args.max_ticks,
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
