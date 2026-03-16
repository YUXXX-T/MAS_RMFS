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

    args = parser.parse_args()

    # --- 加载配置 ---
    logger = SimLogger("Main")
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # --- 从配置实例化策略 ---
    og_name, og_params = config.policies.order_generator
    ta_name, ta_params = config.policies.task_assigner
    pp_name, pp_params = config.policies.path_planner
    rp_name, rp_params = config.policies.pod_return_planner

    OrderGeneratorCls = get_policy("order_generator", og_name)
    TaskAssignerCls = get_policy("task_assigner", ta_name)
    PathPlannerCls = get_policy("path_planner", pp_name)
    PodReturnPlannerCls = get_policy("pod_return_planner", rp_name)

    logger.info(f"Policies: order_generator={og_name}, "
                f"task_assigner={ta_name}, "
                f"path_planner={pp_name}, "
                f"pod_return_planner={rp_name}")

    order_generator = OrderGeneratorCls(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
        **og_params,
    )
    task_assigner = TaskAssignerCls(**ta_params)
    path_planner = PathPlannerCls(**pp_params)
    pod_return_planner = PodReturnPlannerCls(**rp_params)

    # 将归还规划器注入任务分配器
    task_assigner.pod_return_planner = pod_return_planner

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

    # --- 创建引擎 ---
    engine = SimulationEngine(
        config=config,
        order_generator=order_generator,
        task_assigner=task_assigner,
        path_planner=path_planner,
        visualizer=visualizer,
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
