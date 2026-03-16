"""
MAS-RMFS: Multi-Agent Simulation for Robotic Mobile Fulfillment System
======================================================================
Entry point for the simulation.

Usage:
    python main.py                          # Use default config
    python main.py --config path/to/cfg.json  # Use custom config
    python main.py --visualize              # Enable terminal visualization
    python main.py --mpl                    # Enable matplotlib dashboard
"""

import argparse
import os
import sys

from Config.config_loader import load_config, SimulationConfig
from Engine.simulation_engine import SimulationEngine
import Policies  # noqa: F401  — triggers algorithm self-registration
from Policies.policy_registry import get_policy
from Visualization.visualizer import TerminalVisualizer, MatplotlibVisualizer
from Debug.logger import SimLogger


def main():
    parser = argparse.ArgumentParser(
        description="MAS-RMFS: Multi-Agent Simulation for Robotic Mobile Fulfillment System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Config", "default_config.json"),
        help="Path to the JSON configuration file.",
    )

    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument(
        "--visualize",
        action="store_true",
        help="Enable terminal-based ASCII visualization each tick.",
    )
    viz_group.add_argument(
        "--mpl",
        action="store_true",
        help="Enable animated matplotlib 2×2 dashboard.",
    )

    args = parser.parse_args()

    # --- Load configuration ---
    logger = SimLogger("Main")
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # --- Instantiate policies from config ---
    og_name, og_params = config.policies.order_generator
    ta_name, ta_params = config.policies.task_assigner
    pp_name, pp_params = config.policies.path_planner

    OrderGeneratorCls = get_policy("order_generator", og_name)
    TaskAssignerCls = get_policy("task_assigner", ta_name)
    PathPlannerCls = get_policy("path_planner", pp_name)

    logger.info(f"Policies: order_generator={og_name}, "
                f"task_assigner={ta_name}, "
                f"path_planner={pp_name}")

    order_generator = OrderGeneratorCls(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
        **og_params,
    )
    task_assigner = TaskAssignerCls(**ta_params)
    path_planner = PathPlannerCls(**pp_params)

    # --- Optional visualizer ---
    if args.mpl:
        visualizer = MatplotlibVisualizer()
    elif args.visualize:
        visualizer = TerminalVisualizer()
    else:
        visualizer = None

    # --- Create and run engine ---
    engine = SimulationEngine(
        config=config,
        order_generator=order_generator,
        task_assigner=task_assigner,
        path_planner=path_planner,
        visualizer=visualizer,
    )
    engine.run()


if __name__ == "__main__":
    main()
