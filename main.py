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
from Policies.order_generator import RandomOrderGenerator
from Policies.task_assigner import GreedyTaskAssigner
from Policies.path_planner import AStarPathPlanner
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

    # --- Instantiate policies ---
    order_generator = RandomOrderGenerator(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
    )
    task_assigner = GreedyTaskAssigner()
    path_planner = AStarPathPlanner()

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
