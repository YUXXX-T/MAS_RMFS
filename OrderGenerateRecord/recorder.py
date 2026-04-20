"""
Order Recorder
==============
使用指定的 OrderGenerator 预生成一定数量的订单，
并将订单数据保存为 JSON 文件，供仿真回放使用。

用法：
    python -m OrderGenerateRecord.recorder                         # 默认参数
    python -m OrderGenerateRecord.recorder --order_amounts 200     # 生成 200 个订单
    python -m OrderGenerateRecord.recorder --generator RandomOrderGenerator
    python -m OrderGenerateRecord.recorder --output my_orders.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Config.config_loader import load_config
from WorldState.world import WorldState
import Policies  # noqa: F401 — 触发算法自动注册
from Policies.policy_registry import get_policy


def record_orders(
    config_path: str,
    generator_name: str | None = None,
    order_amounts: int = 100,
    output_path: str | None = None,
):
    config = load_config(config_path)

    og_name, og_params = config.policies.order_generator
    if generator_name is not None:
        og_name = generator_name
        og_params = {}

    OrderGeneratorCls = get_policy("order_generator", og_name)
    order_generator = OrderGeneratorCls(
        order_interval=config.simulation.order_interval,
        max_items_per_order=config.simulation.max_items_per_order,
        fixed_order_size=config.simulation.fixed_order_size,
        max_items_per_sku=config.simulation.max_items_per_sku,
        **og_params,
    )

    world = WorldState(config)

    recorded_orders = []
    tick = 0
    max_ticks = order_amounts * config.simulation.order_interval * 3

    while len(recorded_orders) < order_amounts and tick < max_ticks:
        world.tick = tick
        new_orders = order_generator.generate(world)
        for order in new_orders:
            recorded_orders.append({
                "tick": tick,
                "sku_demands": order.sku_demands,
                "station_id": order.station_id,
            })
        tick += 1

    recorded_orders = recorded_orders[:order_amounts]

    data = {
        "generator": og_name,
        "generator_params": og_params,
        "order_interval": config.simulation.order_interval,
        "total_orders": len(recorded_orders),
        "orders": recorded_orders,
    }

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "record", "orders.json"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Recorded {len(recorded_orders)} orders to {output_path}")
    print(f"  Generator: {og_name}")
    print(f"  Order interval: {config.simulation.order_interval}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="预生成订单并保存为 JSON 文件"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "..", "Config", "default_config.json"
        ),
        help="JSON 配置文件路径",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help="OrderGenerator 名称（如 RandomOrderGenerator, ZipfOrderGenerator）。"
             "不指定则使用配置文件中的设置。",
    )
    parser.add_argument(
        "--order_amounts",
        type=int,
        default=100,
        help="要生成的订单数量（默认 100）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径（默认 ./OrderGenerateRecord/record/orders.json）",
    )
    args = parser.parse_args()

    record_orders(
        config_path=args.config,
        generator_name=args.generator,
        order_amounts=args.order_amounts,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
