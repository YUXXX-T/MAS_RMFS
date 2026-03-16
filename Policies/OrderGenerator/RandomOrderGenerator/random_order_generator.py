"""
Random Order Generator
======================
Default implementation: generates random orders at a configurable interval.
"""

import random
from typing import List

from Policies.OrderGenerator.base_order_generator import BaseOrderGenerator
from WorldState.order_state import Order


class RandomOrderGenerator(BaseOrderGenerator):
    """
    Generates random orders at regular intervals.

    Each order picks 1..max_items random available pods and a random station.

    Parameters
    ----------
    order_interval : int
        Generate a new order every N ticks.
    max_items_per_order : int
        Maximum number of pods per order.
    """

    def __init__(self, order_interval: int = 5, max_items_per_order: int = 2):
        self.order_interval = order_interval
        self.max_items_per_order = max_items_per_order

    def generate(self, world_state) -> List[Order]:
        """Generate random orders every `order_interval` ticks."""
        orders = []

        # Only generate on the correct interval ticks (and not on tick 0)
        if world_state.tick == 0 or world_state.tick % self.order_interval != 0:
            return orders

        # Get available pods (at home and not carried)
        available_pods = world_state.pod_state.get_available_pods()
        if not available_pods:
            return orders

        # Get station IDs
        station_ids = list(world_state.map_state.station_positions.keys())
        if not station_ids:
            return orders

        # Pick random pods for this order
        num_items = min(
            random.randint(1, self.max_items_per_order),
            len(available_pods),
        )
        chosen_pods = random.sample(available_pods, num_items)
        chosen_pod_ids = [p.pod_id for p in chosen_pods]

        # Pick a random station
        station_id = random.choice(station_ids)

        order = Order(
            pod_ids=chosen_pod_ids,
            station_id=station_id,
            created_at=world_state.tick,
        )
        orders.append(order)

        return orders
