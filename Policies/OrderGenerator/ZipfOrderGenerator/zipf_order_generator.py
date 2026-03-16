"""
Zipf Order Generator
====================
Generates orders using a Zipf (power-law) distribution over pods.

This models real-world warehouse behavior where a small number of
popular products (pods) account for a disproportionately large share
of orders — the "80/20 rule".
"""

import random
import numpy as np
from typing import List

from Policies.OrderGenerator.base_order_generator import BaseOrderGenerator
from WorldState.order_state import Order


class ZipfOrderGenerator(BaseOrderGenerator):
    """
    Generates orders at regular intervals, selecting pods according
    to a Zipf (power-law) distribution.

    Pods are ranked by their pod_id.  Lower-ranked pods are ordered
    exponentially more often than higher-ranked ones, controlled
    by the ``zipf_param`` exponent.

    参数
    ----------
    order_interval : int
        Generate a new order every N ticks.
    max_items_per_order : int
        Maximum number of pods per order.
    zipf_param : float
        Zipf exponent (a > 1).  Higher values make the distribution
        more skewed toward popular pods.  Typical values: 1.2–2.0.
        Default is 1.5.
    """

    def __init__(
        self,
        order_interval: int = 5,
        max_items_per_order: int = 2,
        zipf_param: float = 1.5,
    ):
        self.order_interval = order_interval
        self.max_items_per_order = max_items_per_order
        self.zipf_param = zipf_param

    def _zipf_weights(self, n: int) -> np.ndarray:
        """
        Compute Zipf probability weights for *n* items.

        返回值
        -------
        np.ndarray
            Normalised probability array of length *n*.
        """
        ranks = np.arange(1, n + 1, dtype=float)
        weights = 1.0 / np.power(ranks, self.zipf_param)
        return weights / weights.sum()

    def generate(self, world_state) -> List[Order]:
        """Generate orders with pod selection biased by Zipf distribution."""
        orders: List[Order] = []

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

        # Number of items for this order
        num_items = min(
            random.randint(1, self.max_items_per_order),
            len(available_pods),
        )

        # Compute Zipf weights over available pods
        weights = self._zipf_weights(len(available_pods))

        # Sample pods without replacement using Zipf probabilities
        indices = np.random.choice(
            len(available_pods),
            size=num_items,
            replace=False,
            p=weights,
        )
        chosen_pod_ids = [available_pods[i].pod_id for i in indices]

        # Pick a random station
        station_id = random.choice(station_ids)

        order = Order(
            pod_ids=chosen_pod_ids,
            station_id=station_id,
            created_at=world_state.tick,
        )
        orders.append(order)

        return orders
