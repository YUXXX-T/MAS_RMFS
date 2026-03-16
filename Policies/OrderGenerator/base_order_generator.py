"""
Base Order Generator
====================
Abstract base class (interface) for order generation policies.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.order_state import Order


class BaseOrderGenerator(ABC):
    """
    Interface for order generation policies.

    Implementations decide when and how to create new customer orders.
    """

    @abstractmethod
    def generate(self, world_state: "WorldState") -> List["Order"]:
        """
        Generate new orders based on the current world state.

        Parameters
        ----------
        world_state : WorldState
            Current simulation state including tick counter.

        Returns
        -------
        list[Order]
            Newly generated orders (may be empty).
        """
        ...
