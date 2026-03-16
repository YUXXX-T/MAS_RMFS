from .base_order_generator import BaseOrderGenerator
from .RandomOrderGenerator import RandomOrderGenerator
from .ZipfOrderGenerator import ZipfOrderGenerator

from Policies.policy_registry import register
register("order_generator", "RandomOrderGenerator", RandomOrderGenerator)
register("order_generator", "ZipfOrderGenerator", ZipfOrderGenerator)

__all__ = ["BaseOrderGenerator", "RandomOrderGenerator", "ZipfOrderGenerator"]
