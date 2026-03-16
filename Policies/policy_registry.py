"""
Policy Registry
===============
Central registry mapping algorithm names (strings) to their classes.

Each algorithm self-registers when its subpackage is imported.
The simulation reads the desired algorithm name from the JSON config
and uses this registry to instantiate the correct class — no source
code changes required to swap algorithms.

Usage
-----
Registering (done inside each algorithm's __init__.py)::

    from Policies.policy_registry import register
    register("order_generator", "RandomOrderGenerator", RandomOrderGenerator)

Looking up (done in main.py)::

    from Policies.policy_registry import get_policy
    cls = get_policy("order_generator", "RandomOrderGenerator")
    instance = cls(order_interval=5)
"""

from typing import Dict, Type

# category -> {name -> class}
_registry: Dict[str, Dict[str, Type]] = {}


def register(category: str, name: str, cls: Type) -> None:
    """
    Register a policy class under the given category and name.

    Parameters
    ----------
    category : str
        One of "order_generator", "task_assigner", "path_planner".
    name : str
        Human-readable algorithm name (e.g. "AStarPathPlanner").
    cls : type
        The concrete class implementing the algorithm.
    """
    _registry.setdefault(category, {})[name] = cls


def get_policy(category: str, name: str) -> Type:
    """
    Look up a registered policy class by category and name.

    Parameters
    ----------
    category : str
        Policy category.
    name : str
        Algorithm name as registered.

    Returns
    -------
    type
        The policy class.

    Raises
    ------
    ValueError
        If the category or name is not found, with a helpful message
        listing available options.
    """
    if category not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise ValueError(
            f"Unknown policy category '{category}'. "
            f"Available categories: {available}"
        )

    policies = _registry[category]
    if name not in policies:
        available = ", ".join(sorted(policies.keys())) or "(none)"
        raise ValueError(
            f"Unknown {category} algorithm '{name}'. "
            f"Available options: {available}"
        )

    return policies[name]


def list_policies(category: str = None) -> Dict[str, list]:
    """
    List registered policies, optionally filtered by category.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category -> list of registered algorithm names.
    """
    if category:
        return {category: sorted(_registry.get(category, {}).keys())}
    return {cat: sorted(names.keys()) for cat, names in _registry.items()}
