"""
Greedy Task Assigner
====================
Default implementation: greedy assignment of orders to nearest idle robots.
"""

from typing import List, Tuple

from Policies.TaskAssigner.base_task_assigner import BaseTaskAssigner
from WorldState.task_state import Task, TaskType, TaskStatus
from WorldState.order_state import OrderStatus
from WorldState.agent_state import AgentStatus


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Compute Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class GreedyTaskAssigner(BaseTaskAssigner):
    """
    Greedy task assigner.

    For each pending order, assigns each pod to the nearest idle robot.
    Creates a chain of tasks: PICK → DELIVER → RETURN for each pod.
    """

    def assign(self, world_state) -> List[Task]:
        """Assign pending orders to idle agents greedily by distance."""
        new_tasks = []
        pending_orders = world_state.order_state.get_pending_orders()

        for order in pending_orders:
            all_assigned = True

            for pod_id in order.pod_ids:
                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried:
                    continue

                # Find nearest idle agent
                idle_agents = world_state.get_idle_agents()
                if not idle_agents:
                    all_assigned = False
                    break

                # Sort by distance to pod
                idle_agents.sort(
                    key=lambda a: _manhattan_distance(a.position, pod.current_position)
                )
                agent = idle_agents[0]

                # Get station position
                station_pos = world_state.map_state.station_positions.get(
                    order.station_id
                )
                if station_pos is None:
                    continue

                # Create PICK task: agent goes to pod location
                pick_task = Task(
                    task_type=TaskType.PICK,
                    order_id=order.order_id,
                    pod_id=pod_id,
                    source=agent.position,
                    destination=pod.current_position,
                )
                pick_task.agent_id = agent.agent_id
                pick_task.status = TaskStatus.ASSIGNED

                # Create DELIVER task: carry pod to station
                deliver_task = Task(
                    task_type=TaskType.DELIVER,
                    order_id=order.order_id,
                    pod_id=pod_id,
                    source=pod.current_position,
                    destination=station_pos,
                )
                deliver_task.agent_id = agent.agent_id
                deliver_task.status = TaskStatus.ASSIGNED

                # Create RETURN task: return pod to its home
                return_task = Task(
                    task_type=TaskType.RETURN,
                    order_id=order.order_id,
                    pod_id=pod_id,
                    source=station_pos,
                    destination=pod.home_position,
                )
                return_task.agent_id = agent.agent_id
                return_task.status = TaskStatus.ASSIGNED

                # Register tasks
                world_state.task_state.add_task(pick_task)
                world_state.task_state.add_task(deliver_task)
                world_state.task_state.add_task(return_task)

                new_tasks.extend([pick_task, deliver_task, return_task])

                # Mark agent as busy
                agent.status = AgentStatus.MOVING_TO_POD
                agent.assigned_task_id = pick_task.task_id

            if all_assigned:
                order.status = OrderStatus.IN_PROGRESS

        return new_tasks
