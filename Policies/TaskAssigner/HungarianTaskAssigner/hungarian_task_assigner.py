"""
Hungarian Task Assigner
=======================
Optimal global assignment baseline using the Hungarian algorithm
to minimize total Manhattan distance between idle agents and pods.

匈牙利任务分配器
=======================
基于匈牙利算法的最优全局分配 baseline，最小化空闲 agent 到 pod 的总曼哈顿距离。
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from Policies.TaskAssigner.base_task_assigner import BaseTaskAssigner
from WorldState.task_state import Task, TaskType, TaskStatus
from WorldState.order_state import OrderStatus
from WorldState.agent_state import AgentStatus


def _manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class HungarianTaskAssigner(BaseTaskAssigner):
    """
    Optimal global task assigner using the Hungarian algorithm.

    Collects all (order, pod) pairs and all idle agents, builds a cost
    matrix of Manhattan distances, and solves the linear sum assignment
    to minimize total travel distance.

    基于匈牙利算法的最优全局任务分配器。
    收集所有 (订单, pod) 对与所有空闲 agent，构建曼哈顿距离代价矩阵，
    通过线性和分配求解最小化总行驶距离。
    """

    def assign(self, world_state) -> List[Task]:
        pending_orders = world_state.order_state.get_pending_orders()
        for order in pending_orders:
            if self.pod_retriever is not None:
                if not order.pod_ids:
                    order.pod_ids = self.pod_retriever.retrieve(order, world_state)
                else:
                    any_available = any(
                        (p := world_state.pod_state.get_pod(pid)) is not None
                        and not p.is_carried
                        for pid in order.pod_ids
                    )
                    if not any_available:
                        order.pod_ids = self.pod_retriever.retrieve(order, world_state)

        idle_agents = world_state.get_idle_agents()
        if not idle_agents or not pending_orders:
            return []

        reserved_pods = {
            t.pod_id
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        seen_pods: set = set()
        pods_to_assign: List[Tuple] = []
        for order in pending_orders:
            for pod_id in order.pod_ids:
                if pod_id in seen_pods or pod_id in reserved_pods:
                    continue
                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried:
                    continue
                pods_to_assign.append((order, pod))
                seen_pods.add(pod_id)

        if not pods_to_assign:
            return []

        n_agents = len(idle_agents)
        n_pods = len(pods_to_assign)
        cost = np.zeros((n_agents, n_pods))
        for i, agent in enumerate(idle_agents):
            for j, (_, pod) in enumerate(pods_to_assign):
                cost[i, j] = _manhattan_distance(agent.position, pod.current_position)

        row_idx, col_idx = linear_sum_assignment(cost)

        new_tasks: List[Task] = []
        assigned_orders = set()

        for i, j in zip(row_idx, col_idx):
            agent = idle_agents[i]
            order, pod = pods_to_assign[j]

            if pod.pod_id in reserved_pods:
                continue

            station_pos = world_state.map_state.station_positions.get(
                order.station_id
            )
            if station_pos is None:
                continue

            pick_task = Task(
                task_type=TaskType.PICK,
                order_id=order.order_id,
                pod_id=pod.pod_id,
                source=agent.position,
                destination=pod.current_position,
            )
            pick_task.agent_id = agent.agent_id
            pick_task.status = TaskStatus.ASSIGNED

            deliver_task = Task(
                task_type=TaskType.DELIVER,
                order_id=order.order_id,
                pod_id=pod.pod_id,
                source=pod.current_position,
                destination=station_pos,
            )
            deliver_task.agent_id = agent.agent_id
            deliver_task.status = TaskStatus.ASSIGNED

            if self.pod_return_planner is not None:
                return_dest = self.pod_return_planner.plan_return(
                    pod, station_pos, world_state
                )
            else:
                return_dest = pod.home_position

            return_task = Task(
                task_type=TaskType.RETURN,
                order_id=order.order_id,
                pod_id=pod.pod_id,
                source=station_pos,
                destination=return_dest,
            )
            return_task.agent_id = agent.agent_id
            return_task.status = TaskStatus.ASSIGNED

            world_state.task_state.add_task(pick_task)
            world_state.task_state.add_task(deliver_task)
            world_state.task_state.add_task(return_task)

            new_tasks.extend([pick_task, deliver_task, return_task])
            reserved_pods.add(pod.pod_id)

            agent.status = AgentStatus.MOVING_TO_POD
            agent.assigned_task_id = pick_task.task_id

            assigned_orders.add(order.order_id)

        for order in pending_orders:
            if order.order_id not in assigned_orders:
                continue
            order_task_pods = {
                t.pod_id
                for t in world_state.task_state.get_tasks_for_order(order.order_id)
            }
            if all(pid in order_task_pods for pid in order.pod_ids):
                order.status = OrderStatus.IN_PROGRESS

        return new_tasks
