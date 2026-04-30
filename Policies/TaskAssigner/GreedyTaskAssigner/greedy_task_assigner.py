"""

Greedy Task Assigner
====================
Default implementation: greedy assignment of orders to nearest idle robots.

贪婪任务分配器
====================
默认实现：将订单贪婪地分配给距离最近的空闲机器人。

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

    Supports two modes via config ``task_execution_mode``:
    - ``"parallel"`` (default): each pod in an order is assigned to a
      different idle robot, so multiple robots work in parallel.
    - ``"serial"``: all pods in an order are assigned to the same robot,
      which processes them one by one in sequence.

    贪婪任务分配器
    ====================
    对于每一个待处理订单，将对应的货架分配给距离最近的空闲机器人。
    为每个货架创建一条任务链：拣货 → 配送 → 归位。

    支持两种模式（通过 config 中的 task_execution_mode 控制）：
    - "parallel"（默认）：同一订单的不同 pod 分配给不同机器人并行处理。
    - "serial"：同一订单的所有 pod 分配给同一台机器人串行处理。
    """

    def assign(self, world_state) -> List[Task]:
        """
        Assign pending orders to idle agents.
        根据 task_execution_mode 选择并行或串行模式分配任务。
        在分配前，先调用 PodRetriever 将 SKU 需求转化为 pod 列表。
        """
        # 先对所有待处理订单调用 PodRetriever 填充 pod_ids
        pending_orders = world_state.order_state.get_pending_orders()
        for order in pending_orders:
            if not order.pod_ids and self.pod_retriever is not None:
                retrieved_ids = self.pod_retriever.retrieve(order, world_state)
                order.pod_ids = retrieved_ids

        mode = world_state.config.simulation.task_execution_mode
        if mode == "serial":
            return self._assign_serial(world_state)
        else:
            return self._assign_parallel(world_state)

    def _assign_parallel(self, world_state) -> List[Task]:
        """
        并行模式：每个 pod 分配给不同的空闲机器人。
        Parallel mode: each pod is assigned to the nearest idle robot.
        """
        new_tasks = []
        pending_orders = world_state.order_state.get_pending_orders()

        # Build set of pods already reserved by an active task
        reserved_pods = {
            t.pod_id
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        for order in pending_orders:
            all_assigned = True

            for pod_id in order.pod_ids:
                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried or pod_id in reserved_pods:
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

                # Create RETURN task: return pod to designated location
                if self.pod_return_planner is not None:
                    return_dest = self.pod_return_planner.plan_return(
                        pod, station_pos, world_state
                    )
                else:
                    return_dest = pod.home_position

                return_task = Task(
                    task_type=TaskType.RETURN,
                    order_id=order.order_id,
                    pod_id=pod_id,
                    source=station_pos,
                    destination=return_dest,
                )
                return_task.agent_id = agent.agent_id
                return_task.status = TaskStatus.ASSIGNED

                # Register tasks
                world_state.task_state.add_task(pick_task)
                world_state.task_state.add_task(deliver_task)
                world_state.task_state.add_task(return_task)

                new_tasks.extend([pick_task, deliver_task, return_task])
                reserved_pods.add(pod_id)

                # Mark agent as busy
                agent.status = AgentStatus.MOVING_TO_POD
                agent.assigned_task_id = pick_task.task_id

            if all_assigned:
                order.status = OrderStatus.IN_PROGRESS

        return new_tasks

    def _assign_serial(self, world_state) -> List[Task]:
        """
        串行模式：同一订单的所有 pod 由同一台机器人依次完成。
        Serial mode: all pods in an order are handled sequentially by one robot.

        Task 创建顺序：
        PICK_pod1 → DELIVER_pod1 → RETURN_pod1 → PICK_pod2 → DELIVER_pod2 → RETURN_pod2 → ...
        """
        new_tasks = []
        pending_orders = world_state.order_state.get_pending_orders()

        # Build set of pods already reserved by an active task
        reserved_pods = {
            t.pod_id
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        for order in pending_orders:
            # 检查所有 pod 是否可用
            order_pods = []
            all_available = True
            for pod_id in order.pod_ids:
                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried or pod_id in reserved_pods:
                    all_available = False
                    break
                order_pods.append(pod)

            if not all_available or not order_pods:
                order.pod_ids = []
                continue

            # 找一个空闲机器人（按到第一个 pod 的距离排序）
            idle_agents = world_state.get_idle_agents()
            if not idle_agents:
                continue

            idle_agents.sort(
                key=lambda a: _manhattan_distance(
                    a.position, order_pods[0].current_position
                )
            )
            agent = idle_agents[0]

            # Get station position
            station_pos = world_state.map_state.station_positions.get(
                order.station_id
            )
            if station_pos is None:
                continue

            # 为所有 pod 串行创建 PICK → DELIVER → RETURN
            order_tasks = []
            for pod in order_pods:
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

                order_tasks.extend([pick_task, deliver_task, return_task])
                reserved_pods.add(pod.pod_id)

            new_tasks.extend(order_tasks)

            # Mark agent as busy with the first task
            agent.status = AgentStatus.MOVING_TO_POD
            agent.assigned_task_id = order_tasks[0].task_id

            order.status = OrderStatus.IN_PROGRESS

        return new_tasks

