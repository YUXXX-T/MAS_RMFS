"""
Simulation Engine Module
========================
Core simulation loop orchestrating order generation, task assignment,
path planning, agent movement, and pickup/delivery logic.
"""

import signal
import sys
from typing import Optional

from typing import TYPE_CHECKING

from Config.config_loader import SimulationConfig
from WorldState.world import WorldState
from WorldState.agent_state import AgentStatus
from WorldState.task_state import Task, TaskType, TaskStatus
from WorldState.order_state import OrderStatus
from Policies.OrderGenerator import BaseOrderGenerator
from Policies.TaskAssigner import BaseTaskAssigner
from Policies.PathPlanner import BasePathPlanner
from Debug.logger import SimLogger

if TYPE_CHECKING:
    from TrajectoryRecord.trajectory_recorder import TrajectoryRecorder


class SimulationEngine:
    """
    Main simulation engine running a continuous tick loop.

    The loop runs indefinitely until interrupted (Ctrl+C), at which point
    it prints a summary and exits gracefully.

    参数
    ----------
    config : SimulationConfig
        Loaded simulation configuration.
    order_generator : BaseOrderGenerator
        Policy for generating orders.
    task_assigner : BaseTaskAssigner
        Policy for assigning tasks to agents.
    path_planner : BasePathPlanner
        Policy for computing agent paths.
    visualizer : object or None
        Optional visualizer with a `render(world_state)` method.
    """

    def __init__(
        self,
        config: SimulationConfig,
        order_generator: BaseOrderGenerator,
        task_assigner: BaseTaskAssigner,
        path_planner: BasePathPlanner,
        visualizer=None,
        trajectory_recorder: "TrajectoryRecorder | None" = None,
        trajectory_output: str = "",
        max_ticks: int = 0,
    ):
        self.config = config
        self.world = WorldState(config)
        self.order_generator = order_generator
        self.task_assigner = task_assigner
        self.path_planner = path_planner
        self.visualizer = visualizer
        self.trajectory_recorder = trajectory_recorder
        self.trajectory_output = trajectory_output
        self.max_ticks = max_ticks

        self.logger = SimLogger(
            "Engine",
            level=config.simulation.log_level,
            log_file=config.simulation.log_file,
        )
        self._running = True

        # 预分配模式
        self._pre_allocate()

    def _pre_allocate(self):
        """预分配模式：在仿真开始前批量生成订单并分配任务。"""
        n = self.config.simulation.initial_orders
        if n <= 0:
            return

        self.logger.info(f"Pre-allocating {n} orders...")

        generated = 0
        # Order generator only fires on tick % interval == 0 and tick > 0
        # Temporarily advance tick to force generation, then reset.
        fake_tick = self.config.simulation.order_interval
        while generated < n:
            self.world.tick = fake_tick  # pretend we're at interval tick
            orders = self.order_generator.generate(self.world)
            self.world.tick = 0         # reset
            for order in orders:
                if generated >= n:
                    break
                self.world.order_state.add_order(order)
                self.logger.info(
                    f"[Pre-alloc] Order #{order.order_id}: "
                    f"sku_demands={order.sku_demands} -> station {order.station_id}"
                )
                generated += 1
            fake_tick += self.config.simulation.order_interval
            if fake_tick > 100000:
                break  # safety

        self.world.tick = 0  # ensure tick starts at 0

        # 立即分配任务
        new_tasks = self.task_assigner.assign(self.world)
        self.logger.info(
            f"[Pre-alloc] Generated {generated} orders, assigned {len(new_tasks)} tasks"
        )

    def run(self):
        """
        Start the continuous simulation loop.

        Press Ctrl+C to stop gracefully.
        """
        # 注册信号处理器以优雅停机
        original_handler = signal.getsignal(signal.SIGINT)

        def _shutdown(signum, frame):
            self.logger.info("Shutdown signal received. Finishing current tick...")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)

        self.logger.info("=" * 60)
        self.logger.info("MAS-RMFS Simulation Started")
        self.logger.info(f"  Map: {self.world.map_state.rows}x{self.world.map_state.cols}")
        self.logger.info(f"  Agents: {len(self.world.agents)}")
        self.logger.info(f"  Pods: {self.world.pod_state.total_pods}")
        self.logger.info(f"  Stations: {len(self.world.map_state.station_positions)}")
        if self.max_ticks > 0:
            self.logger.info(f"  Max ticks: {self.max_ticks}")
        self.logger.info("  Press Ctrl+C to stop.")
        self.logger.info("="  * 60)

        try:
            while self._running:
                self._tick()
                # 达到最大 tick 数时自动停止
                if self.max_ticks > 0 and self.world.tick >= self.max_ticks:
                    self.logger.info(
                        f"Reached max ticks ({self.max_ticks}). Stopping simulation."
                    )
                    break
                if self.config.simulation.tick_delay > 0:
                    import time
                    time.sleep(self.config.simulation.tick_delay)
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            if self.trajectory_recorder and self.trajectory_output:
                self.trajectory_recorder.save(self.trajectory_output)
            self._print_summary()
            signal.signal(signal.SIGINT, original_handler)

    def _tick(self):
        """Execute one simulation tick."""
        tick = self.world.tick

        # --- 步骤 1：生成订单 ---
        new_orders = self.order_generator.generate(self.world)
        for order in new_orders:
            self.world.order_state.add_order(order)
            self.logger.info(
                f"[Tick {tick}] New Order #{order.order_id}: "
                f"sku_demands={order.sku_demands} -> station {order.station_id}"
            )

        # --- 步骤 2：分配任务 ---
        new_tasks = self.task_assigner.assign(self.world)
        for task in new_tasks:
            self.logger.debug(
                f"[Tick {tick}] Task #{task.task_id} ({task.task_type.name}) "
                f"assigned to Agent #{task.agent_id}"
            )

        # --- Step 3: Plan paths & activate tasks ---
        self._plan_and_activate(tick)

        # --- Step 4: Capture pre-move positions & Move agents ---
        prev_positions = {agent.agent_id: agent.position for agent in self.world.agents}
        self._move_agents(tick)

        # --- 步骤 5：检测冲突 (vertex & oncoming/swap) ---
        self._detect_conflicts(tick, prev_positions)

        # --- Step 6: Handle pickups, deliveries, returns ---
        self._handle_actions(tick)

        # --- 步骤 7：检查订单完成 ---
        self._check_order_completion(tick)

        # --- 步骤 8：可视化（可选） ---
        if self.visualizer:
            self.visualizer.render(self.world)

        # --- 步骤 9：记录轨迹（可选） ---
        if self.trajectory_recorder:
            self.trajectory_recorder.snapshot(self.world)

        # Advance tick
        self.world.advance_tick()

    def _plan_and_activate(self, tick: int):
        """Plan paths for agents that have assigned tasks but no active path."""
        for agent in self.world.agents:
            if agent.is_idle or agent.is_waiting:
                continue

            # If agent has no path and no active task, find next assigned task
            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is None:
                    continue
                # Activate this task
                next_task.status = TaskStatus.IN_PROGRESS
                active_task = next_task

                # Update agent status based on task type
                if active_task.task_type == TaskType.PICK:
                    agent.status = AgentStatus.MOVING_TO_POD
                elif active_task.task_type == TaskType.DELIVER:
                    agent.status = AgentStatus.CARRYING
                elif active_task.task_type == TaskType.RETURN:
                    agent.status = AgentStatus.RETURNING

                agent.assigned_task_id = active_task.task_id

            # Plan path if agent doesn't have one
            if not agent.has_path:
                path = self.path_planner.plan(
                    agent, active_task.destination, self.world
                )
                if path:
                    agent.assign_path(path)
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} planned path "
                        f"to {active_task.destination} ({len(path)} steps)"
                    )
                else:
                    self.logger.warning(
                        f"[Tick {tick}] Agent #{agent.agent_id} could not find "
                        f"path to {active_task.destination}"
                    )

    def _move_agents(self, tick: int):
        """Move each agent one step along their path."""
        for agent in self.world.agents:
            if agent.is_waiting:
                continue  # Frozen while performing an action
            if agent.has_path:
                new_pos = agent.advance()
                if new_pos:
                    # If carrying a pod, move the pod too
                    if agent.carried_pod_id is not None:
                        pod = self.world.pod_state.get_pod(agent.carried_pod_id)
                        if pod:
                            pod.current_position = new_pos
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} moved to {new_pos}"
                    )

    def _detect_conflicts(self, tick: int, prev_positions: dict):
        """Detect vertex conflicts and oncoming (head-on swap) conflicts.

        参数
        ----------
        tick : int
            当前仿真 tick。
        prev_positions : dict[int, tuple[int, int]]
            Mapping of agent_id -> position *before* this tick's movement.
        """
        # --- 顶点冲突s: two agents on the same cell ---
        pos_to_agents: dict[tuple, list] = {}
        for agent in self.world.agents:
            pos_to_agents.setdefault(agent.position, []).append(agent.agent_id)

        for pos, agent_ids in pos_to_agents.items():
            if len(agent_ids) > 1:
                ids_str = ", ".join(f"#{aid}" for aid in agent_ids)
                self.logger.warning(
                    f"[Tick {tick}] CONFLICT: Agents {ids_str} "
                    f"occupy the same cell {pos}"
                )

        # --- Oncoming (head-on / swap) conflicts ---
        # Two agents swap positions: A was at X and moved to Y while
        # B was at Y and moved to X.  Use O(n) reverse index instead of O(n²).
        prev_pos_to_agent: dict[tuple, object] = {}
        for agent in self.world.agents:
            prev_pos_to_agent[prev_positions[agent.agent_id]] = agent

        checked = set()  # avoid reporting same pair twice
        for agent in self.world.agents:
            a_prev = prev_positions[agent.agent_id]
            if agent.position == a_prev:
                continue  # didn't move
            other = prev_pos_to_agent.get(agent.position)
            if other is None:
                continue
            b_prev = prev_positions[other.agent_id]
            if other.position == a_prev and b_prev == agent.position:
                pair = (min(agent.agent_id, other.agent_id),
                        max(agent.agent_id, other.agent_id))
                if pair not in checked:
                    checked.add(pair)
                    self.logger.warning(
                        f"[Tick {tick}] ONCOMING CONFLICT: "
                        f"Agent #{agent.agent_id} ({a_prev}->{agent.position}) and "
                        f"Agent #{other.agent_id} ({b_prev}->{other.position}) "
                        f"swapped positions (head-on collision)"
                    )

    def _handle_actions(self, tick: int):
        """Handle pickup, delivery, and return actions with configurable delays."""
        sim = self.config.simulation

        for agent in self.world.agents:
            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None:
                continue

            # --- Countdown in progress: decrement and skip ---
            if agent.is_waiting:
                agent.wait_ticks -= 1
                if agent.wait_ticks > 0:
                    continue
                # Countdown just finished — fall through to perform the action
            else:
                # --- Not waiting: check if agent just arrived ---
                if agent.position != active_task.destination:
                    continue
                if agent.has_path:
                    continue  # Still moving

                # --- Pod availability check for PICK tasks ---
                if active_task.task_type == TaskType.PICK:
                    pod = self.world.pod_state.get_pod(active_task.pod_id)
                    if (pod is None
                            or pod.is_carried
                            or pod.current_position != agent.position):
                        self._handle_pod_unavailable(agent, active_task, tick)
                        continue

                # --- Determine required wait duration ---
                if active_task.task_type == TaskType.PICK:
                    required_wait = sim.pickup_duration
                elif active_task.task_type == TaskType.DELIVER:
                    required_wait = sim.station_process_duration
                elif active_task.task_type == TaskType.RETURN:
                    required_wait = sim.dropoff_duration
                else:
                    required_wait = 0

                # --- Start countdown ---
                if required_wait > 0:
                    if active_task.task_type == TaskType.DELIVER:
                        agent.status = AgentStatus.DELIVERING
                    agent.wait_ticks = required_wait
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} waiting "
                        f"{required_wait} ticks for {active_task.task_type.name} "
                        f"at {agent.position}"
                    )
                    continue  # Come back next tick

            # --- Perform the action ---
            if active_task.task_type == TaskType.PICK:
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    pod.pick_up(agent.agent_id)
                    agent.carried_pod_id = pod.pod_id
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} picked up "
                        f"Pod #{pod.pod_id} at {agent.position}"
                    )
                active_task.status = TaskStatus.COMPLETED
                agent.clear_path()

            elif active_task.task_type == TaskType.DELIVER:
                agent.status = AgentStatus.DELIVERING
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} delivered "
                        f"Pod #{pod.pod_id} to station at {agent.position}"
                    )
                    order = self.world.order_state.orders.get(active_task.order_id)
                    if order:
                        # 扣减 pod 中对应 SKU 的数量 / Deduct SKU quantities
                        for sku, demand in order.sku_demands.items():
                            if sku in pod.sku_inventory:
                                pod.sku_inventory[sku] = max(
                                    0, pod.sku_inventory[sku] - demand
                                )
                        order.mark_pod_delivered(active_task.pod_id)

                active_task.status = TaskStatus.COMPLETED
                agent.clear_path()

            elif active_task.task_type == TaskType.RETURN:
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    drop_pos = active_task.destination
                    pod.put_down(drop_pos)
                    agent.carried_pod_id = None
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} returned "
                        f"Pod #{pod.pod_id} to {drop_pos}"
                    )
                active_task.status = TaskStatus.COMPLETED
                agent.clear_path()
                agent.assigned_task_id = None

                # Check if agent has more tasks
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is None:
                    agent.status = AgentStatus.IDLE

    def _check_order_completion(self, tick: int):
        """Check and update order completion status."""
        for order in self.world.order_state.get_in_progress_orders():
            if not self.world.task_state.all_order_tasks_completed(order.order_id):
                continue

            if order.is_fully_delivered:
                order.status = OrderStatus.COMPLETED
                order.completed_at = tick
                self.logger.info(
                    f"[Tick {tick}] Order #{order.order_id} COMPLETED "
                    f"(created at tick {order.created_at}, "
                    f"duration={tick - order.created_at} ticks)"
                )
            else:
                order.status = OrderStatus.PENDING
                order.pod_ids.clear()
                self.logger.info(
                    f"[Tick {tick}] Order #{order.order_id} reset to PENDING "
                    f"(some pods were cancelled, retrying)"
                )

    def _handle_pod_unavailable(self, agent, pick_task, tick: int):
        """Handle the case where a robot arrives but the pod is gone or taken."""
        task_state = self.world.task_state

        pick_task.status = TaskStatus.CANCELLED
        related = task_state.get_related_chain_tasks(pick_task)
        for t in related:
            t.status = TaskStatus.CANCELLED

        self.logger.warning(
            f"[Tick {tick}] Agent #{agent.agent_id}: Pod #{pick_task.pod_id} "
            f"unavailable at {agent.position}. "
            f"Cancelled {1 + len(related)} tasks."
        )

        order = self.world.order_state.orders.get(pick_task.order_id)
        if order is None:
            self._reset_agent_to_idle(agent)
            return

        reserved_pods = {
            t.pod_id
            for t in task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }

        alt_pod = self._find_alternative_pod(
            order, pick_task.pod_id, reserved_pods
        )
        if alt_pod is None:
            if pick_task.pod_id in order.pod_ids:
                order.pod_ids.remove(pick_task.pod_id)
            self.logger.info(
                f"[Tick {tick}] Agent #{agent.agent_id}: No alternative pod "
                f"for Order #{order.order_id}. Setting agent to IDLE."
            )
            self._reset_agent_to_idle(agent)
            return

        station_pos = self.world.map_state.station_positions.get(
            order.station_id
        )
        if station_pos is None:
            self._reset_agent_to_idle(agent)
            return

        pod_return_planner = self.task_assigner.pod_return_planner
        if pod_return_planner is not None:
            return_dest = pod_return_planner.plan_return(
                alt_pod, station_pos, self.world
            )
        else:
            return_dest = alt_pod.home_position

        new_pick = Task(
            task_type=TaskType.PICK,
            order_id=order.order_id,
            pod_id=alt_pod.pod_id,
            source=agent.position,
            destination=alt_pod.current_position,
        )
        new_pick.agent_id = agent.agent_id
        new_pick.status = TaskStatus.ASSIGNED

        new_deliver = Task(
            task_type=TaskType.DELIVER,
            order_id=order.order_id,
            pod_id=alt_pod.pod_id,
            source=alt_pod.current_position,
            destination=station_pos,
        )
        new_deliver.agent_id = agent.agent_id
        new_deliver.status = TaskStatus.ASSIGNED

        new_return = Task(
            task_type=TaskType.RETURN,
            order_id=order.order_id,
            pod_id=alt_pod.pod_id,
            source=station_pos,
            destination=return_dest,
        )
        new_return.agent_id = agent.agent_id
        new_return.status = TaskStatus.ASSIGNED

        task_state.add_task(new_pick)
        task_state.add_task(new_deliver)
        task_state.add_task(new_return)

        if pick_task.pod_id in order.pod_ids:
            idx = order.pod_ids.index(pick_task.pod_id)
            order.pod_ids[idx] = alt_pod.pod_id

        agent.clear_path()
        agent.assigned_task_id = None
        agent.status = AgentStatus.IDLE
        agent.wait_ticks = 0

        self.logger.info(
            f"[Tick {tick}] Agent #{agent.agent_id}: Assigned alternative "
            f"Pod #{alt_pod.pod_id} for Order #{order.order_id}"
        )

    def _find_alternative_pod(self, order, original_pod_id, reserved_pods):
        """Find an alternative pod that satisfies at least some SKU demands."""
        needed_skus = {
            sku for sku, qty in order.sku_demands.items() if qty > 0
        }
        available_pods = self.world.pod_state.get_available_pods()

        best_pod = None
        best_score = 0
        for pod in available_pods:
            if pod.pod_id in reserved_pods or pod.is_carried:
                continue
            if pod.pod_id == original_pod_id:
                continue
            score = sum(
                1 for sku in needed_skus
                if sku in pod.sku_inventory and pod.sku_inventory[sku] > 0
            )
            if score > best_score:
                best_score = score
                best_pod = pod
        return best_pod

    def _reset_agent_to_idle(self, agent):
        """Reset an agent to IDLE state."""
        agent.clear_path()
        agent.assigned_task_id = None
        agent.status = AgentStatus.IDLE
        agent.wait_ticks = 0
        agent.carried_pod_id = None

    def _print_summary(self):
        """Print simulation summary on shutdown."""
        self.logger.info("=" * 60)
        self.logger.info("SIMULATION SUMMARY")
        self.logger.info(f"  Total ticks:      {self.world.tick}")
        self.logger.info(f"  Total orders:     {self.world.order_state.total_orders}")
        self.logger.info(f"  Completed orders: {self.world.order_state.total_completed}")
        in_progress = len(self.world.order_state.get_in_progress_orders())
        pending = len(self.world.order_state.get_pending_orders())
        self.logger.info(f"  In-progress:      {in_progress}")
        self.logger.info(f"  Pending:          {pending}")
        self.logger.info("=" * 60)
