"""
Simulation Engine Module
========================
Core simulation loop orchestrating order generation, task assignment,
path planning, agent movement, and pickup/delivery logic.
"""

import signal
import sys
from typing import Optional

from Config.config_loader import SimulationConfig
from WorldState.world import WorldState
from WorldState.agent_state import AgentStatus
from WorldState.task_state import TaskType, TaskStatus
from WorldState.order_state import OrderStatus
from Policies.OrderGenerator import BaseOrderGenerator
from Policies.TaskAssigner import BaseTaskAssigner
from Policies.PathPlanner import BasePathPlanner
from Debug.logger import SimLogger
from Metrics.tracker import MetricsTracker


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
    ):
        self.config = config
        self.world = WorldState(config)
        self.order_generator = order_generator
        self.task_assigner = task_assigner
        self.path_planner = path_planner
        self.visualizer = visualizer

        self.logger = SimLogger(
            "Engine",
            level=config.simulation.log_level,
            log_file=config.simulation.log_file,
        )
        self.metrics = MetricsTracker()
        self.on_tick_callbacks: list = []
        self._running = True

        # Conflict details from the most recent tick — populated by
        # _detect_conflicts so callbacks (e.g. SnapshotCollector) can serialize them.
        self.last_vertex_conflicts: list[tuple] = []   # [((r,c), [agent_ids...]), ...]
        self.last_swap_conflicts: list[tuple] = []     # [((r1,c1), (r2,c2), aid_a, aid_b), ...]

    def run(self):
        """
        Start the continuous simulation loop.

        Press Ctrl+C to stop gracefully.
        """
        # 注册信号处理器以优雅停机（SIGINT=Ctrl+C, SIGTERM=`timeout` / `kill`）
        original_int = signal.getsignal(signal.SIGINT)
        original_term = signal.getsignal(signal.SIGTERM)

        def _shutdown(signum, frame):
            sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            self.logger.info(f"{sig_name} received. Finishing current tick...")
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        self.logger.info("=" * 60)
        self.logger.info("MAS-RMFS Simulation Started")
        self.logger.info(f"  Map: {self.world.map_state.rows}x{self.world.map_state.cols}")
        self.logger.info(f"  Agents: {len(self.world.agents)}")
        self.logger.info(f"  Pods: {self.world.pod_state.total_pods}")
        self.logger.info(f"  Stations: {len(self.world.map_state.station_positions)}")
        self.logger.info("  Press Ctrl+C to stop.")
        self.logger.info("=" * 60)

        try:
            while self._running:
                self._tick()
                max_ticks = self.config.simulation.max_ticks
                if max_ticks and self.world.tick >= max_ticks:
                    self.logger.info(
                        f"Reached max_ticks={max_ticks}; stopping."
                    )
                    self._running = False
                    break
                if self.config.simulation.tick_delay > 0:
                    import time
                    time.sleep(self.config.simulation.tick_delay)
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            self._print_summary()
            signal.signal(signal.SIGINT, original_int)
            signal.signal(signal.SIGTERM, original_term)

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

        # --- 步骤 2：分配任务（计时） ---
        self.metrics.start_timer("assign")
        new_tasks = self.task_assigner.assign(self.world)
        assign_ms = self.metrics.stop_timer("assign")
        for task in new_tasks:
            self.logger.debug(
                f"[Tick {tick}] Task #{task.task_id} ({task.task_type.name}) "
                f"assigned to Agent #{task.agent_id}"
            )

        # --- Step 3: Plan paths & activate tasks（计时） ---
        self.metrics.start_timer("plan")
        self._plan_and_activate(tick)
        plan_ms = self.metrics.stop_timer("plan")

        # --- Step 4: Capture pre-move positions & Move agents ---
        prev_positions = {agent.agent_id: agent.position for agent in self.world.agents}
        self._move_agents(tick)

        # --- 步骤 5：检测冲突 (vertex & oncoming/swap) ---
        vertex_conflicts = self._detect_conflicts(tick, prev_positions)

        # --- Step 6: Handle pickups, deliveries, returns ---
        self._handle_actions(tick)

        # --- 步骤 7：检查订单完成 ---
        self._check_order_completion(tick)

        # --- 步骤 8：记录指标 ---
        self.metrics.record(self.world, vertex_conflicts, assign_ms, plan_ms)

        # --- 步骤 8.5：tick 回调（快照采集等） ---
        for cb in self.on_tick_callbacks:
            cb(self)

        # --- 步骤 9：可视化（可选） ---
        if self.visualizer:
            self.visualizer.render(self.world)

        # Advance tick
        self.world.advance_tick()

    def _plan_and_activate(self, tick: int):
        """Plan paths for agents that have assigned tasks but no active path."""

        # --- Phase 1: activate tasks for agents that need them ---
        agents_need_plan = []
        for agent in self.world.agents:
            if agent.is_idle or agent.is_waiting:
                continue

            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is None:
                    continue
                next_task.status = TaskStatus.IN_PROGRESS
                active_task = next_task

                if active_task.task_type == TaskType.PICK:
                    agent.status = AgentStatus.MOVING_TO_POD
                elif active_task.task_type == TaskType.DELIVER:
                    agent.status = AgentStatus.CARRYING
                elif active_task.task_type == TaskType.RETURN:
                    agent.status = AgentStatus.RETURNING

                agent.assigned_task_id = active_task.task_id

            if not agent.has_path:
                agents_need_plan.append((agent, active_task.destination))

        if not agents_need_plan:
            return

        # --- Phase 2: plan paths (batch or individual) ---
        if hasattr(self.path_planner, "plan_batch"):
            paths = self.path_planner.plan_batch(agents_need_plan, self.world)
            for agent, goal in agents_need_plan:
                path = paths.get(agent.agent_id)
                if path:
                    agent.assign_path(path)
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} planned path "
                        f"to {goal} ({len(path)} steps)"
                    )
                else:
                    self.logger.warning(
                        f"[Tick {tick}] Agent #{agent.agent_id} could not find "
                        f"path to {goal}"
                    )
        else:
            for agent, goal in agents_need_plan:
                path = self.path_planner.plan(agent, goal, self.world)
                if path:
                    agent.assign_path(path)
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} planned path "
                        f"to {goal} ({len(path)} steps)"
                    )
                else:
                    self.logger.warning(
                        f"[Tick {tick}] Agent #{agent.agent_id} could not find "
                        f"path to {goal}"
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

    def _detect_conflicts(self, tick: int, prev_positions: dict) -> int:
        """Detect vertex conflicts and oncoming (head-on swap) conflicts.

        参数
        ----------
        tick : int
            当前仿真 tick。
        prev_positions : dict[int, tuple[int, int]]
            Mapping of agent_id -> position *before* this tick's movement.

        返回
        ----
        int
            Number of positions where vertex conflicts occurred.
        """
        # --- 顶点冲突s: two agents on the same cell ---
        vertex_conflict_count = 0
        self.last_vertex_conflicts = []
        self.last_swap_conflicts = []
        pos_to_agents: dict[tuple, list] = {}
        for agent in self.world.agents:
            pos_to_agents.setdefault(agent.position, []).append(agent.agent_id)

        for pos, agent_ids in pos_to_agents.items():
            if len(agent_ids) > 1:
                vertex_conflict_count += 1
                self.last_vertex_conflicts.append((pos, list(agent_ids)))
                ids_str = ", ".join(f"#{aid}" for aid in agent_ids)
                self.logger.warning(
                    f"[Tick {tick}] CONFLICT: Agents {ids_str} "
                    f"occupy the same cell {pos}"
                )

        # --- Oncoming (head-on / swap) conflicts ---
        # Two agents swap positions: A was at X and moved to Y while
        # B was at Y and moved to X.  This means they crossed the same
        # edge in opposite directions during this tick.
        agents = self.world.agents
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                a_prev = prev_positions[a.agent_id]
                b_prev = prev_positions[b.agent_id]
                # Check if they swapped (and actually moved)
                if (
                    a.position == b_prev
                    and b.position == a_prev
                    and a_prev != a.position  # A actually moved
                ):
                    self.last_swap_conflicts.append(
                        (a_prev, b_prev, a.agent_id, b.agent_id)
                    )
                    self.logger.warning(
                        f"[Tick {tick}] ONCOMING CONFLICT: "
                        f"Agent #{a.agent_id} ({a_prev}->{a.position}) and "
                        f"Agent #{b.agent_id} ({b_prev}->{b.position}) "
                        f"swapped positions (head-on collision)"
                    )

        return vertex_conflict_count

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
            if self.world.task_state.all_order_tasks_completed(order.order_id):
                order.status = OrderStatus.COMPLETED
                order.completed_at = tick
                self.logger.info(
                    f"[Tick {tick}] Order #{order.order_id} COMPLETED "
                    f"(created at tick {order.created_at}, "
                    f"duration={tick - order.created_at} ticks)"
                )
                # Free up all agents that worked on this order
                order_tasks = self.world.task_state.get_tasks_for_order(order.order_id)
                agent_ids = {t.agent_id for t in order_tasks if t.agent_id is not None}
                for aid in agent_ids:
                    agent = self.world.get_agent(aid)
                    if agent.assigned_task_id is None and not agent.is_idle:
                        next_task = self.world.task_state.get_next_task_for_agent(aid)
                        if next_task is None:
                            agent.status = AgentStatus.IDLE

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

        summary = self.metrics.summarize()
        if summary:
            self.logger.info("-" * 40)
            self.logger.info("METRICS")
            self.logger.info(f"  Throughput (final):          {summary['final_throughput']}")
            self.logger.info(f"  Throughput (avg/100 ticks):  {summary['avg_throughput_per_100tick']:.2f}")
            self.logger.info(f"  Deadlock events (ticks):     {summary['total_deadlock_events']}")
            self.logger.info(f"  Congestion events (total):   {summary['total_congestion_events']}")
            self.logger.info(f"  Avg planning time:           {summary['avg_plan_ms']:.2f} ms")
            self.logger.info(f"  Avg assignment time:         {summary['avg_assign_ms']:.2f} ms")

        self.logger.info("=" * 60)
