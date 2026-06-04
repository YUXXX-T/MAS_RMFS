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

        # --- Step 3: Station queue cascade ---
        self.world.station_state.tick(self.world)

        # --- Step 3b: Process station exits (EXITING -> exit_position) ---
        self._process_station_exits(tick)

        # --- Step 3c: Absorb agents stranded at entry from previous tick ---
        absorbed_entries = set()
        self._check_queue_arrivals(tick, absorbed_entries)

        # --- Step 3d: Clear stale DELIVER paths near free entry ---
        self._clear_stale_entry_paths(tick)

        # --- Step 4: Plan paths & activate tasks ---
        self._plan_and_activate(tick)

        # --- Step 5: Capture pre-move positions & Move agents ---
        prev_positions = {agent.agent_id: agent.position for agent in self.world.agents}
        self._move_agents(tick)

        # --- Step 6: Check queue arrivals (new arrivals) ---
        self._check_queue_arrivals(tick, absorbed_entries)

        # --- 步骤 7：检测冲突 (vertex & oncoming/swap) ---
        self._detect_conflicts(tick, prev_positions)

        # --- Step 8: Handle pickups, deliveries, returns ---
        self._handle_actions(tick)

        # --- Step 9: Check order completion ---
        self._check_order_completion(tick)

        # --- Step 10: Visualize ---
        if self.visualizer:
            self.visualizer.render(self.world)

        # --- Step 11: Record trajectory ---
        if self.trajectory_recorder:
            self.trajectory_recorder.snapshot(self.world)

        # Advance tick
        self.world.advance_tick()

    def _plan_and_activate(self, tick: int):
        """Plan paths for agents that have assigned tasks but no active path."""
        station_state = self.world.station_state

        active_station_ids = set()
        for agent in self.world.agents:
            if agent.status == AgentStatus.EXITING:
                t = self.world.task_state.get_active_task_for_agent(agent.agent_id)
                if t:
                    active_station_ids.add(t.station_id)
            elif agent.status == AgentStatus.CARRYING and not agent.has_path:
                t = self.world.task_state.get_active_task_for_agent(agent.agent_id)
                if t and t.task_type == TaskType.DELIVER:
                    active_station_ids.add(t.station_id)

        handoff_blocked = set()
        for sq in station_state.stations.values():
            if sq.station_id not in active_station_ids:
                continue
            if sq.entry_position:
                handoff_blocked.add(sq.entry_position)
            if sq.exit_position:
                handoff_blocked.add(sq.exit_position)

        agents_sorted = sorted(
            self.world.agents,
            key=lambda a: (0 if a.position in handoff_blocked else 1, a.agent_id),
        )

        for agent in agents_sorted:
            if agent.status in (AgentStatus.QUEUING, AgentStatus.DELIVERING,
                                AgentStatus.EXITING):
                continue
            if agent.is_idle and agent.position in handoff_blocked:
                self.logger.warning(
                    f"[Tick {tick}] Agent #{agent.agent_id} is IDLE on handoff "
                    f"cell {agent.position} — may block station exit/entry"
                )
            if agent.is_idle or agent.is_waiting:
                continue

            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)

            # -- DELIVER branch: fully transactional, separate from generic flow --
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is not None and next_task.task_type == TaskType.DELIVER:
                    queue = station_state.get_queue(next_task.station_id)
                    if queue is None or queue.entry_position is None:
                        continue

                    if not queue.reserve(agent.agent_id):
                        continue  # station at capacity, defer

                    goal = queue.entry_position

                    if any(a.position == goal and a.agent_id != agent.agent_id
                           and not a.has_path
                           for a in self.world.agents):
                        queue.unreserve(agent.agent_id)
                        continue
                    extra_blocked = handoff_blocked - {goal, agent.position}
                    path = self.path_planner.plan(
                        agent, goal, self.world, extra_blocked=extra_blocked
                    )
                    if not path:
                        queue.unreserve(agent.agent_id)
                        self.logger.warning(
                            f"[Tick {tick}] Agent #{agent.agent_id} could not find "
                            f"path to entry {goal} for DELIVER"
                        )
                        continue

                    next_task.status = TaskStatus.IN_PROGRESS
                    agent.status = AgentStatus.CARRYING
                    agent.assigned_task_id = next_task.task_id
                    agent.assign_path(path)
                    agent.plan_failed_streak = 0
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} DELIVER → "
                        f"entry {goal} ({len(path)} steps)"
                    )
                    continue

            # -- Generic branch: PICK, RETURN, already-active tasks --
            if active_task is None:
                next_task = self.world.task_state.get_next_task_for_agent(agent.agent_id)
                if next_task is None:
                    continue
                next_task.status = TaskStatus.IN_PROGRESS
                active_task = next_task

                if active_task.task_type == TaskType.PICK:
                    agent.status = AgentStatus.MOVING_TO_POD
                elif active_task.task_type == TaskType.RETURN:
                    agent.status = AgentStatus.RETURNING

                agent.assigned_task_id = active_task.task_id

            if not agent.has_path:
                if (active_task.task_type == TaskType.DELIVER
                        and agent.status == AgentStatus.CARRYING):
                    queue = station_state.get_queue(active_task.station_id)
                    if queue and queue.entry_position:
                        goal = queue.entry_position
                    else:
                        goal = active_task.destination
                else:
                    goal = active_task.destination
                if agent.position == goal:
                    continue  # already at destination, _handle_actions will process
                extra_blocked = handoff_blocked - {goal, agent.position}
                path = self.path_planner.plan(
                    agent, goal, self.world,
                    extra_blocked=extra_blocked if extra_blocked else None,
                )
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
                    if agent.position in handoff_blocked:
                        self._nudge_idle_neighbors(agent, tick)

    def _move_agents(self, tick: int):
        """Move each agent one step along their path."""
        for agent in self.world.agents:
            if agent.status in (AgentStatus.QUEUING, AgentStatus.DELIVERING,
                                AgentStatus.EXITING):
                continue
            if agent.is_waiting:
                continue
            if agent.has_path:
                new_pos = agent.advance()
                if new_pos:
                    if agent.carried_pod_id is not None:
                        pod = self.world.pod_state.get_pod(agent.carried_pod_id)
                        if pod:
                            pod.current_position = new_pos
                    self.logger.debug(
                        f"[Tick {tick}] Agent #{agent.agent_id} moved to {new_pos}"
                    )

    def _check_queue_arrivals(self, tick: int, absorbed_entries: set):
        """Absorb CARRYING agents at entry_position into queue slots."""
        for agent in self.world.agents:
            if agent.status != AgentStatus.CARRYING:
                continue

            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None or active_task.task_type != TaskType.DELIVER:
                continue

            queue = self.world.station_state.get_queue(active_task.station_id)
            if queue is None or queue.entry_position is None:
                continue

            if agent.position != queue.entry_position:
                continue

            if agent.has_path:
                agent.clear_path()

            if queue.entry_position in absorbed_entries:
                continue

            if queue.check_in_from_entry(agent.agent_id, self.world):
                absorbed_entries.add(queue.entry_position)
                self.logger.info(
                    f"[Tick {tick}] Agent #{agent.agent_id} checked into queue "
                    f"at station {active_task.station_id} (entry={agent.position})"
                )

    def _clear_stale_entry_paths(self, tick: int):
        """Clear stale paths for DELIVER agents near a free entry.

        The space-time path planner may schedule waits or detours because
        it predicted the entry would be occupied.  When the entry has
        since become free, the stale path wastes ticks.  Clearing it
        lets ``_plan_and_activate`` re-plan a direct route this same tick.
        For adjacent agents, assigns a direct 1-step path to entry.
        """
        station_state = self.world.station_state
        claimed_entries = set()

        for agent in self.world.agents:
            if agent.status != AgentStatus.CARRYING or not agent.has_path:
                continue
            active_task = self.world.task_state.get_active_task_for_agent(
                agent.agent_id
            )
            if active_task is None or active_task.task_type != TaskType.DELIVER:
                continue
            queue = station_state.get_queue(active_task.station_id)
            if queue is None or queue.entry_position is None:
                continue

            remaining = agent.path[agent.path_index:]
            if not remaining or remaining[-1] != queue.entry_position:
                continue

            er, ec = queue.entry_position
            ar, ac = agent.position
            manhattan = abs(er - ar) + abs(ec - ac)
            if manhattan > 3:
                continue

            if len(remaining) <= manhattan:
                continue

            entry_pos = queue.entry_position
            if entry_pos in claimed_entries:
                continue

            entry_blocked = False
            for other in self.world.agents:
                if other.agent_id == agent.agent_id:
                    continue
                if other.position == entry_pos:
                    entry_blocked = True
                    break
                if other.has_path and other.path_index < len(other.path):
                    if other.path[other.path_index] == entry_pos:
                        entry_blocked = True
                        break
            if entry_blocked:
                continue

            if manhattan == 1:
                agent.assign_path([entry_pos])
                claimed_entries.add(entry_pos)
            else:
                agent.clear_path()
            self.logger.debug(
                f"[Tick {tick}] Cleared stale path for Agent #{agent.agent_id} "
                f"near entry {entry_pos} (dist={manhattan}, "
                f"remaining={len(remaining)} steps)"
            )

    def _process_station_exits(self, tick: int):
        """Two-phase exit: service→exit (1 step), then finalize next tick."""
        for agent in self.world.agents:
            if agent.status != AgentStatus.EXITING:
                continue
            active_task = self.world.task_state.get_active_task_for_agent(agent.agent_id)
            if active_task is None:
                continue
            queue = self.world.station_state.get_queue(active_task.station_id)
            if queue is None:
                continue

            if queue.exit_position and agent.position == queue.exit_position:
                active_task.status = TaskStatus.COMPLETED
                agent.clear_path()
                agent.status = AgentStatus.CARRYING
                agent.assigned_task_id = None
                self.logger.info(
                    f"[Tick {tick}] Agent #{agent.agent_id} finalized exit "
                    f"at station {active_task.station_id}, pos={agent.position}"
                )
                continue

            if queue.release_to_exit(agent.agent_id, self.world):
                self.logger.info(
                    f"[Tick {tick}] Agent #{agent.agent_id} moved to exit "
                    f"at station {active_task.station_id}, pos={agent.position}"
                )

    def _nudge_idle_neighbors(self, stuck_agent, tick: int):
        """Nudge idle agents adjacent to *stuck_agent* so it can leave a handoff cell.

        Assigns a 1-step path to each idle neighbor, moving it to a free
        adjacent cell.  The nudge takes effect this tick's _move_agents,
        clearing the corridor for the stuck agent on the next tick.
        """
        ms = self.world.map_state
        occupied = {a.position for a in self.world.agents}
        pod_positions = {
            p.current_position
            for p in self.world.pod_state.pods.values()
            if not p.is_carried
        }
        sr, sc = stuck_agent.position
        for other in self.world.agents:
            if other.agent_id == stuck_agent.agent_id:
                continue
            if other.status != AgentStatus.IDLE or other.has_path:
                continue
            odr = abs(other.position[0] - sr)
            odc = abs(other.position[1] - sc)
            if odr + odc > 2:
                continue
            for nr, nc in ms.get_neighbors(other.position[0], other.position[1]):
                if not ms.is_walkable(nr, nc):
                    continue
                if (nr, nc) in occupied or (nr, nc) in pod_positions:
                    continue
                other.assign_path([(nr, nc)])
                occupied.discard(other.position)
                occupied.add((nr, nc))
                self.logger.info(
                    f"[Tick {tick}] Nudge: Agent #{other.agent_id} "
                    f"{other.position} -> ({nr},{nc}) to clear handoff "
                    f"for Agent #{stuck_agent.agent_id}"
                )
                break

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

            if agent.status == AgentStatus.EXITING:
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
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} delivered "
                        f"Pod #{pod.pod_id} to station at {agent.position}"
                    )
                    order = self.world.order_state.orders.get(active_task.order_id)
                    if order:
                        for sku, demand in order.sku_demands.items():
                            if sku in pod.sku_inventory:
                                pod.sku_inventory[sku] = max(
                                    0, pod.sku_inventory[sku] - demand
                                )
                        order.mark_pod_delivered(active_task.pod_id)

                agent.status = AgentStatus.EXITING

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

        station_pos = self.world.station_state.get_service_position(
            order.station_id
        )
        if station_pos is None:
            station_pos = self.world.map_state.station_positions.get(
                order.station_id
            )
        if station_pos is None:
            self._reset_agent_to_idle(agent)
            return

        exit_pos = self.world.station_state.get_exit_position(order.station_id)
        return_source = exit_pos or station_pos

        pod_return_planner = self.task_assigner.pod_return_planner
        if pod_return_planner is not None:
            return_dest = pod_return_planner.plan_return(
                alt_pod, return_source, self.world
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
        new_deliver.station_id = order.station_id

        new_return = Task(
            task_type=TaskType.RETURN,
            order_id=order.order_id,
            pod_id=alt_pod.pod_id,
            source=return_source,
            destination=return_dest,
        )
        new_return.agent_id = agent.agent_id
        new_return.status = TaskStatus.ASSIGNED
        new_return.station_id = order.station_id

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
