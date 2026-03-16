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


class SimulationEngine:
    """
    Main simulation engine running a continuous tick loop.

    The loop runs indefinitely until interrupted (Ctrl+C), at which point
    it prints a summary and exits gracefully.

    Parameters
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
        self._running = True

    def run(self):
        """
        Start the continuous simulation loop.

        Press Ctrl+C to stop gracefully.
        """
        # Register signal handler for graceful shutdown
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
        self.logger.info("  Press Ctrl+C to stop.")
        self.logger.info("=" * 60)

        try:
            while self._running:
                self._tick()
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise
        finally:
            self._print_summary()
            signal.signal(signal.SIGINT, original_handler)

    def _tick(self):
        """Execute one simulation tick."""
        tick = self.world.tick

        # --- Step 1: Generate orders ---
        new_orders = self.order_generator.generate(self.world)
        for order in new_orders:
            self.world.order_state.add_order(order)
            self.logger.info(
                f"[Tick {tick}] New Order #{order.order_id}: "
                f"pods={order.pod_ids} -> station {order.station_id}"
            )

        # --- Step 2: Assign tasks ---
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

        # --- Step 5: Detect conflicts (vertex & oncoming/swap) ---
        self._detect_conflicts(tick, prev_positions)

        # --- Step 6: Handle pickups, deliveries, returns ---
        self._handle_actions(tick)

        # --- Step 7: Check order completion ---
        self._check_order_completion(tick)

        # --- Step 8: Visualize (optional) ---
        if self.visualizer:
            self.visualizer.render(self.world)

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

        Parameters
        ----------
        tick : int
            Current simulation tick.
        prev_positions : dict[int, tuple[int, int]]
            Mapping of agent_id -> position *before* this tick's movement.
        """
        # --- Vertex conflicts: two agents on the same cell ---
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
                    self.logger.warning(
                        f"[Tick {tick}] ONCOMING CONFLICT: "
                        f"Agent #{a.agent_id} ({a_prev}->{a.position}) and "
                        f"Agent #{b.agent_id} ({b_prev}->{b.position}) "
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
                        order.mark_pod_delivered(active_task.pod_id)

                active_task.status = TaskStatus.COMPLETED
                agent.clear_path()

            elif active_task.task_type == TaskType.RETURN:
                pod = self.world.pod_state.get_pod(active_task.pod_id)
                if pod:
                    pod.put_down(pod.home_position)
                    agent.carried_pod_id = None
                    self.logger.info(
                        f"[Tick {tick}] Agent #{agent.agent_id} returned "
                        f"Pod #{pod.pod_id} to home {pod.home_position}"
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
