"""
Station State Module
====================
Manages station queue zones: service slots, queue slots, and buffer slots.

Each station has a layered queue zone. Robots enter at the tail (or buffer
if tail is occupied), cascade forward slot-by-slot each tick, and get
serviced at the service slot. Release happens externally via _handle_actions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger("MAS_RMFS.Engine")
from typing import Dict, List, Optional, Set, Tuple

from Config.config_loader import MapConfig, StationConfig, StationQueueConfig
from WorldState.map_state import CellType


class SlotType(Enum):
    SERVICE = auto()
    QUEUE = auto()
    BUFFER = auto()


@dataclass
class StationSlot:
    position: Tuple[int, int]
    slot_type: SlotType
    index: int
    agent_id: Optional[int] = None

    @property
    def is_free(self) -> bool:
        return self.agent_id is None

    @property
    def is_occupied(self) -> bool:
        return self.agent_id is not None


class StationQueueState:
    """Per-station queue manager.

    Movement within the zone uses BFS shortest-path: each tick, every
    QUEUING agent advances one cell toward service along the physically
    shortest route through zone cells.  Agents closer to service move first
    so they clear space for those behind.
    """

    def __init__(self, station_id: int, service: StationSlot,
                 queue_slots: List[StationSlot],
                 buffer_slots: List[StationSlot],
                 entry_position: Optional[Tuple[int, int]] = None,
                 exit_position: Optional[Tuple[int, int]] = None):
        self.station_id = station_id
        self.service = service
        self.queue_slots = queue_slots  # Q0=front, Q[-1]=tail
        self.buffer_slots = buffer_slots
        self.entry_position = entry_position
        self.exit_position = exit_position
        self._assigned_agents: Set[int] = set()

        self._zone_cells: Set[Tuple[int, int]] = self._build_zone_cells()
        self._slot_by_pos: Dict[Tuple[int, int], StationSlot] = {
            s.position: s for s in self._all_slots()
        }
        self._slot_priority: Dict[Tuple[int, int], int] = self._build_slot_priority()

    def _build_zone_cells(self) -> Set[Tuple[int, int]]:
        cells = {self.service.position}
        for s in self.queue_slots:
            cells.add(s.position)
        for s in self.buffer_slots:
            cells.add(s.position)
        if self.entry_position is not None:
            cells.add(self.entry_position)
        return cells

    def _build_slot_priority(self) -> Dict[Tuple[int, int], int]:
        """Slot ordering: service=0, Q0=1, Q1=2, ..., B0=n+1, ... entry=max."""
        pri = {self.service.position: 0}
        for i, s in enumerate(self.queue_slots):
            pri[s.position] = i + 1
        n = len(self.queue_slots)
        for i, s in enumerate(self.buffer_slots):
            pri[s.position] = n + 1 + i
        if self.entry_position is not None:
            pri[self.entry_position] = n + len(self.buffer_slots) + 1
        return pri

    @property
    def tail(self) -> Optional[StationSlot]:
        return self.queue_slots[-1] if self.queue_slots else None

    def get_all_slot_positions(self) -> Set[Tuple[int, int]]:
        positions = {self.service.position}
        for s in self.queue_slots:
            positions.add(s.position)
        for s in self.buffer_slots:
            positions.add(s.position)
        return positions

    @property
    def capacity(self) -> int:
        return 1 + len(self.queue_slots) + len(self.buffer_slots)

    def reserve(self, agent_id: int) -> bool:
        """Check if station has physical capacity and track assignment.

        Only counts agents physically in slots, not in-transit agents.
        If agent arrives and buffer is full, it waits at entry.
        """
        occupied = sum(1 for s in self._all_slots() if s.is_occupied)
        if occupied >= self.capacity:
            return False
        self._assigned_agents.add(agent_id)
        return True

    def unreserve(self, agent_id: int):
        """Remove agent from assigned set (path planning failed)."""
        self._assigned_agents.discard(agent_id)

    # ---- BFS zone helpers --------------------------------------------------

    def _bfs_dist(self, start: Tuple[int, int], goal: Tuple[int, int],
                  blocked: Set[Tuple[int, int]]) -> int:
        """BFS distance from *start* to *goal* through zone cells, avoiding *blocked*.

        Returns the step count, or a large sentinel (999) if unreachable.
        """
        if start == goal:
            return 0
        from collections import deque
        visited = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, dist = queue.popleft()
            for nr, nc in self._neighbors(pos):
                npos = (nr, nc)
                if npos == goal:
                    return dist + 1
                if npos in visited or npos in blocked or npos not in self._zone_cells:
                    continue
                visited.add(npos)
                queue.append((npos, dist + 1))
        return 999

    def _bfs_next_step(self, start: Tuple[int, int], goal: Tuple[int, int],
                       blocked: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Return the next cell on the BFS shortest path from *start* to *goal*.

        Only moves through zone cells, avoiding *blocked*.
        Returns None if no path or already at goal.
        """
        if start == goal:
            return None
        from collections import deque
        visited = {start}
        queue = deque([(start, [start])])
        while queue:
            pos, path = queue.popleft()
            for nr, nc in self._neighbors(pos):
                npos = (nr, nc)
                if npos in visited:
                    continue
                new_path = path + [npos]
                if npos == goal:
                    return new_path[1]
                if npos in blocked or npos not in self._zone_cells:
                    continue
                visited.add(npos)
                queue.append((npos, new_path))
        return None

    @staticmethod
    def _neighbors(pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    # ---- Main advance (replaces cascade) -----------------------------------

    def cascade(self, world_state) -> List[Tuple[int, Tuple[int, int]]]:
        """BFS-based advance: each QUEUING agent moves 1 step toward service.

        Processing order: agents at lower slot-priority (closer to service)
        move first, freeing their slot for the agent behind.  Each agent
        only targets slots with strictly lower priority than its own — it
        never moves backward in the queue.
        """
        from WorldState.agent_state import AgentStatus

        queuing_agents: List[Tuple[Optional[StationSlot], object]] = []
        for slot in self._all_slots():
            if slot.agent_id is not None:
                agent = world_state.get_agent(slot.agent_id)
                if agent.status == AgentStatus.QUEUING:
                    queuing_agents.append((slot, agent))

        if self.entry_position is not None:
            for agent in world_state.agents:
                if (agent.status == AgentStatus.QUEUING
                        and agent.position == self.entry_position):
                    already = any(a.agent_id == agent.agent_id for _, a in queuing_agents)
                    if not already:
                        queuing_agents.append((None, agent))

        if not queuing_agents:
            return []

        max_pri = max(self._slot_priority.values()) + 1
        queuing_agents.sort(
            key=lambda x: (self._slot_priority.get(x[1].position, max_pri), x[1].agent_id)
        )

        occupied_cells = {a.position for a in world_state.agents}
        service_pos = self.service.position
        moved_agents = []
        claimed_targets: Set[Tuple[int, int]] = set()

        for from_slot, agent in queuing_agents:
            if agent.position == service_pos:
                continue

            my_pri = self._slot_priority.get(agent.position, max_pri)
            target = self._find_target_for_agent(my_pri, claimed_targets)
            if target is None:
                continue

            claimed_targets.add(target)

            blocked = (occupied_cells - {agent.position}) & self._zone_cells
            next_pos = self._bfs_next_step(agent.position, target, blocked)
            if next_pos is None:
                continue

            if next_pos in occupied_cells:
                slot_at_next = self._slot_by_pos.get(next_pos)
                if slot_at_next is None or slot_at_next.agent_id is not None:
                    continue

            old_pos = agent.position

            if from_slot is not None:
                from_slot.agent_id = None
            to_slot = self._slot_by_pos.get(next_pos)
            if to_slot is not None:
                to_slot.agent_id = agent.agent_id

            agent.position = next_pos
            occupied_cells.discard(old_pos)
            occupied_cells.add(next_pos)

            if agent.carried_pod_id is not None:
                pod = world_state.pod_state.get_pod(agent.carried_pod_id)
                if pod:
                    pod.current_position = next_pos

            moved_agents.append((agent.agent_id, next_pos))
            logger.debug(
                "Station %d advance: Agent #%d %s -> %s",
                self.station_id, agent.agent_id, old_pos, next_pos,
            )

            if next_pos == service_pos:
                agent.status = AgentStatus.DELIVERING

        return moved_agents

    def _find_target_for_agent(self, my_priority: int,
                               claimed: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find the closest-to-service free slot with priority < my_priority."""
        if (self.service.is_free
                and self.service.position not in claimed
                and 0 < my_priority):
            return self.service.position

        for slot in self.queue_slots:
            if self._slot_priority[slot.position] >= my_priority:
                break
            if slot.is_free and slot.position not in claimed:
                return slot.position

        for slot in self.buffer_slots:
            if self._slot_priority[slot.position] >= my_priority:
                break
            if slot.is_free and slot.position not in claimed:
                return slot.position

        return None

    def release(self, agent_id: int):
        """Free slot when DELIVER completes."""
        for slot in self._all_slots():
            if slot.agent_id == agent_id:
                slot.agent_id = None
                break
        self._assigned_agents.discard(agent_id)

    def check_in_from_entry(self, agent_id: int, world_state) -> bool:
        """Mark agent as QUEUING when it arrives at entry_position.

        The agent stays at entry — movement into the zone happens during
        the next cascade (BFS advance) tick.  Returns False if entry is
        wrong or zone is completely full.
        """
        agent = world_state.get_agent(agent_id)
        if agent.position != self.entry_position:
            return False

        has_space = any(s.is_free for s in self._all_slots())
        if not has_space:
            return False

        from WorldState.agent_state import AgentStatus
        agent.status = AgentStatus.QUEUING
        logger.debug(
            "Station %d: Agent #%d checked in at entry %s (QUEUING, stays put)",
            self.station_id, agent_id, self.entry_position,
        )
        return True

    def release_to_exit(self, agent_id: int, world_state) -> bool:
        """Move agent from service slot to exit_position.

        Returns False if exit is occupied. Slot stays occupied (backpressure).
        """
        if self.exit_position is None:
            return False
        occupied = {a.position for a in world_state.agents if a.agent_id != agent_id}
        if self.exit_position in occupied:
            return False
        for slot in self._all_slots():
            if slot.agent_id == agent_id:
                agent = world_state.get_agent(agent_id)
                old_pos = slot.position
                slot.agent_id = None
                agent.position = self.exit_position
                if agent.carried_pod_id is not None:
                    pod = world_state.pod_state.get_pod(agent.carried_pod_id)
                    if pod:
                        pod.current_position = self.exit_position
                self._assigned_agents.discard(agent_id)
                logger.debug(
                    "Station %d: Agent #%d released from %s%s -> exit %s",
                    self.station_id, agent_id,
                    slot.slot_type.name, old_pos, self.exit_position,
                )
                return True
        return False

    def get_slot_for_agent(self, agent_id: int) -> Optional[StationSlot]:
        """Find slot occupied by agent_id."""
        for slot in self._all_slots():
            if slot.agent_id == agent_id:
                return slot
        return None

    def _all_slots(self):
        yield self.service
        yield from self.queue_slots
        yield from self.buffer_slots


def _auto_layout(station: StationConfig, map_rows: int, map_cols: int,
                 queue_cfg: StationQueueConfig, map_state=None):
    """Auto-generate queue zone positions based on station edge position.

    Truncates queue/buffer if a position overlaps with a pod home or obstacle.
    Returns (service, queue_positions, buffer_positions, entry, exit).
    """
    r, c = station.row, station.col

    if r == 0:
        dr, dc = 1, 0
    elif r == map_rows - 1:
        dr, dc = -1, 0
    elif c == 0:
        dr, dc = 0, 1
    elif c == map_cols - 1:
        dr, dc = 0, -1
    else:
        dr, dc = 1, 0

    def _is_valid(pos):
        pr, pc = pos
        if not (0 <= pr < map_rows and 0 <= pc < map_cols):
            return False
        if map_state is not None:
            cell = map_state.grid[pr][pc]
            if cell in (CellType.OBSTACLE, CellType.POD_HOME):
                return False
        return True

    def _is_free(pos):
        """Entry/exit must be FREE only (not POD_HOME)."""
        pr, pc = pos
        if not (0 <= pr < map_rows and 0 <= pc < map_cols):
            return False
        if map_state is not None:
            return map_state.grid[pr][pc] == CellType.FREE
        return True

    service_pos = (r, c)
    if not _is_valid(service_pos):
        return service_pos, [], [], None, None

    queue_positions = []
    for i in range(queue_cfg.queue_length):
        qr = service_pos[0] + dr * (i + 1)
        qc = service_pos[1] + dc * (i + 1)
        pos = (qr, qc)
        if not _is_valid(pos):
            break
        queue_positions.append(pos)

    tail_pos = queue_positions[-1] if queue_positions else service_pos
    if dr != 0:
        perp_dr, perp_dc = 0, 1
    else:
        perp_dr, perp_dc = 1, 0

    buffer_positions = []
    for i in range(queue_cfg.buffer_length):
        br = tail_pos[0] + perp_dr * (i + 1)
        bc = tail_pos[1] + perp_dc * (i + 1)
        pos = (br, bc)
        if not _is_valid(pos):
            break
        buffer_positions.append(pos)

    zone_cells = {service_pos} | set(queue_positions) | set(buffer_positions)

    # Entry: adjacent to last buffer (or tail if no buffer)
    entry_pos = None
    anchor = buffer_positions[-1] if buffer_positions else tail_pos
    entry_candidates = [
        (anchor[0] + perp_dr, anchor[1] + perp_dc),
        (anchor[0] + dr, anchor[1] + dc),
        (anchor[0] - dr, anchor[1] - dc),
    ]
    for cand in entry_candidates:
        if _is_free(cand) and cand not in zone_cells:
            entry_pos = cand
            break

    # Exit: perpendicular to service opposite from buffer, then same side
    exit_pos = None
    exit_candidates = [
        (service_pos[0] - perp_dr, service_pos[1] - perp_dc),
        (service_pos[0] + perp_dr, service_pos[1] + perp_dc),
    ]
    for cand in exit_candidates:
        if _is_free(cand) and cand not in zone_cells:
            exit_pos = cand
            break

    return service_pos, queue_positions, buffer_positions, entry_pos, exit_pos


class StationState:
    """Container managing all station queue zones."""

    def __init__(self, config, map_state):
        self.stations: Dict[int, StationQueueState] = {}
        self._all_zone_positions: Set[Tuple[int, int]] = set()

        map_cfg = config.map
        zone_marks = {
            CellType.STATION_SERVICE: [],
            CellType.STATION_QUEUE: [],
            CellType.STATION_BUFFER: [],
            CellType.STATION_EXIT: [],
            CellType.STATION_ENTRY: [],
        }

        for station_cfg in map_cfg.stations:
            q_cfg = station_cfg.queue
            if q_cfg is None:
                continue

            # Resolve positions: explicit or auto-layout
            if q_cfg.service is not None and q_cfg.queue is not None:
                service_pos = q_cfg.service
                queue_positions = q_cfg.queue
                buffer_positions = q_cfg.buffer or []
                entry_pos = q_cfg.entry
                exit_pos = q_cfg.exit
            else:
                service_pos, queue_positions, buffer_positions, entry_pos, exit_pos = (
                    _auto_layout(station_cfg, map_cfg.rows, map_cfg.cols, q_cfg, map_state)
                )

            if entry_pos is None:
                logger.warning("Station %d: no valid entry_position found", station_cfg.id)
            if exit_pos is None:
                logger.warning("Station %d: no valid exit_position found", station_cfg.id)

            # Build slots
            service_slot = StationSlot(
                position=service_pos, slot_type=SlotType.SERVICE, index=0
            )
            queue_slots = [
                StationSlot(position=pos, slot_type=SlotType.QUEUE, index=i)
                for i, pos in enumerate(queue_positions)
            ]
            buffer_slots = [
                StationSlot(position=pos, slot_type=SlotType.BUFFER, index=i)
                for i, pos in enumerate(buffer_positions)
            ]

            sq = StationQueueState(
                station_id=station_cfg.id,
                service=service_slot,
                queue_slots=queue_slots,
                buffer_slots=buffer_slots,
                entry_position=entry_pos,
                exit_position=exit_pos,
            )
            self.stations[station_cfg.id] = sq

            # Collect positions for map marking
            station_pos = (station_cfg.row, station_cfg.col)
            if service_pos != station_pos:
                zone_marks[CellType.STATION_SERVICE].append(service_pos)
            for pos in queue_positions:
                zone_marks[CellType.STATION_QUEUE].append(pos)
            for pos in buffer_positions:
                zone_marks[CellType.STATION_BUFFER].append(pos)
            if exit_pos is not None:
                zone_marks[CellType.STATION_EXIT].append(exit_pos)
            if entry_pos is not None:
                zone_marks[CellType.STATION_ENTRY].append(entry_pos)

            self._all_zone_positions |= sq.get_all_slot_positions()

        # Mark zone cells on the map grid
        map_state.mark_station_zone(zone_marks)

    def tick(self, world_state) -> List[Tuple[int, Tuple[int, int]]]:
        """Cascade all stations. No release — that happens in _handle_actions."""
        all_moves = []
        for sq in self.stations.values():
            moves = sq.cascade(world_state)
            all_moves.extend(moves)
        return all_moves

    def get_queue(self, station_id: int) -> Optional[StationQueueState]:
        return self.stations.get(station_id)

    def get_service_position(self, station_id: int) -> Optional[Tuple[int, int]]:
        sq = self.stations.get(station_id)
        if sq is not None:
            return sq.service.position
        return None

    def get_entry_position(self, station_id: int) -> Optional[Tuple[int, int]]:
        sq = self.stations.get(station_id)
        if sq is not None:
            return sq.entry_position
        return None

    def get_exit_position(self, station_id: int) -> Optional[Tuple[int, int]]:
        sq = self.stations.get(station_id)
        if sq is not None:
            return sq.exit_position
        return None

    def get_all_zone_positions(self) -> Set[Tuple[int, int]]:
        return self._all_zone_positions
