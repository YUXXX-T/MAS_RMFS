"""
Visualizer Module
=================
Visualization backends for the MAS-RMFS simulation.

Includes:
    - TerminalVisualizer  — ASCII grid to stdout
    - MatplotlibVisualizer — animated 2×2 dashboard
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from WorldState.world import WorldState


class BaseVisualizer(ABC):
    """Interface for simulation visualization."""

    @abstractmethod
    def render(self, world_state: "WorldState"):
        """Render the current world state."""
        ...


class TerminalVisualizer(BaseVisualizer):
    """
    Simple text-based visualizer that prints a grid to stdout.

    Legend:
        .  = free cell
        #  = obstacle
        S  = station
        P  = pod (at home)
        R  = robot (idle)
        *  = robot carrying pod
    """

    def render(self, world_state):
        """Print a text grid of the current world state."""
        ms = world_state.map_state
        grid = [["." for _ in range(ms.cols)] for _ in range(ms.rows)]

        # Mark cell types
        for r in range(ms.rows):
            for c in range(ms.cols):
                from WorldState.map_state import CellType
                cell = ms.grid[r][c]
                if cell == CellType.OBSTACLE:
                    grid[r][c] = "#"
                elif cell == CellType.STATION:
                    grid[r][c] = "S"
                elif cell == CellType.POD_HOME:
                    grid[r][c] = "P"

        # Mark pods not at home (being carried or displaced)
        for pod in world_state.pod_state.pods.values():
            if not pod.is_carried and not pod.is_at_home:
                pr, pc = pod.current_position
                grid[pr][pc] = "p"

        # Mark agents
        for agent in world_state.agents:
            r, c = agent.position
            if agent.carried_pod_id is not None:
                grid[r][c] = "*"
            else:
                grid[r][c] = "R"

        # Print
        header = f"=== Tick {world_state.tick} ==="
        print(header)
        print("  " + " ".join(str(c) for c in range(ms.cols)))
        for r in range(ms.rows):
            print(f"{r} " + " ".join(grid[r]))
        print()


# ---------------------------------------------------------------------------
# Matplotlib animated dashboard
# ---------------------------------------------------------------------------

class MatplotlibVisualizer(BaseVisualizer):
    """
    Animated 2×2 matplotlib dashboard updated every simulation tick.

    Panels
    ------
    Top-left     : Live warehouse grid (obstacles, stations, pods, robots).
    Top-right    : Cumulative path-density heatmap (YlOrRd).
    Bottom-left  : Agent-status timeline (agent × tick colour-coded).
    Bottom-right : Cumulative completed-orders throughput curve.
    """

    # Status → integer code for the timeline colour-map
    _STATUS_CODES = {
        "IDLE": 0,
        "MOVING_TO_POD": 1,
        "CARRYING": 2,
        "DELIVERING": 3,
        "RETURNING": 4,
        "MOVING": 5,
    }

    def __init__(self):
        import matplotlib
        matplotlib.use("TkAgg")          # ensure interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        self._plt = plt
        self._fig = None
        self._axes = None
        self._initialised = False

        # Cumulative data stored across ticks
        self._density: np.ndarray | None = None     # (rows, cols) float
        self._status_history: List[List[int]] = []   # tick → [status_code per agent]
        self._throughput: List[int] = []             # completed orders per tick

        # Distinct robot colours (up to 10 agents)
        self._robot_cmap = plt.cm.get_cmap("tab10")

        # Custom discrete colour-map for the timeline panel
        # Order: IDLE, MOVING_TO_POD, CARRYING, DELIVERING, RETURNING, MOVING
        self._timeline_cmap = ListedColormap(
            ["#1a1a2e",   # IDLE        – near-black
             "#4361ee",   # MOVING_TO_POD – blue
             "#f0a500",   # CARRYING    – gold
             "#e07c24",   # DELIVERING  – orange
             "#7b2cbf",   # RETURNING   – purple
             "#2ec4b6"]   # MOVING      – teal
        )
        self._timeline_labels = list(self._STATUS_CODES.keys())

    # ---- public API --------------------------------------------------------

    def render(self, world_state: "WorldState"):
        """Called once per tick by the simulation engine."""
        plt = self._plt

        if not self._initialised:
            self._setup(world_state)

        # Accumulate data
        self._accumulate(world_state)

        # Redraw each panel
        for ax in self._axes.flat:
            ax.clear()

        self._draw_grid(self._axes[0, 0], world_state)
        self._draw_density(self._axes[0, 1], world_state)
        self._draw_timeline(self._axes[1, 0], world_state)
        self._draw_throughput(self._axes[1, 1], world_state)

        self._fig.suptitle(
            f"MAS-RMFS  ·  Tick {world_state.tick}",
            fontsize=14, fontweight="bold", color="#e0e0e0",
        )
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.05)

    # ---- initialisation ----------------------------------------------------

    def _setup(self, world_state):
        plt = self._plt
        plt.ion()
        self._fig, self._axes = plt.subplots(
            2, 2, figsize=(14, 10),
            facecolor="#0f0f1a",
        )
        for ax in self._axes.flat:
            ax.set_facecolor("#16162a")
            ax.tick_params(colors="#aaaaaa", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333355")

        rows = world_state.map_state.rows
        cols = world_state.map_state.cols
        self._density = np.zeros((rows, cols), dtype=float)
        self._initialised = True

    # ---- data accumulation -------------------------------------------------

    def _accumulate(self, world_state):
        # Path density: +1 for every cell occupied by a robot this tick
        for agent in world_state.agents:
            r, c = agent.position
            self._density[r, c] += 1.0

        # Status history: one row per tick
        codes = []
        for agent in world_state.agents:
            codes.append(self._STATUS_CODES.get(agent.status.name, 0))
        self._status_history.append(codes)

        # Throughput: cumulative completed orders
        self._throughput.append(world_state.order_state.total_completed)

    # ---- panel drawers -----------------------------------------------------

    def _draw_grid(self, ax, world_state):
        """Top-left: live warehouse grid."""
        from WorldState.map_state import CellType

        ms = world_state.map_state
        rows, cols = ms.rows, ms.cols

        # Background cell colour matrix
        grid = np.ones((rows, cols, 3)) * np.array([0.09, 0.09, 0.16])

        for r in range(rows):
            for c in range(cols):
                cell = ms.grid[r][c]
                if cell == CellType.OBSTACLE:
                    grid[r, c] = [0.25, 0.25, 0.30]
                elif cell == CellType.POD_HOME:
                    grid[r, c] = [0.10, 0.18, 0.20]

        ax.imshow(grid, origin="upper", aspect="equal")

        # Grid lines
        for i in range(rows + 1):
            ax.axhline(i - 0.5, color="#333355", linewidth=0.5)
        for j in range(cols + 1):
            ax.axvline(j - 0.5, color="#333355", linewidth=0.5)

        # Stations (★)
        for sid, (sr, sc) in ms.station_positions.items():
            ax.plot(sc, sr, marker="*", markersize=18, color="#ff4757",
                    markeredgecolor="#ff6b81", markeredgewidth=0.8)
            ax.text(sc, sr - 0.38, f"S{sid}", ha="center", va="bottom",
                    fontsize=7, color="#ff6b81", fontweight="bold")

        # Pods at home (▲)
        for pod in world_state.pod_state.pods.values():
            if not pod.is_carried:
                pr, pc = pod.current_position
                ax.plot(pc, pr, marker="^", markersize=9, color="#2ec4b6",
                        markeredgecolor="#3dd6c8", markeredgewidth=0.6)

        # Robots (●) with ID labels
        for agent in world_state.agents:
            r, c = agent.position
            colour = self._robot_cmap(agent.agent_id % 10)

            if agent.carried_pod_id is not None:
                # Glow ring when carrying a pod
                ax.plot(c, r, "o", markersize=18, color=colour, alpha=0.25)
                ax.plot(c, r, "o", markersize=12, color=colour,
                        markeredgecolor="white", markeredgewidth=1.2)
            else:
                ax.plot(c, r, "o", markersize=12, color=colour,
                        markeredgecolor="white", markeredgewidth=0.8)

            ax.text(c + 0.32, r - 0.32, str(agent.agent_id),
                    fontsize=7, color="white", fontweight="bold")

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title("Warehouse Grid", color="#e0e0e0", fontsize=11, pad=8)

    def _draw_density(self, ax, world_state):
        """Top-right: cumulative path density heatmap."""
        im = ax.imshow(
            self._density, cmap="YlOrRd", origin="upper", aspect="equal",
            interpolation="nearest",
        )
        rows, cols = self._density.shape
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title("Path Density (cumulative)", color="#e0e0e0",
                      fontsize=11, pad=8)

        # Lightweight colour-bar (recreate on first call, update value range)
        if not hasattr(self, "_density_cb"):
            self._density_cb = self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._density_cb.ax.tick_params(colors="#aaaaaa", labelsize=7)
        else:
            self._density_cb.update_normal(im)

    def _draw_timeline(self, ax, world_state):
        """Bottom-left: agent status timeline (agent × tick)."""
        if not self._status_history:
            return

        n_agents = len(world_state.agents)
        n_ticks = len(self._status_history)

        # Build matrix: rows = agents, cols = ticks
        mat = np.zeros((n_agents, n_ticks), dtype=int)
        for t, row in enumerate(self._status_history):
            for a, code in enumerate(row):
                mat[a, t] = code

        ax.imshow(
            mat, cmap=self._timeline_cmap, aspect="auto", origin="upper",
            vmin=0, vmax=len(self._STATUS_CODES) - 1,
            interpolation="nearest",
        )

        ax.set_yticks(range(n_agents))
        ax.set_yticklabels([f"R{i}" for i in range(n_agents)])
        ax.set_xlabel("Tick", color="#aaaaaa", fontsize=9)
        ax.set_title("Agent Status Timeline", color="#e0e0e0",
                      fontsize=11, pad=8)

        # Legend (status labels)
        if not hasattr(self, "_timeline_legend_drawn"):
            from matplotlib.patches import Patch
            patches = [
                Patch(facecolor=self._timeline_cmap.colors[i], label=lbl)
                for i, lbl in enumerate(self._timeline_labels)
            ]
            ax.legend(
                handles=patches, loc="lower left", fontsize=6,
                ncol=3, framealpha=0.6, facecolor="#16162a",
                edgecolor="#333355", labelcolor="#cccccc",
            )
            self._timeline_legend_drawn = True

    def _draw_throughput(self, ax, world_state):
        """Bottom-right: cumulative completed orders."""
        ticks = list(range(len(self._throughput)))

        ax.fill_between(ticks, self._throughput, alpha=0.15, color="#2ecc71")
        ax.plot(ticks, self._throughput, color="#2ecc71", linewidth=2)

        # Numeric annotation
        current = self._throughput[-1] if self._throughput else 0
        rate = current / max(world_state.tick, 1)
        ax.text(
            0.98, 0.92,
            f"Completed: {current}\nRate: {rate:.2f} orders/tick",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#2ecc71", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#16162a",
                      edgecolor="#2ecc71", alpha=0.8),
        )

        ax.set_xlabel("Tick", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Completed Orders", color="#aaaaaa", fontsize=9)
        ax.set_title("Throughput", color="#e0e0e0", fontsize=11, pad=8)
