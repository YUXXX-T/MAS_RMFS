"""
Trajectory Replay Visualizer
==============================
独立的轨迹回放可视化工具。加载 .traj.json.gz 文件并以动画形式回放。

Standalone trajectory replay visualizer. Loads a .traj.json.gz file
and replays robot movements with an animated matplotlib dashboard.

用法:
    python -m Visualization.trajectory_visualizer <file.traj.json.gz>
    python -m Visualization.trajectory_visualizer <file.traj.json.gz> --speed 5 --light

操作:
    空格键      播放 / 暂停
    左/右箭头   上一帧 / 下一帧（暂停状态下）
    上/下箭头   加速 / 减速
    Home/End    跳转到第一帧 / 最后一帧
"""

import argparse
import os
import sys

# 确保项目根目录在 sys.path 中，支持直接运行本脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from TrajectoryRecord.trajectory_recorder import TrajectoryData


# Agent status -> color code
_STATUS_CODES = {
    "IDLE": 0,
    "MOVING_TO_POD": 1,
    "CARRYING": 2,
    "DELIVERING": 3,
    "RETURNING": 4,
    "MOVING": 5,
    "QUEUING": 6,
    "EXITING": 7,
}

_STATUS_LABELS = list(_STATUS_CODES.keys())


class TrajectoryPlayer:
    """Animated trajectory replay with playback controls."""

    def __init__(self, data: TrajectoryData, night_mode: bool = True,
                 initial_speed: int = 1, trail_length: int = 20):
        self.data = data
        self.night_mode = night_mode
        self.speed = max(1, initial_speed)
        self.trail_length = trail_length

        self.current_frame = 0
        self.playing = False
        self.total_frames = len(data.frames)

        # Theme
        if night_mode:
            self._bg = "#0f0f1a"
            self._ax_bg = "#16162a"
            self._tick_clr = "#aaaaaa"
            self._spine_clr = "#333355"
            self._title_clr = "#e0e0e0"
            self._grid_base = [0.09, 0.09, 0.16]
            self._grid_obs = [0.25, 0.25, 0.30]
            self._grid_pod_home = [0.10, 0.18, 0.20]
            self._grid_line_clr = "#333355"
            self._slider_clr = "#4361ee"
            self._idle_clr = "#1a1a2e"
            self._info_clr = "#cccccc"
        else:
            self._bg = "#f5f5f8"
            self._ax_bg = "#ffffff"
            self._tick_clr = "#333333"
            self._spine_clr = "#bbbbcc"
            self._title_clr = "#222222"
            self._grid_base = [0.92, 0.92, 0.95]
            self._grid_obs = [0.60, 0.60, 0.65]
            self._grid_pod_home = [0.80, 0.90, 0.92]
            self._grid_line_clr = "#bbbbcc"
            self._slider_clr = "#4361ee"
            self._idle_clr = "#dddde8"
            self._info_clr = "#333333"

        self._timeline_cmap = ListedColormap([
            self._idle_clr, "#4361ee", "#f0a500",
            "#e07c24", "#7b2cbf", "#2ec4b6",
            "#f59e0b", "#6b7280",
        ])

        self._robot_cmap = plt.cm.get_cmap("tab10")

        # Pre-compute density and trajectories per agent
        self._all_positions = {}  # agent_id -> list of (row, col)
        for frame in data.frames:
            for agent in frame["agents"]:
                aid = agent["id"]
                if aid not in self._all_positions:
                    self._all_positions[aid] = []
                self._all_positions[aid].append(tuple(agent["pos"]))

        # Build static map grid image
        self._build_base_grid()

    def _build_base_grid(self):
        rows, cols = self.data.rows, self.data.cols
        self._base_grid = np.ones((rows, cols, 3)) * np.array(self._grid_base)
        for r, c in self.data.obstacles:
            self._base_grid[r, c] = self._grid_obs
        for r, c in self.data.pod_homes:
            self._base_grid[r, c] = self._grid_pod_home

    def run(self):
        self._fig = plt.figure(
            figsize=(16, 10), facecolor=self._bg,
            num="MAS-RMFS Trajectory Replay",
        )

        # Layout: left big panel (grid), right column (density + timeline)
        gs = self._fig.add_gridspec(
            3, 2, width_ratios=[1.4, 1], height_ratios=[5, 5, 1],
            hspace=0.35, wspace=0.25,
            left=0.05, right=0.95, top=0.92, bottom=0.08,
        )
        self._ax_grid = self._fig.add_subplot(gs[0:2, 0])
        self._ax_density = self._fig.add_subplot(gs[0, 1])
        self._ax_timeline = self._fig.add_subplot(gs[1, 1])

        for ax in (self._ax_grid, self._ax_density, self._ax_timeline):
            ax.set_facecolor(self._ax_bg)
            ax.tick_params(colors=self._tick_clr, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(self._spine_clr)

        # Slider
        ax_slider = self._fig.add_subplot(gs[2, :])
        ax_slider.set_facecolor(self._bg)
        self._slider = Slider(
            ax_slider, "Frame", 0, max(1, self.total_frames - 1),
            valinit=0, valstep=1,
            color=self._slider_clr,
        )
        self._slider.on_changed(self._on_slider)
        self._slider.label.set_color(self._info_clr)
        self._slider.valtext.set_color(self._info_clr)

        # Keyboard events
        self._fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Info text
        self._info_text = self._fig.text(
            0.5, 0.96, "", ha="center", va="center",
            fontsize=13, fontweight="bold", color=self._title_clr,
        )
        self._control_text = self._fig.text(
            0.5, 0.015,
            "[Space] Play/Pause    [Left/Right] Step    [Up/Down] Speed    [Home/End] Jump",
            ha="center", va="center", fontsize=8, color=self._tick_clr,
        )

        # Initial draw
        self._draw_all()
        self._timer = self._fig.canvas.new_timer(interval=100)
        self._timer.add_callback(self._on_timer)
        self._timer.start()

        plt.show()

    # ---- Drawing ---------------------------------------------------------

    def _draw_all(self):
        self._draw_grid_panel()
        self._draw_density_panel()
        self._draw_timeline_panel()
        self._update_info()
        self._slider.set_val(self.current_frame)
        self._fig.canvas.draw_idle()

    def _draw_grid_panel(self):
        ax = self._ax_grid
        ax.clear()
        ax.set_facecolor(self._ax_bg)

        rows, cols = self.data.rows, self.data.cols
        ax.imshow(self._base_grid, origin="upper", aspect="equal")

        # Grid lines (only for small maps)
        if rows <= 60 and cols <= 60:
            for i in range(rows + 1):
                ax.axhline(i - 0.5, color=self._grid_line_clr, linewidth=0.3)
            for j in range(cols + 1):
                ax.axvline(j - 0.5, color=self._grid_line_clr, linewidth=0.3)

        # Stations
        for sid, (sr, sc) in self.data.stations.items():
            ax.plot(sc, sr, marker="*", markersize=14, color="#ff4757",
                    markeredgecolor="#ff6b81", markeredgewidth=0.6)

        # Current frame agents + trails
        if self.current_frame < self.total_frames:
            frame = self.data.frames[self.current_frame]
            for agent_info in frame["agents"]:
                aid = agent_info["id"]
                r, c = agent_info["pos"]
                colour = self._robot_cmap(aid % 10)

                # Trail
                positions = self._all_positions[aid]
                trail_start = max(0, self.current_frame - self.trail_length)
                trail = positions[trail_start:self.current_frame + 1]
                if len(trail) > 1:
                    trail_cols = [p[1] for p in trail]
                    trail_rows = [p[0] for p in trail]
                    for i in range(len(trail) - 1):
                        alpha = 0.1 + 0.5 * (i / len(trail))
                        ax.plot(
                            trail_cols[i:i + 2], trail_rows[i:i + 2],
                            color=colour, alpha=alpha, linewidth=1.5,
                        )

                # Robot dot
                is_carrying = agent_info.get("pod") is not None
                if is_carrying:
                    ax.plot(c, r, "o", markersize=14, color=colour, alpha=0.25)
                    ax.plot(c, r, "o", markersize=9, color=colour,
                            markeredgecolor="white", markeredgewidth=1.0)
                else:
                    ax.plot(c, r, "o", markersize=9, color=colour,
                            markeredgecolor="white", markeredgewidth=0.7)

                # ID label (only for small agent counts)
                if self.data.num_agents <= 50:
                    ax.text(c + 0.3, r - 0.3, str(aid),
                            fontsize=6, color="white", fontweight="bold")

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        tick = self.data.frames[self.current_frame]["tick"] if self.current_frame < self.total_frames else 0
        ax.set_title(f"Warehouse Grid  (tick {tick})",
                     color=self._title_clr, fontsize=11, pad=6)

    def _draw_density_panel(self):
        ax = self._ax_density
        ax.clear()
        ax.set_facecolor(self._ax_bg)

        rows, cols = self.data.rows, self.data.cols
        density = np.zeros((rows, cols), dtype=float)
        for i in range(min(self.current_frame + 1, self.total_frames)):
            for agent in self.data.frames[i]["agents"]:
                r, c = agent["pos"]
                density[r, c] += 1.0

        ax.imshow(density, cmap="YlOrRd", origin="upper", aspect="equal",
                  interpolation="nearest")
        ax.set_title("Path Density (cumulative)",
                     color=self._title_clr, fontsize=10, pad=6)

    def _draw_timeline_panel(self):
        ax = self._ax_timeline
        ax.clear()
        ax.set_facecolor(self._ax_bg)

        n_agents = self.data.num_agents
        n_frames = min(self.current_frame + 1, self.total_frames)
        if n_frames == 0 or n_agents == 0:
            return

        mat = np.zeros((n_agents, n_frames), dtype=int)
        for t in range(n_frames):
            for agent in self.data.frames[t]["agents"]:
                aid = agent["id"]
                if aid < n_agents:
                    mat[aid, t] = _STATUS_CODES.get(agent.get("status", "IDLE"), 0)

        ax.imshow(mat, cmap=self._timeline_cmap, aspect="auto", origin="upper",
                  vmin=0, vmax=len(_STATUS_CODES) - 1, interpolation="nearest")

        # Current frame indicator
        ax.axvline(n_frames - 1, color="white", linewidth=0.8, alpha=0.6)

        if n_agents <= 30:
            ax.set_yticks(range(n_agents))
            ax.set_yticklabels([f"R{i}" for i in range(n_agents)])
        ax.set_xlabel("Frame", color=self._tick_clr, fontsize=8)
        ax.set_title("Agent Status Timeline",
                     color=self._title_clr, fontsize=10, pad=6)

        # Legend
        patches = [
            Patch(facecolor=self._timeline_cmap.colors[i], label=lbl)
            for i, lbl in enumerate(_STATUS_LABELS)
        ]
        ax.legend(
            handles=patches, loc="lower left", fontsize=5, ncol=3,
            framealpha=0.6, facecolor=self._ax_bg,
            edgecolor=self._spine_clr, labelcolor=self._tick_clr,
        )

    def _update_info(self):
        state = "Playing" if self.playing else "Paused"
        self._info_text.set_text(
            f"MAS-RMFS Trajectory Replay  |  "
            f"Frame {self.current_frame}/{self.total_frames - 1}  |  "
            f"Speed: x{self.speed}  |  {state}"
        )

    # ---- Events ----------------------------------------------------------

    def _on_key(self, event):
        if event.key == " ":
            self.playing = not self.playing
            self._update_info()
            self._fig.canvas.draw_idle()
        elif event.key == "right" and not self.playing:
            self._step(1)
        elif event.key == "left" and not self.playing:
            self._step(-1)
        elif event.key == "up":
            self.speed = min(50, self.speed + 1)
            self._update_info()
            self._fig.canvas.draw_idle()
        elif event.key == "down":
            self.speed = max(1, self.speed - 1)
            self._update_info()
            self._fig.canvas.draw_idle()
        elif event.key == "home":
            self.current_frame = 0
            self._draw_all()
        elif event.key == "end":
            self.current_frame = self.total_frames - 1
            self._draw_all()

    def _on_slider(self, val):
        new_frame = int(val)
        if new_frame != self.current_frame:
            self.current_frame = new_frame
            self._draw_all()

    def _on_timer(self):
        if self.playing and self.current_frame < self.total_frames - 1:
            self._step(self.speed)

    def _step(self, delta):
        self.current_frame = max(0, min(self.total_frames - 1,
                                        self.current_frame + delta))
        self._draw_all()


def main():
    parser = argparse.ArgumentParser(
        description="MAS-RMFS 轨迹回放可视化工具"
    )
    parser.add_argument(
        "trajectory_file",
        type=str,
        help="轨迹数据文件路径 (.traj.json.gz)",
    )
    parser.add_argument(
        "--speed", type=int, default=1,
        help="初始播放速度（默认: 1）",
    )
    parser.add_argument(
        "--trail", type=int, default=20,
        help="轨迹尾迹长度（默认: 20 帧）",
    )
    parser.add_argument(
        "--light", action="store_true",
        help="使用浅色主题",
    )
    args = parser.parse_args()

    print(f"Loading trajectory: {args.trajectory_file}")
    data = TrajectoryData.load(args.trajectory_file)
    print(f"  Map: {data.rows}x{data.cols}")
    print(f"  Agents: {data.num_agents}")
    print(f"  Frames: {data.total_ticks}")
    print(f"  Recorded: {data.record_time}")

    player = TrajectoryPlayer(
        data,
        night_mode=not args.light,
        initial_speed=args.speed,
        trail_length=args.trail,
    )
    player.run()


if __name__ == "__main__":
    main()
