"""
Qt UI for MAS-RMFS Simulation
==============================
PyQt6 main window that embeds the Panda3D visualiser and provides
runtime controls (Play / Pause, Stop, Speed slider) plus a toggleable
charts panel (Agent Status Timeline, Path Density, Throughput).

The Panda3D render is embedded inside a QWidget using
``WindowProperties.setParentWindow()``, giving a single unified window.

Usage (from main.py)::

    ui = SimulationUI(engine, visualizer)
    ui.run()          # enters Qt event loop — replaces engine.run()
"""

import time
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QFrame, QGroupBox, QSizePolicy,
    QSplitter,
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")          # use Qt backend (not TkAgg)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Engine.simulation_engine import SimulationEngine
    from Visualization.panda3d_visualizer import Panda3DVisualizer


# ─── Status codes (same as MatplotlibVisualizer) ─────────────────────
_STATUS_CODES = {
    "IDLE": 0, "MOVING_TO_POD": 1, "CARRYING": 2,
    "DELIVERING": 3, "RETURNING": 4, "MOVING": 5,
}
_STATUS_LABELS = list(_STATUS_CODES.keys())


# ─── Styles ──────────────────────────────────────────────────────────
_DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Consolas', monospace;
}
QGroupBox {
    border: 1px solid #333355;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #aaaacc;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #2a2a4a;
    color: #e0e0e0;
    border: 1px solid #444466;
    border-radius: 5px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: bold;
    min-width: 80px;
}
QPushButton#speedPreset {
    min-width: 0px;
    padding: 6px 4px;
    font-size: 11px;
}
QPushButton:hover {
    background-color: #3a3a5a;
    border-color: #6666aa;
}
QPushButton:pressed {
    background-color: #4a4a6a;
}
QPushButton#playBtn {
    background-color: #1b4332;
    border-color: #2d6a4f;
}
QPushButton#playBtn:hover {
    background-color: #2d6a4f;
}
QPushButton#stopBtn {
    background-color: #641220;
    border-color: #a4133c;
}
QPushButton#stopBtn:hover {
    background-color: #a4133c;
}
QPushButton#chartBtn {
    background-color: #1a3a5c;
    border-color: #2a5a8c;
}
QPushButton#chartBtn:hover {
    background-color: #2a5a8c;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #333355;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6c63ff;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #8b83ff;
}
QLabel {
    color: #ccccdd;
    font-size: 12px;
}
QLabel#tickLabel {
    font-size: 22px;
    font-weight: bold;
    color: #6c63ff;
    font-family: 'Consolas', monospace;
}
QLabel#statusLabel {
    font-size: 14px;
    font-weight: bold;
    color: #2ecc71;
}
QSplitter::handle {
    background-color: #333355;
    width: 3px;
}
QSplitter::handle:vertical {
    height: 4px;
}
"""

_LIGHT_STYLE = """
QMainWindow, QWidget {
    background-color: #f0f0f5;
    color: #222233;
    font-family: 'Segoe UI', 'Consolas', monospace;
}
QGroupBox {
    border: 1px solid #ccccdd;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 14px;
    font-weight: bold;
    color: #555577;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #e0e0ee;
    color: #222233;
    border: 1px solid #bbbbcc;
    border-radius: 5px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: bold;
    min-width: 80px;
}
QPushButton#speedPreset {
    min-width: 0px;
    padding: 6px 4px;
    font-size: 11px;
}
QPushButton:hover {
    background-color: #d0d0e4;
    border-color: #8888aa;
}
QPushButton:pressed {
    background-color: #c0c0d8;
}
QPushButton#playBtn {
    background-color: #d4edda;
    border-color: #28a745;
    color: #155724;
}
QPushButton#playBtn:hover {
    background-color: #c3e6cb;
}
QPushButton#stopBtn {
    background-color: #f8d7da;
    border-color: #dc3545;
    color: #721c24;
}
QPushButton#stopBtn:hover {
    background-color: #f5c6cb;
}
QPushButton#chartBtn {
    background-color: #d0e8f7;
    border-color: #4a90d9;
    color: #1a4a7a;
}
QPushButton#chartBtn:hover {
    background-color: #bddcf4;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #ccccdd;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6c63ff;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #8b83ff;
}
QLabel {
    color: #444466;
    font-size: 12px;
}
QLabel#tickLabel {
    font-size: 22px;
    font-weight: bold;
    color: #6c63ff;
    font-family: 'Consolas', monospace;
}
QLabel#statusLabel {
    font-size: 14px;
    font-weight: bold;
    color: #28a745;
}
QSplitter::handle {
    background-color: #ccccdd;
    width: 3px;
}
QSplitter::handle:vertical {
    height: 4px;
}
"""


class SimulationUI(QMainWindow):
    """
    Qt main window with embedded Panda3D render, control panel,
    and toggleable matplotlib charts (timeline, density, throughput).
    """

    def __init__(
        self,
        engine: "SimulationEngine",
        visualizer: "Panda3DVisualizer",
        night_mode: bool = True,
    ):
        self._qt_app = QApplication.instance() or QApplication([])
        super().__init__()
        self._engine = engine
        self._viz = visualizer
        self._night_mode = night_mode

        # Simulation state
        self._paused = True
        self._stopped = False
        self._tick_delay = engine.config.simulation.tick_delay
        self._last_tick_time = 0.0

        self._viz.paused = self._paused
        self._viz.stopped = self._stopped
        self._viz.tick_delay = self._tick_delay

        # Chart data
        self._density: np.ndarray | None = None
        self._status_history: list[list[int]] = []
        self._throughput: list[int] = []
        self._chart_visible = False
        self._chart_last_update = 0.0
        self._chart_inited = False

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE if night_mode else _LIGHT_STYLE)

        # Sim timer
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._sim_step)
        self._sim_timer.start(16)

        self._panda_embedded = False

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("MAS-RMFS  \u2014  Simulation")
        self.resize(1400, 900)

        # Outer vertical splitter: top = main area, bottom = charts
        self._outer_splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(self._outer_splitter)

        # ── Top area (controls + Panda3D) ──
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_layout.addWidget(inner_splitter)

        # Left panel (controls)
        left_widget = QWidget()
        left_widget.setMinimumWidth(300)
        left_widget.setMaximumWidth(420)
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Title
        title = QLabel("\U0001f916 MAS-RMFS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # ── Status section ──
        status_box = QGroupBox("Simulation")
        status_layout = QVBoxLayout(status_box)

        self._status_label = QLabel("\u23f8  PAUSED")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._status_label)

        self._tick_label = QLabel("Tick: 0")
        self._tick_label.setObjectName("tickLabel")
        self._tick_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._tick_label)

        stats_row = QHBoxLayout()
        self._orders_label = QLabel("Orders: 0 / 0")
        self._agents_label = QLabel("Agents: 0")
        stats_row.addWidget(self._orders_label)
        stats_row.addWidget(self._agents_label)
        status_layout.addLayout(stats_row)
        layout.addWidget(status_box)

        # ── Controls section ──
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)

        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("\u25b6  Play")
        self._play_btn.setObjectName("playBtn")
        self._play_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self._play_btn)

        self._stop_btn = QPushButton("\u23f9  Stop")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.clicked.connect(self._stop_sim)
        btn_row.addWidget(self._stop_btn)
        ctrl_layout.addLayout(btn_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        ctrl_layout.addWidget(sep)

        speed_label = QLabel("\u23f1  Tick Delay (seconds)")
        speed_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ctrl_layout.addWidget(speed_label)

        slider_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(0, 200)
        self._speed_slider.setValue(int(self._tick_delay * 100))
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        slider_row.addWidget(self._speed_slider)

        self._speed_value = QLabel(f"{self._tick_delay:.2f}s")
        self._speed_value.setMinimumWidth(48)
        slider_row.addWidget(self._speed_value)
        ctrl_layout.addLayout(slider_row)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for label, val in [("0.1\u00d7", 1.0), ("0.5\u00d7", 0.5), ("1\u00d7", 0.25), ("2\u00d7", 0.1), ("Max", 0.0)]:
            btn = QPushButton(label)
            btn.setObjectName("speedPreset")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _, v=val: self._set_speed(v))
            preset_row.addWidget(btn)
        ctrl_layout.addLayout(preset_row)
        layout.addWidget(ctrl_box)

        # ── Charts toggle ──
        self._chart_btn = QPushButton("\U0001f4ca  Show Charts")
        self._chart_btn.setObjectName("chartBtn")
        self._chart_btn.clicked.connect(self._toggle_charts)
        layout.addWidget(self._chart_btn)

        # ── Info section ──
        info_box = QGroupBox("Info")
        info_layout = QVBoxLayout(info_box)
        self._pods_label = QLabel("Pods: 0")
        self._completed_label = QLabel("Completed: 0")
        self._pending_label = QLabel("Pending: 0")
        self._inprogress_label = QLabel("In Progress: 0")
        info_layout.addWidget(self._pods_label)
        info_layout.addWidget(self._completed_label)
        info_layout.addWidget(self._inprogress_label)
        info_layout.addWidget(self._pending_label)
        layout.addWidget(info_box)

        layout.addStretch()

        hint = QLabel("Space: Play/Pause  |  Esc: Stop  |  C: Charts")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(hint)

        inner_splitter.addWidget(left_widget)

        # Right panel (Panda3D container)
        self._panda_container = QWidget()
        self._panda_container.setMinimumSize(600, 400)
        self._panda_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        inner_splitter.addWidget(self._panda_container)
        inner_splitter.setStretchFactor(0, 0)
        inner_splitter.setStretchFactor(1, 1)
        inner_splitter.setSizes([340, 1060])

        self._outer_splitter.addWidget(top_widget)

        # ── Bottom area: matplotlib charts panel (hidden by default) ──
        self._charts_widget = self._build_charts_widget()
        self._charts_widget.setVisible(False)
        self._outer_splitter.addWidget(self._charts_widget)
        self._outer_splitter.setStretchFactor(0, 3)
        self._outer_splitter.setStretchFactor(1, 1)

    def _build_charts_widget(self):
        """Create the matplotlib charts panel with 3 subplots."""
        nm = self._night_mode
        bg = "#0f0f1a" if nm else "#f5f5f8"
        ax_bg = "#16162a" if nm else "#ffffff"
        self._chart_tick_clr = "#aaaaaa" if nm else "#333333"
        self._chart_spine_clr = "#333355" if nm else "#bbbbcc"
        self._chart_title_clr = "#e0e0e0" if nm else "#222222"
        self._chart_ax_bg = ax_bg
        idle_clr = "#1a1a2e" if nm else "#dddde8"

        self._timeline_cmap = ListedColormap([
            idle_clr,      # IDLE
            "#4361ee",     # MOVING_TO_POD
            "#f0a500",     # CARRYING
            "#e07c24",     # DELIVERING
            "#7b2cbf",     # RETURNING
            "#2ec4b6",     # MOVING
        ])

        fig = plt.figure(figsize=(14, 3.5), facecolor=bg)
        self._chart_fig = fig
        axes = fig.subplots(1, 3)
        self._chart_axes = axes
        for ax in axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=self._chart_tick_clr, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(self._chart_spine_clr)

        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumHeight(200)
        return canvas

    # ── Chart drawing ─────────────────────────────────────────────────

    def _accumulate_chart_data(self, world_state):
        """Collect data each tick for charts."""
        if self._density is None:
            ms = world_state.map_state
            self._density = np.zeros((ms.rows, ms.cols), dtype=float)

        for agent in world_state.agents:
            r, c = agent.position
            self._density[r, c] += 1.0

        codes = [_STATUS_CODES.get(a.status.name, 0) for a in world_state.agents]
        self._status_history.append(codes)
        self._throughput.append(world_state.order_state.total_completed)

    def _redraw_charts(self, world_state):
        """Redraw all 3 chart panels."""
        for ax in self._chart_axes:
            ax.clear()
            ax.set_facecolor(self._chart_ax_bg)
            ax.tick_params(colors=self._chart_tick_clr, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(self._chart_spine_clr)

        self._draw_timeline(self._chart_axes[0], world_state)
        self._draw_density(self._chart_axes[1], world_state)
        self._draw_throughput(self._chart_axes[2], world_state)

        self._chart_fig.tight_layout(pad=1.5)
        self._charts_widget.draw_idle()

    def _draw_timeline(self, ax, world_state):
        if not self._status_history:
            return
        n_agents = len(world_state.agents)
        n_ticks = len(self._status_history)
        mat = np.zeros((n_agents, n_ticks), dtype=int)
        for t, row in enumerate(self._status_history):
            for a, code in enumerate(row):
                mat[a, t] = code

        ax.imshow(mat, cmap=self._timeline_cmap, aspect="auto",
                  origin="upper", vmin=0, vmax=5, interpolation="nearest")
        ax.set_yticks(range(n_agents))
        ax.set_yticklabels([f"R{i}" for i in range(n_agents)])
        ax.set_xlabel("Tick", color=self._chart_tick_clr, fontsize=8)
        ax.set_title("Agent Status Timeline", color=self._chart_title_clr,
                      fontsize=10, pad=6)

        patches = [Patch(facecolor=self._timeline_cmap.colors[i], label=lbl)
                   for i, lbl in enumerate(_STATUS_LABELS)]
        ax.legend(handles=patches, loc="lower left", fontsize=5, ncol=3,
                  framealpha=0.6, facecolor=self._chart_ax_bg,
                  edgecolor=self._chart_spine_clr,
                  labelcolor=self._chart_tick_clr)

    def _draw_density(self, ax, world_state):
        if self._density is None:
            return
        ax.imshow(self._density, cmap="YlOrRd", origin="upper",
                  aspect="equal", interpolation="nearest")
        rows, cols = self._density.shape
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title("Path Density (cumulative)", color=self._chart_title_clr,
                      fontsize=10, pad=6)

    def _draw_throughput(self, ax, world_state):
        ticks = list(range(len(self._throughput)))
        ax.fill_between(ticks, self._throughput, alpha=0.15, color="#2ecc71")
        ax.plot(ticks, self._throughput, color="#2ecc71", linewidth=2)

        current = self._throughput[-1] if self._throughput else 0
        rate = current / max(world_state.tick, 1)
        ax.text(0.98, 0.92, f"Completed: {current}\nRate: {rate:.2f}/tick",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#2ecc71", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self._chart_ax_bg,
                          edgecolor="#2ecc71", alpha=0.8))

        ax.set_xlabel("Tick", color=self._chart_tick_clr, fontsize=8)
        ax.set_ylabel("Completed Orders", color=self._chart_tick_clr, fontsize=8)
        ax.set_title("Throughput", color=self._chart_title_clr, fontsize=10, pad=6)

    # ── Embed Panda3D ─────────────────────────────────────────────────

    def _embed_panda(self):
        handle = int(self._panda_container.winId())
        self._viz._parent_window_handle = handle
        self._panda_embedded = True

    def _resize_panda(self):
        if (self._viz._app is None or self._viz._app.win is None
                or not self._panda_embedded):
            return
        from panda3d.core import WindowProperties
        w = self._panda_container.width()
        h = self._panda_container.height()
        if w > 0 and h > 0:
            wp = WindowProperties()
            wp.setSize(w, h)
            wp.setOrigin(0, 0)
            self._viz._app.win.requestProperties(wp)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_panda()

    # ── Simulation loop step ──────────────────────────────────────────

    def _sim_step(self):
        if self._stopped:
            return

        now = time.time()

        if not self._paused:
            if now - self._last_tick_time >= self._tick_delay:
                try:
                    self._engine._tick()
                except Exception as e:
                    self._status_label.setText(f"\u274c  ERROR: {e}")
                    self._paused = True
                    return
                self._last_tick_time = now

                # Accumulate chart data every tick
                self._accumulate_chart_data(self._engine.world)

                # Redraw charts at ~2 fps (every 500ms) if visible
                if self._chart_visible and now - self._chart_last_update > 0.5:
                    self._redraw_charts(self._engine.world)
                    self._chart_last_update = now

        # Pump Panda3D render
        if self._viz._initialised:
            self._viz._update_pods(self._engine.world)
            self._viz._update_agents(self._engine.world)
            self._viz._update_hud(self._engine.world)
            self._viz._app.taskMgr.step()
        elif not self._paused or not self._panda_embedded:
            if not self._panda_embedded:
                self._embed_panda()
            self._viz.render(self._engine.world)
            QTimer.singleShot(100, self._resize_panda)

        self._update_info()

    def _update_info(self):
        ws = self._engine.world
        self._tick_label.setText(f"Tick: {ws.tick}")
        total = ws.order_state.total_orders
        done = ws.order_state.total_completed
        self._orders_label.setText(f"Orders: {done} / {total}")
        self._agents_label.setText(f"Agents: {len(ws.agents)}")
        self._pods_label.setText(f"Pods: {ws.pod_state.total_pods}")
        self._completed_label.setText(f"Completed: {done}")

        pending = sum(1 for o in ws.order_state.orders.values()
                      if o.status.name == "PENDING")
        in_prog = sum(1 for o in ws.order_state.orders.values()
                      if o.status.name == "IN_PROGRESS")
        self._pending_label.setText(f"Pending: {pending}")
        self._inprogress_label.setText(f"In Progress: {in_prog}")

    # ── Button handlers ───────────────────────────────────────────────

    def _toggle_pause(self):
        self._paused = not self._paused
        self._viz.paused = self._paused
        if self._paused:
            self._play_btn.setText("\u25b6  Play")
            self._status_label.setText("\u23f8  PAUSED")
            self._status_label.setStyleSheet(
                "color: #f0a500; font-size: 14px; font-weight: bold;")
        else:
            self._play_btn.setText("\u23f8  Pause")
            self._status_label.setText("\u25b6  RUNNING")
            self._status_label.setStyleSheet(
                "color: #2ecc71; font-size: 14px; font-weight: bold;")
            self._last_tick_time = time.time()

    def _stop_sim(self):
        self._stopped = True
        self._paused = True
        self._viz.stopped = True
        self._viz.paused = True
        self._sim_timer.stop()
        self._play_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("\u23f9  STOPPED")
        self._status_label.setStyleSheet(
            "color: #e74c3c; font-size: 14px; font-weight: bold;")
        self._engine._print_summary()

        # Final chart refresh
        if self._chart_visible and self._status_history:
            self._redraw_charts(self._engine.world)

    def _on_speed_change(self, value):
        self._tick_delay = value / 100.0
        self._viz.tick_delay = self._tick_delay
        self._speed_value.setText(f"{self._tick_delay:.2f}s")

    def _set_speed(self, delay: float):
        self._tick_delay = delay
        self._viz.tick_delay = delay
        self._speed_slider.setValue(int(delay * 100))
        self._speed_value.setText(f"{delay:.2f}s")

    def _toggle_charts(self):
        self._chart_visible = not self._chart_visible
        self._charts_widget.setVisible(self._chart_visible)
        if self._chart_visible:
            self._chart_btn.setText("\U0001f4ca  Hide Charts")
            # Immediately redraw with current data
            if self._status_history:
                self._redraw_charts(self._engine.world)
        else:
            self._chart_btn.setText("\U0001f4ca  Show Charts")

    # ── Keyboard shortcuts ────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._toggle_pause()
        elif event.key() == Qt.Key.Key_Escape:
            self._stop_sim()
        elif event.key() == Qt.Key.Key_C:
            self._toggle_charts()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        if not self._stopped:
            self._stop_sim()
        event.accept()

    # ── Public API ────────────────────────────────────────────────────

    def run(self):
        self._engine.logger.info("=" * 60)
        self._engine.logger.info("MAS-RMFS Simulation Started (Qt UI)")
        ws = self._engine.world
        self._engine.logger.info(f"  Map: {ws.map_state.rows}x{ws.map_state.cols}")
        self._engine.logger.info(f"  Agents: {len(ws.agents)}")
        self._engine.logger.info(f"  Pods: {ws.pod_state.total_pods}")
        self._engine.logger.info(f"  Stations: {len(ws.map_state.station_positions)}")
        self._engine.logger.info("=" * 60)

        self.show()
        self._qt_app.exec()
