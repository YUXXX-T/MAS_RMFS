"""
MAS-RMFS 仿真的 Qt 用户界面
==============================
PyQt6 主窗口，内嵌 Panda3D 可视化器并提供
运行时控件（播放/暂停、停止、速度滑块）以及可切换的
图表面板（智能体状态时间线、路径密度、吞吐量）。

Panda3D 渲染通过以下方式嵌入 QWidget：
``WindowProperties.setParentWindow()``, 提供单个统一窗口。

用法（from main.py）：：

    ui = SimulationUI(engine, visualizer)
    ui.run()          # 进入 Qt 事件循环 — 替代 engine.run()
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
matplotlib.use("QtAgg")          # 使用 Qt 后端（非 TkAgg）
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Engine.simulation_engine import SimulationEngine
    from Visualization.panda3d_visualizer import Panda3DVisualizer
    from TrajectoryRecord.trajectory_recorder import TrajectoryData


# ─── 状态编码（与 MatplotlibVisualizer 相同） ─────────────────────
_STATUS_CODES = {
    "IDLE": 0, "MOVING_TO_POD": 1, "CARRYING": 2,
    "DELIVERING": 3, "RETURNING": 4, "MOVING": 5,
}
_STATUS_LABELS = list(_STATUS_CODES.keys())


# ─── 样式 ──────────────────────────────────────────────────────────
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

        # 仿真状态
        self._paused = True
        self._stopped = False
        self._tick_delay = engine.config.simulation.tick_delay
        self._last_tick_time = 0.0

        self._viz.paused = self._paused
        self._viz.stopped = self._stopped
        self._viz.tick_delay = self._tick_delay

        # 图表数据
        self._density: np.ndarray | None = None
        self._status_history: list[list[int]] = []
        self._throughput: list[int] = []
        self._chart_visible = False
        self._chart_last_update = 0.0
        self._chart_inited = False

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE if night_mode else _LIGHT_STYLE)

        # 仿真定时器
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._sim_step)
        self._sim_timer.start(16)

        self._panda_embedded = False

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("MAS-RMFS  \u2014  Simulation")
        self.resize(1400, 900)

        # 水平分割器：左 = 控件，右 = Panda3D
        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(inner_splitter)

        # 左侧面板（控件）
        left_widget = QWidget()
        left_widget.setMinimumWidth(300)
        left_widget.setMaximumWidth(420)
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 标题
        title = QLabel("\U0001f916 MAS-RMFS")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # ── 状态区域 ──
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

        # ── 控件区域 ──
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

        # ── 图表切换 ──
        self._chart_btn = QPushButton("\U0001f4ca  Show Charts")
        self._chart_btn.setObjectName("chartBtn")
        self._chart_btn.clicked.connect(self._toggle_charts)
        layout.addWidget(self._chart_btn)

        # ── 信息区域 ──
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

        # 右侧面板：垂直分割器（Panda3D + 图表）
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setMinimumSize(600, 400)

        # Panda3D 容器
        self._panda_container = QWidget()
        self._panda_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._right_splitter.addWidget(self._panda_container)

        # 嵌入式图表面板（默认隐藏）
        self._charts_panel = self._build_charts_panel()
        self._charts_panel.setVisible(False)
        self._right_splitter.addWidget(self._charts_panel)
        self._right_splitter.setStretchFactor(0, 3)
        self._right_splitter.setStretchFactor(1, 1)
        # 图表显示/隐藏时重新调整 Panda3D 大小
        self._right_splitter.splitterMoved.connect(lambda *_: self._resize_panda())

        inner_splitter.addWidget(self._right_splitter)
        inner_splitter.setStretchFactor(0, 0)
        inner_splitter.setStretchFactor(1, 1)
        inner_splitter.setSizes([340, 1060])

    def _build_charts_panel(self):
        """创建嵌入式 matplotlib 图表面板。"""
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

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)

        fig = plt.figure(figsize=(14, 3.5), facecolor=bg)
        self._chart_fig = fig
        axes = fig.subplots(1, 3)
        self._chart_axes = axes
        for ax in axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=self._chart_tick_clr, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(self._chart_spine_clr)

        self._charts_canvas = FigureCanvasQTAgg(fig)
        self._charts_canvas.setMinimumHeight(150)
        panel_layout.addWidget(self._charts_canvas)
        return panel

    # ── 图表绘制 ─────────────────────────────────────────────────

    def _accumulate_chart_data(self, world_state):
        """每个 tick 为图表收集数据。"""
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
        """重绘全部 3 个图表面板。"""
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
        self._charts_canvas.draw_idle()

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

    # ── 嵌入 Panda3D ─────────────────────────────────────────────────

    def _embed_panda(self):
        handle = int(self._panda_container.winId())
        self._viz._parent_window_handle = handle
        # 传递容器初始物理像素尺寸给 Panda3D（考虑 DPI 缩放）
        dpr = self._panda_container.devicePixelRatio()
        self._viz._parent_initial_size = (
            int(self._panda_container.width() * dpr),
            int(self._panda_container.height() * dpr),
        )
        self._panda_embedded = True

    def _resize_panda(self):
        if (self._viz._app is None or self._viz._app.win is None
                or not self._panda_embedded):
            return
        from panda3d.core import WindowProperties
        # 使用物理像素（考虑 DPI 缩放）
        dpr = self._panda_container.devicePixelRatio()
        w = int(self._panda_container.width() * dpr)
        h = int(self._panda_container.height() * dpr)
        if w > 0 and h > 0:
            wp = WindowProperties()
            wp.setSize(w, h)
            wp.setOrigin(0, 0)
            self._viz._app.win.requestProperties(wp)

            # 通知 ShowBase 更新宽高比（镜头、aspect2d、pixel2d 等）
            self._viz._app.adjustWindowAspectRatio(w / h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_panda()

    # ── 仿真循环步骤 ──────────────────────────────────────────

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

                # 每个 tick 累积图表数据
                self._accumulate_chart_data(self._engine.world)

                # 图表可见时以约 2fps（每 500ms）重绘
                if self._chart_visible and now - self._chart_last_update > 0.5:
                    self._redraw_charts(self._engine.world)
                    self._chart_last_update = now

        # 驱动 Panda3D 渲染
        if self._viz._initialised:
            self._viz._update_pods(self._engine.world)
            self._viz._update_agents(self._engine.world)
            self._viz._update_hud(self._engine.world)
            self._viz._app.taskMgr.step()
        elif not self._paused or not self._panda_embedded:
            if not self._panda_embedded:
                self._embed_panda()
            self._viz.render(self._engine.world)
            # 多次延迟强制同步尺寸，确保窗口完全填满容器
            for delay in (50, 200, 500):
                QTimer.singleShot(delay, self._resize_panda)

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

    # ── 按钮处理器 ───────────────────────────────────────────────

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

        # 最终图表刷新
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
        self._charts_panel.setVisible(self._chart_visible)
        if self._chart_visible:
            self._chart_btn.setText("\U0001f4ca  Hide Charts")
            # 立即用当前数据重绘
            if self._status_history:
                self._redraw_charts(self._engine.world)
        else:
            self._chart_btn.setText("\U0001f4ca  Show Charts")
        # 图表显示/隐藏后重新调整 Panda3D 大小
        QTimer.singleShot(50, self._resize_panda)

    # ── 键盘快捷键 ────────────────────────────────────────────

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

    # ── 公共 API ────────────────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════
# 轨迹回放 UI
# ═══════════════════════════════════════════════════════════════════════


class ReplayUI(QMainWindow):
    """
    Qt 主窗口，内嵌 Panda3D 回放轨迹数据。

    提供播放/暂停、单步、速度控制、帧滑块等播放控件，
    以及可切换的 matplotlib 图表面板。

    用法（from main.py）：
        ui = ReplayUI(trajectory_data, visualizer, night_mode=True)
        ui.run()
    """

    def __init__(
        self,
        data: "TrajectoryData",
        visualizer: "Panda3DVisualizer",
        night_mode: bool = True,
        initial_fps: int = 10,
    ):
        self._qt_app = QApplication.instance() or QApplication([])
        super().__init__()
        self._data = data
        self._viz = visualizer
        self._night_mode = night_mode
        self._initial_fps = max(1, min(60, initial_fps))

        # 回放状态
        from TrajectoryRecord.replay_state import ReplayWorldState
        self._replay_world = ReplayWorldState(data)
        self._paused = True
        self._speed = 1  # 每次前进的帧数
        self._frame_interval = 1.0 / self._initial_fps  # 每帧间隔 (秒)
        self._last_advance_time = 0.0

        self._viz.paused = True
        self._viz.stopped = False
        self._viz.tick_delay = self._frame_interval

        # 图表数据
        self._density: np.ndarray | None = None
        self._status_history: list[list[int]] = []
        self._chart_visible = False
        self._chart_last_frame = -1

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE if night_mode else _LIGHT_STYLE)

        # 定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(16)

        self._panda_embedded = False

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("MAS-RMFS  \u2014  Trajectory Replay")
        self.resize(1400, 900)

        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(inner_splitter)

        # 左侧面板
        left_widget = QWidget()
        left_widget.setMinimumWidth(300)
        left_widget.setMaximumWidth(420)
        layout = QVBoxLayout(left_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("\U0001f3ac MAS-RMFS Replay")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # ── 回放状态 ──
        status_box = QGroupBox("Playback")
        status_layout = QVBoxLayout(status_box)

        self._status_label = QLabel("\u23f8  PAUSED")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._status_label)

        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setObjectName("tickLabel")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._frame_label)

        info_row = QHBoxLayout()
        self._tick_label = QLabel("Tick: 0")
        self._agents_label = QLabel(f"Agents: {self._data.num_agents}")
        info_row.addWidget(self._tick_label)
        info_row.addWidget(self._agents_label)
        status_layout.addLayout(info_row)
        layout.addWidget(status_box)

        # ── 控件 ──
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)

        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("\u25b6  Play")
        self._play_btn.setObjectName("playBtn")
        self._play_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self._play_btn)

        self._prev_btn = QPushButton("\u23ee  Prev")
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        btn_row.addWidget(self._prev_btn)

        self._next_btn = QPushButton("Next  \u23ed")
        self._next_btn.clicked.connect(lambda: self._step(1))
        btn_row.addWidget(self._next_btn)
        ctrl_layout.addLayout(btn_row)

        # 帧滑块
        slider_label = QLabel("\U0001f39e  Frame")
        slider_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ctrl_layout.addWidget(slider_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, self._data.total_ticks - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        ctrl_layout.addWidget(self._frame_slider)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        ctrl_layout.addWidget(sep)

        # 帧间隔控制 (FPS)
        fps_label = QLabel("\u23f1  Playback FPS")
        fps_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ctrl_layout.addWidget(fps_label)

        fps_row = QHBoxLayout()
        self._fps_slider = QSlider(Qt.Orientation.Horizontal)
        self._fps_slider.setRange(1, 60)
        self._fps_slider.setValue(self._initial_fps)
        self._fps_slider.valueChanged.connect(self._on_fps_change)
        fps_row.addWidget(self._fps_slider)

        self._fps_value = QLabel(f"{self._initial_fps} fps")
        self._fps_value.setMinimumWidth(55)
        fps_row.addWidget(self._fps_value)
        ctrl_layout.addLayout(fps_row)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for label, val in [("2fps", 2), ("5fps", 5), ("10fps", 10), ("30fps", 30), ("60fps", 60)]:
            btn = QPushButton(label)
            btn.setObjectName("speedPreset")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _, v=val: self._set_fps(v))
            preset_row.addWidget(btn)
        ctrl_layout.addLayout(preset_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        ctrl_layout.addWidget(sep2)

        # 步长控制 (每帧前进多少 frame)
        step_label = QLabel("\u23e9  Step Size")
        step_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ctrl_layout.addWidget(step_label)

        step_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 50)
        self._speed_slider.setValue(1)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        step_row.addWidget(self._speed_slider)

        self._speed_value = QLabel("x1")
        self._speed_value.setMinimumWidth(40)
        step_row.addWidget(self._speed_value)
        ctrl_layout.addLayout(step_row)
        layout.addWidget(ctrl_box)

        # 图表切换
        self._chart_btn = QPushButton("\U0001f4ca  Show Charts")
        self._chart_btn.setObjectName("chartBtn")
        self._chart_btn.clicked.connect(self._toggle_charts)
        layout.addWidget(self._chart_btn)

        # 信息区域
        info_box = QGroupBox("Info")
        info_layout = QVBoxLayout(info_box)
        self._map_label = QLabel(f"Map: {self._data.rows}x{self._data.cols}")
        self._pods_label = QLabel(f"Pods: {len(self._data.pod_homes)}")
        self._stations_label = QLabel(f"Stations: {len(self._data.stations)}")
        self._recorded_label = QLabel(f"Recorded: {self._data.record_time}")
        info_layout.addWidget(self._map_label)
        info_layout.addWidget(self._pods_label)
        info_layout.addWidget(self._stations_label)
        info_layout.addWidget(self._recorded_label)
        layout.addWidget(info_box)

        layout.addStretch()

        hint = QLabel("Space: Play/Pause  |  Left/Right: Step  |  C: Charts")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(hint)

        inner_splitter.addWidget(left_widget)

        # 右侧面板
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setMinimumSize(600, 400)

        # Panda3D 容器
        self._panda_container = QWidget()
        self._panda_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._right_splitter.addWidget(self._panda_container)

        # 图表面板
        self._charts_panel = self._build_charts_panel()
        self._charts_panel.setVisible(False)
        self._right_splitter.addWidget(self._charts_panel)
        self._right_splitter.setStretchFactor(0, 3)
        self._right_splitter.setStretchFactor(1, 1)
        self._right_splitter.splitterMoved.connect(lambda *_: self._resize_panda())

        inner_splitter.addWidget(self._right_splitter)
        inner_splitter.setStretchFactor(0, 0)
        inner_splitter.setStretchFactor(1, 1)
        inner_splitter.setSizes([340, 1060])

    def _build_charts_panel(self):
        nm = self._night_mode
        bg = "#0f0f1a" if nm else "#f5f5f8"
        ax_bg = "#16162a" if nm else "#ffffff"
        self._chart_tick_clr = "#aaaaaa" if nm else "#333333"
        self._chart_spine_clr = "#333355" if nm else "#bbbbcc"
        self._chart_title_clr = "#e0e0e0" if nm else "#222222"
        self._chart_ax_bg = ax_bg
        idle_clr = "#1a1a2e" if nm else "#dddde8"

        self._timeline_cmap = ListedColormap([
            idle_clr, "#4361ee", "#f0a500",
            "#e07c24", "#7b2cbf", "#2ec4b6",
        ])

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)

        fig = plt.figure(figsize=(14, 3.5), facecolor=bg)
        self._chart_fig = fig
        axes = fig.subplots(1, 2)
        self._chart_axes = axes
        for ax in axes:
            ax.set_facecolor(ax_bg)
            ax.tick_params(colors=self._chart_tick_clr, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(self._chart_spine_clr)

        self._charts_canvas = FigureCanvasQTAgg(fig)
        self._charts_canvas.setMinimumHeight(150)
        panel_layout.addWidget(self._charts_canvas)
        return panel

    # ── 图表绘制 ─────────────────────────────────────────────────

    def _compute_chart_data(self):
        """按当前帧计算密度和时间线数据。"""
        idx = self._replay_world.current_frame_index
        if idx == self._chart_last_frame:
            return
        self._chart_last_frame = idx

        rows, cols = self._data.rows, self._data.cols
        self._density = np.zeros((rows, cols), dtype=float)
        self._status_history = []
        for i in range(min(idx + 1, len(self._data.frames))):
            frame = self._data.frames[i]
            codes = []
            for agent in frame["agents"]:
                r, c = agent["pos"]
                self._density[r, c] += 1.0
                codes.append(_STATUS_CODES.get(agent.get("status", "IDLE"), 0))
            self._status_history.append(codes)

    def _redraw_charts(self):
        self._compute_chart_data()
        for ax in self._chart_axes:
            ax.clear()
            ax.set_facecolor(self._chart_ax_bg)
            ax.tick_params(colors=self._chart_tick_clr, labelsize=7)
            for sp in ax.spines.values():
                sp.set_color(self._chart_spine_clr)

        self._draw_density(self._chart_axes[0])
        self._draw_timeline(self._chart_axes[1])

        self._chart_fig.tight_layout(pad=1.5)
        self._charts_canvas.draw_idle()

    def _draw_density(self, ax):
        if self._density is None:
            return
        ax.imshow(self._density, cmap="YlOrRd", origin="upper",
                  aspect="equal", interpolation="nearest")
        ax.set_title("Path Density (cumulative)", color=self._chart_title_clr,
                      fontsize=10, pad=6)

    def _draw_timeline(self, ax):
        if not self._status_history:
            return
        n_agents = self._data.num_agents
        n_ticks = len(self._status_history)
        mat = np.zeros((n_agents, n_ticks), dtype=int)
        for t, row in enumerate(self._status_history):
            for a, code in enumerate(row):
                if a < n_agents:
                    mat[a, t] = code

        ax.imshow(mat, cmap=self._timeline_cmap, aspect="auto",
                  origin="upper", vmin=0, vmax=5, interpolation="nearest")

        # 当前帧指示线
        ax.axvline(n_ticks - 1, color="white", linewidth=0.8, alpha=0.6)

        if n_agents <= 30:
            ax.set_yticks(range(n_agents))
            ax.set_yticklabels([f"R{i}" for i in range(n_agents)])
        ax.set_xlabel("Frame", color=self._chart_tick_clr, fontsize=8)
        ax.set_title("Agent Status Timeline", color=self._chart_title_clr,
                      fontsize=10, pad=6)

        patches = [Patch(facecolor=self._timeline_cmap.colors[i], label=lbl)
                   for i, lbl in enumerate(_STATUS_LABELS)]
        ax.legend(handles=patches, loc="lower left", fontsize=5, ncol=3,
                  framealpha=0.6, facecolor=self._chart_ax_bg,
                  edgecolor=self._chart_spine_clr,
                  labelcolor=self._chart_tick_clr)

    # ── 嵌入 Panda3D ─────────────────────────────────────────────────

    def _embed_panda(self):
        handle = int(self._panda_container.winId())
        self._viz._parent_window_handle = handle
        dpr = self._panda_container.devicePixelRatio()
        self._viz._parent_initial_size = (
            int(self._panda_container.width() * dpr),
            int(self._panda_container.height() * dpr),
        )
        self._panda_embedded = True

    def _resize_panda(self):
        if (self._viz._app is None or self._viz._app.win is None
                or not self._panda_embedded):
            return
        from panda3d.core import WindowProperties
        dpr = self._panda_container.devicePixelRatio()
        w = int(self._panda_container.width() * dpr)
        h = int(self._panda_container.height() * dpr)
        if w > 0 and h > 0:
            wp = WindowProperties()
            wp.setSize(w, h)
            wp.setOrigin(0, 0)
            self._viz._app.win.requestProperties(wp)
            self._viz._app.adjustWindowAspectRatio(w / h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_panda()

    # ── 回放逻辑 ──────────────────────────────────────────────────

    def _on_timer(self):
        ws = self._replay_world

        # 首次初始化 Panda3D
        if not self._viz._initialised:
            if not self._panda_embedded:
                self._embed_panda()
            ws.set_frame(0)
            self._viz.render(ws)
            for delay in (50, 200, 500):
                QTimer.singleShot(delay, self._resize_panda)
            self._update_info()
            return

        now = time.time()

        # 自动播放：按帧间隔节流
        if not self._paused and now - self._last_advance_time >= self._frame_interval:
            new_frame = ws.current_frame_index + self._speed
            if new_frame >= ws.total_frames:
                new_frame = ws.total_frames - 1
                self._paused = True
                self._play_btn.setText("\u25b6  Play")
                self._status_label.setText("\u23f9  END")
                self._status_label.setStyleSheet(
                    "color: #e74c3c; font-size: 14px; font-weight: bold;")
            ws.set_frame(new_frame)
            self._last_advance_time = now

        # 驱动 Panda3D
        self._viz._update_pods(ws)
        self._viz._update_agents(ws)
        self._viz._update_hud(ws)
        self._viz._app.taskMgr.step()

        self._update_info()

        # 图表
        if self._chart_visible:
            self._redraw_charts()

    def _step(self, delta: int):
        """手动步进帧。"""
        ws = self._replay_world
        new_frame = max(0, min(ws.total_frames - 1,
                               ws.current_frame_index + delta))
        ws.set_frame(new_frame)
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(new_frame)
        self._frame_slider.blockSignals(False)

    def _update_info(self):
        ws = self._replay_world
        idx = ws.current_frame_index
        self._frame_label.setText(f"Frame: {idx} / {ws.total_frames - 1}")
        self._tick_label.setText(f"Tick: {ws.tick}")
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)

    # ── 控件回调 ──────────────────────────────────────────────────

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._play_btn.setText("\u25b6  Play")
            self._status_label.setText("\u23f8  PAUSED")
            self._status_label.setStyleSheet(
                "color: #f0a500; font-size: 14px; font-weight: bold;")
        else:
            # 如果已到末尾，重头开始
            ws = self._replay_world
            if ws.current_frame_index >= ws.total_frames - 1:
                ws.set_frame(0)
            self._play_btn.setText("\u23f8  Pause")
            self._status_label.setText("\u25b6  PLAYING")
            self._status_label.setStyleSheet(
                "color: #2ecc71; font-size: 14px; font-weight: bold;")
            self._last_advance_time = time.time()

    def _on_frame_slider(self, value):
        self._replay_world.set_frame(value)

    def _on_fps_change(self, value):
        self._frame_interval = 1.0 / value
        self._fps_value.setText(f"{value} fps")

    def _set_fps(self, fps: int):
        self._frame_interval = 1.0 / fps
        self._fps_slider.setValue(fps)
        self._fps_value.setText(f"{fps} fps")

    def _on_speed_change(self, value):
        self._speed = value
        self._speed_value.setText(f"x{value}")

    def _set_speed(self, speed: int):
        self._speed = speed
        self._speed_slider.setValue(speed)
        self._speed_value.setText(f"x{speed}")

    def _toggle_charts(self):
        self._chart_visible = not self._chart_visible
        self._charts_panel.setVisible(self._chart_visible)
        if self._chart_visible:
            self._chart_btn.setText("\U0001f4ca  Hide Charts")
            self._chart_last_frame = -1  # 强制重算
            self._redraw_charts()
        else:
            self._chart_btn.setText("\U0001f4ca  Show Charts")
        QTimer.singleShot(50, self._resize_panda)

    # ── 键盘 ─────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._toggle_pause()
        elif event.key() == Qt.Key.Key_Left:
            self._step(-1)
        elif event.key() == Qt.Key.Key_Right:
            self._step(1)
        elif event.key() == Qt.Key.Key_Up:
            self._set_speed(min(50, self._speed + 1))
        elif event.key() == Qt.Key.Key_Down:
            self._set_speed(max(1, self._speed - 1))
        elif event.key() == Qt.Key.Key_Home:
            self._replay_world.set_frame(0)
        elif event.key() == Qt.Key.Key_End:
            self._replay_world.set_frame(self._replay_world.total_frames - 1)
        elif event.key() == Qt.Key.Key_C:
            self._toggle_charts()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        event.accept()

    # ── 公共 API ────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("MAS-RMFS Trajectory Replay (Panda3D)")
        print(f"  Map: {self._data.rows}x{self._data.cols}")
        print(f"  Agents: {self._data.num_agents}")
        print(f"  Frames: {self._data.total_ticks}")
        print(f"  Recorded: {self._data.record_time}")
        print("=" * 60)
        self.show()
        self._qt_app.exec()


# ═══════════════════════════════════════════════════════════════════
# Multi-Replay UI — 多轨迹同时回放
# ═══════════════════════════════════════════════════════════════════


class MultiReplayUI(QMainWindow):
    """
    Qt 主窗口，支持多个轨迹文件在 RxC 网格布局中同时回放。

    每个轨迹渲染到 Panda3D 窗口内独立的 DisplayRegion 中，
    所有轨迹共享播放控件（播放/暂停、帧滑块、FPS、步长）。

    用法（from main.py）：
        ui = MultiReplayUI(datasets, labels, layout, visualizer, ...)
        ui.run()
    """

    def __init__(
        self,
        datasets: list,
        labels: list[str],
        layout: tuple[int, int],
        visualizer,
        night_mode: bool = True,
        initial_fps: int = 10,
    ):
        self._qt_app = QApplication.instance() or QApplication([])
        super().__init__()
        self._datasets = datasets
        self._labels = labels
        self._layout = layout
        self._viz = visualizer
        self._night_mode = night_mode
        self._initial_fps = max(1, min(60, initial_fps))

        # 回放状态
        from TrajectoryRecord.replay_state import ReplayWorldState
        self._replay_worlds = [ReplayWorldState(d) for d in datasets]
        self._max_frames = max(rw.total_frames for rw in self._replay_worlds)
        self._current_frame = 0
        self._paused = True
        self._speed = 1
        self._frame_interval = 1.0 / self._initial_fps
        self._last_advance_time = 0.0

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE if night_mode else _LIGHT_STYLE)

        # 定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(16)

        self._panda_embedded = False

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        grid_rows, grid_cols = self._layout
        n = len(self._datasets)

        self.setWindowTitle(
            f"MAS-RMFS  \u2014  Multi-Replay {grid_rows}\u00d7{grid_cols}  "
            f"({n} trajectories)"
        )
        self.resize(1500, 950)

        inner_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(inner_splitter)

        # 左侧面板
        left_widget = QWidget()
        left_widget.setMinimumWidth(300)
        left_widget.setMaximumWidth(420)
        lo = QVBoxLayout(left_widget)
        lo.setContentsMargins(12, 12, 12, 12)
        lo.setSpacing(10)

        title = QLabel("\U0001f3ac Multi-Replay")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lo.addWidget(title)

        # ── 回放状态 ──
        status_box = QGroupBox("Playback")
        status_layout = QVBoxLayout(status_box)

        self._status_label = QLabel("\u23f8  PAUSED")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._status_label)

        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setObjectName("tickLabel")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._frame_label)

        info_row = QHBoxLayout()
        self._layout_label = QLabel(f"Layout: {grid_rows}\u00d7{grid_cols}")
        self._count_label = QLabel(f"Files: {n}")
        info_row.addWidget(self._layout_label)
        info_row.addWidget(self._count_label)
        status_layout.addLayout(info_row)
        lo.addWidget(status_box)

        # ── 控件 ──
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)

        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("\u25b6  Play")
        self._play_btn.setObjectName("playBtn")
        self._play_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self._play_btn)

        self._prev_btn = QPushButton("\u23ee  Prev")
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        btn_row.addWidget(self._prev_btn)

        self._next_btn = QPushButton("Next  \u23ed")
        self._next_btn.clicked.connect(lambda: self._step(1))
        btn_row.addWidget(self._next_btn)
        ctrl_layout.addLayout(btn_row)

        # 帧滑块
        slider_label = QLabel("\U0001f39e  Frame")
        slider_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ctrl_layout.addWidget(slider_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, self._max_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        ctrl_layout.addWidget(self._frame_slider)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        ctrl_layout.addWidget(sep)

        # FPS 控制
        fps_label = QLabel("\u23f1  Playback FPS")
        fps_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ctrl_layout.addWidget(fps_label)

        fps_row = QHBoxLayout()
        self._fps_slider = QSlider(Qt.Orientation.Horizontal)
        self._fps_slider.setRange(1, 60)
        self._fps_slider.setValue(self._initial_fps)
        self._fps_slider.valueChanged.connect(self._on_fps_change)
        fps_row.addWidget(self._fps_slider)

        self._fps_value = QLabel(f"{self._initial_fps} fps")
        self._fps_value.setMinimumWidth(55)
        fps_row.addWidget(self._fps_value)
        ctrl_layout.addLayout(fps_row)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for label_text, val in [("2fps", 2), ("5fps", 5), ("10fps", 10),
                                ("30fps", 30), ("60fps", 60)]:
            btn = QPushButton(label_text)
            btn.setObjectName("speedPreset")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _, v=val: self._set_fps(v))
            preset_row.addWidget(btn)
        ctrl_layout.addLayout(preset_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        ctrl_layout.addWidget(sep2)

        # 步长控制
        step_label = QLabel("\u23e9  Step Size")
        step_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ctrl_layout.addWidget(step_label)

        step_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 50)
        self._speed_slider.setValue(1)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        step_row.addWidget(self._speed_slider)

        self._speed_value = QLabel("x1")
        self._speed_value.setMinimumWidth(40)
        step_row.addWidget(self._speed_value)
        ctrl_layout.addLayout(step_row)
        lo.addWidget(ctrl_box)

        # ── 信息区域 ──
        info_box = QGroupBox("Trajectories")
        info_lo = QVBoxLayout(info_box)
        for i, data in enumerate(self._datasets):
            lbl = QLabel(
                f"[{i}] {self._labels[i]}\n"
                f"    Map: {data.rows}\u00d7{data.cols}  "
                f"Agents: {data.num_agents}  "
                f"Frames: {data.total_ticks}"
            )
            lbl.setStyleSheet("font-size: 10px;")
            info_lo.addWidget(lbl)
        lo.addWidget(info_box)

        lo.addStretch()

        hint = QLabel(
            "Space: Play/Pause  |  Left/Right: Step\n"
            "Up/Down: Speed  |  Home/End: Jump"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("font-size: 10px; color: #888;")
        lo.addWidget(hint)

        inner_splitter.addWidget(left_widget)

        # 右侧面板（Panda3D 容器）
        self._panda_container = QWidget()
        self._panda_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._panda_container.setMinimumSize(600, 400)
        inner_splitter.addWidget(self._panda_container)

        inner_splitter.setStretchFactor(0, 0)
        inner_splitter.setStretchFactor(1, 1)
        inner_splitter.setSizes([340, 1160])

    # ── 嵌入 Panda3D ─────────────────────────────────────────────────

    def _embed_panda(self):
        handle = int(self._panda_container.winId())
        self._viz._parent_window_handle = handle
        dpr = self._panda_container.devicePixelRatio()
        self._viz._parent_initial_size = (
            int(self._panda_container.width() * dpr),
            int(self._panda_container.height() * dpr),
        )
        self._panda_embedded = True

    def _resize_panda(self):
        if (self._viz._app is None or self._viz._app.win is None
                or not self._panda_embedded):
            return
        from panda3d.core import WindowProperties
        dpr = self._panda_container.devicePixelRatio()
        w = int(self._panda_container.width() * dpr)
        h = int(self._panda_container.height() * dpr)
        if w > 0 and h > 0:
            wp = WindowProperties()
            wp.setSize(w, h)
            wp.setOrigin(0, 0)
            self._viz._app.win.requestProperties(wp)
            self._viz.on_window_resize(w, h)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_panda()

    # ── 回放逻辑 ──────────────────────────────────────────────────

    def _on_timer(self):
        # 首次初始化 Panda3D
        if not self._viz._initialised:
            if not self._panda_embedded:
                self._embed_panda()
            for rw in self._replay_worlds:
                rw.set_frame(0)
            self._viz.setup(
                datasets=self._datasets,
                layout=self._layout,
                replay_worlds=self._replay_worlds,
                labels=self._labels,
            )
            for delay in (50, 200, 500):
                QTimer.singleShot(delay, self._resize_panda)
            self._update_info()
            return

        now = time.time()

        # 自动播放
        if not self._paused and now - self._last_advance_time >= self._frame_interval:
            new_frame = self._current_frame + self._speed
            if new_frame >= self._max_frames:
                new_frame = self._max_frames - 1
                self._paused = True
                self._play_btn.setText("\u25b6  Play")
                self._status_label.setText("\u23f9  END")
                self._status_label.setStyleSheet(
                    "color: #e74c3c; font-size: 14px; font-weight: bold;")
            self._current_frame = new_frame
            for rw in self._replay_worlds:
                rw.set_frame(min(new_frame, rw.total_frames - 1))
            self._last_advance_time = now

        # 驱动 Panda3D
        self._viz.update_all()
        self._viz.step()
        self._update_info()

    def _step(self, delta: int):
        """手动步进帧。"""
        new_frame = max(0, min(self._max_frames - 1,
                               self._current_frame + delta))
        self._current_frame = new_frame
        for rw in self._replay_worlds:
            rw.set_frame(min(new_frame, rw.total_frames - 1))
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(new_frame)
        self._frame_slider.blockSignals(False)

    def _update_info(self):
        idx = self._current_frame
        self._frame_label.setText(f"Frame: {idx} / {self._max_frames - 1}")
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)

    # ── 控件回调 ──────────────────────────────────────────────────

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._play_btn.setText("\u25b6  Play")
            self._status_label.setText("\u23f8  PAUSED")
            self._status_label.setStyleSheet(
                "color: #f0a500; font-size: 14px; font-weight: bold;")
        else:
            if self._current_frame >= self._max_frames - 1:
                self._current_frame = 0
                for rw in self._replay_worlds:
                    rw.set_frame(0)
            self._play_btn.setText("\u23f8  Pause")
            self._status_label.setText("\u25b6  PLAYING")
            self._status_label.setStyleSheet(
                "color: #2ecc71; font-size: 14px; font-weight: bold;")
            self._last_advance_time = time.time()

    def _on_frame_slider(self, value):
        self._current_frame = value
        for rw in self._replay_worlds:
            rw.set_frame(min(value, rw.total_frames - 1))

    def _on_fps_change(self, value):
        self._frame_interval = 1.0 / value
        self._fps_value.setText(f"{value} fps")

    def _set_fps(self, fps: int):
        self._frame_interval = 1.0 / fps
        self._fps_slider.setValue(fps)
        self._fps_value.setText(f"{fps} fps")

    def _on_speed_change(self, value):
        self._speed = value
        self._speed_value.setText(f"x{value}")

    def _set_speed(self, speed: int):
        self._speed = speed
        self._speed_slider.setValue(speed)
        self._speed_value.setText(f"x{speed}")

    # ── 键盘 ─────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self._toggle_pause()
        elif event.key() == Qt.Key.Key_Left:
            self._step(-1)
        elif event.key() == Qt.Key.Key_Right:
            self._step(1)
        elif event.key() == Qt.Key.Key_Up:
            self._set_speed(min(50, self._speed + 1))
        elif event.key() == Qt.Key.Key_Down:
            self._set_speed(max(1, self._speed - 1))
        elif event.key() == Qt.Key.Key_Home:
            self._current_frame = 0
            for rw in self._replay_worlds:
                rw.set_frame(0)
        elif event.key() == Qt.Key.Key_End:
            self._current_frame = self._max_frames - 1
            for rw in self._replay_worlds:
                rw.set_frame(min(self._max_frames - 1, rw.total_frames - 1))
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        event.accept()

    # ── 公共 API ────────────────────────────────────────────────────

    def run(self):
        grid_rows, grid_cols = self._layout
        n = len(self._datasets)
        print("=" * 60)
        print(f"MAS-RMFS Multi-Trajectory Replay (Panda3D)")
        print(f"  Layout: {grid_rows}x{grid_cols}  ({n} trajectories)")
        print("-" * 60)
        for i, data in enumerate(self._datasets):
            print(f"  [{i}] {self._labels[i]}")
            print(f"      Map: {data.rows}x{data.cols}, "
                  f"Agents: {data.num_agents}, Frames: {data.total_ticks}")
        print("=" * 60)
        self.show()
        self._qt_app.exec()