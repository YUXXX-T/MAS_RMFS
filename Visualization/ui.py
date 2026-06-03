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
    QSplitter, QScrollArea, QGridLayout, QTabWidget,
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
    "QUEUING": 6, "EXITING": 7,
}
_STATUS_LABELS = list(_STATUS_CODES.keys())

# ─── 仿真速度预设 ──────────────────────────────────────────────────
_SIM_SPEED_PRESETS = [
    ("1x",  10, 1),
    ("2x",  20, 1),
    ("5x",  50, 1),
    ("10x", 60, 1),
    ("Max", 60, 10),
]


# ─── 回放指标计算器 ──────────────────────────────────────────────────

class ReplayMetricsComputer:
    """从轨迹帧数据中计算仿真风格仪表盘指标。"""

    def __init__(self, data: "TrajectoryData"):
        self._data = data
        self._prev_statuses: list[str] = []
        self._cumulative_deliveries = 0
        self._delivery_history: list[int] = []
        self._utilization_history: list[float] = []
        self._last_computed_frame = -1

    def compute(self, frame_index: int) -> dict:
        frame = self._data.frames[frame_index]
        agents = frame["agents"]
        n = len(agents)

        status_counts: dict[str, int] = {}
        current_statuses: list[str] = []
        active_pods = 0

        for agent in agents:
            s = agent.get("status", "IDLE")
            status_counts[s] = status_counts.get(s, 0) + 1
            current_statuses.append(s)
            if agent.get("pod") is not None:
                active_pods += 1

        idle_count = status_counts.get("IDLE", 0)
        utilization = ((n - idle_count) / n * 100) if n > 0 else 0.0

        new_deliveries = 0
        if self._prev_statuses and len(self._prev_statuses) == n:
            for prev_s, cur_s in zip(self._prev_statuses, current_statuses):
                if prev_s == "DELIVERING" and cur_s != "DELIVERING":
                    new_deliveries += 1
        self._cumulative_deliveries += new_deliveries

        avg_displacement = 0.0
        if frame_index > 0:
            prev_frame = self._data.frames[frame_index - 1]
            total_disp = 0.0
            for i, agent in enumerate(agents):
                if i < len(prev_frame["agents"]):
                    pr, pc = prev_frame["agents"][i]["pos"]
                    cr, cc = agent["pos"]
                    total_disp += abs(cr - pr) + abs(cc - pc)
            avg_displacement = total_disp / n if n > 0 else 0.0

        self._prev_statuses = current_statuses
        self._delivery_history.append(self._cumulative_deliveries)
        self._utilization_history.append(utilization)
        self._last_computed_frame = frame_index

        return {
            "status_counts": status_counts,
            "utilization": utilization,
            "active_pods": active_pods,
            "total_pods": len(self._data.pod_homes),
            "cumulative_deliveries": self._cumulative_deliveries,
            "avg_displacement": avg_displacement,
            "delivery_history": self._delivery_history,
            "utilization_history": self._utilization_history,
            "num_agents": n,
        }

    def reset(self):
        self._prev_statuses = []
        self._cumulative_deliveries = 0
        self._delivery_history = []
        self._utilization_history = []
        self._last_computed_frame = -1

    def recompute_to_frame(self, target: int) -> dict:
        self.reset()
        result = {}
        for i in range(target + 1):
            result = self.compute(i)
        return result


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
QTabWidget::pane {
    border: 1px solid #333355;
    border-radius: 4px;
}
QTabBar::tab {
    background: #2a2a4a;
    color: #aaaacc;
    border: 1px solid #333355;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 16px;
    margin-right: 2px;
    font-size: 12px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #1a1a2e;
    color: #6c63ff;
    border-bottom: 2px solid #6c63ff;
}
QTabBar::tab:hover:!selected {
    background: #3a3a5a;
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
QTabWidget::pane {
    border: 1px solid #ccccdd;
    border-radius: 4px;
}
QTabBar::tab {
    background: #e0e0e8;
    color: #555566;
    border: 1px solid #ccccdd;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    padding: 6px 16px;
    margin-right: 2px;
    font-size: 12px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: #f0f0f5;
    color: #4361ee;
    border-bottom: 2px solid #4361ee;
}
QTabBar::tab:hover:!selected {
    background: #d0d0d8;
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
            "#f59e0b",     # QUEUING
            "#6b7280",     # EXITING
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
                  origin="upper", vmin=0, vmax=7, interpolation="nearest")
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

                # 达到最大 tick 数时自动暂停并保存轨迹
                if (self._engine.max_ticks > 0
                        and self._engine.world.tick >= self._engine.max_ticks):
                    self._engine.logger.info(
                        f"Reached max ticks ({self._engine.max_ticks}). "
                        f"Stopping simulation."
                    )
                    self._stopped = True
                    self._paused = True
                    if (self._engine.trajectory_recorder
                            and self._engine.trajectory_output):
                        self._engine.trajectory_recorder.save(
                            self._engine.trajectory_output)
                    self._engine._print_summary()

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
        simulation_mode: bool = True,
    ):
        self._qt_app = QApplication.instance() or QApplication([])
        super().__init__()
        self._data = data
        self._viz = visualizer
        self._night_mode = night_mode
        self._initial_fps = max(1, min(60, initial_fps))
        self._simulation_mode = simulation_mode

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

        # 仪表盘指标
        self._metrics = ReplayMetricsComputer(data)
        self._last_metrics: dict = {}

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE if night_mode else _LIGHT_STYLE)

        # 定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(16)

        self._panda_embedded = False

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        sim = self._simulation_mode
        self.setWindowTitle(
            "MAS-RMFS  \u2014  Simulation" if sim
            else "MAS-RMFS  \u2014  Trajectory Replay"
        )
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

        title = QLabel(
            "\U0001f916 MAS-RMFS" if sim
            else "\U0001f3ac MAS-RMFS Replay"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        layout.addWidget(title)

        # ── 状态 ──
        status_box = QGroupBox("Simulation" if sim else "Playback")
        status_layout = QVBoxLayout(status_box)

        self._status_label = QLabel("\u23f8  PAUSED")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._status_label)

        self._frame_label = QLabel("Tick: 0" if sim else "Frame: 0 / 0")
        self._frame_label.setObjectName("tickLabel")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._frame_label)

        info_row = QHBoxLayout()
        self._tick_label = QLabel("Tick: 0")
        self._agents_label = QLabel(f"Agents: {self._data.num_agents}")
        if sim:
            self._tick_label.setVisible(False)
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

        # ── 回放控件容器（仿真模式下隐藏） ──
        self._replay_controls_container = QWidget()
        rc_layout = QVBoxLayout(self._replay_controls_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(6)

        slider_label = QLabel("🎞  Frame")
        slider_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(slider_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, self._data.total_ticks - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        rc_layout.addWidget(self._frame_slider)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        rc_layout.addWidget(sep)

        fps_label = QLabel("⏱  Playback FPS")
        fps_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(fps_label)

        fps_row = QHBoxLayout()
        self._fps_slider = QSlider(Qt.Orientation.Horizontal)
        self._fps_slider.setRange(1, 60)
        self._fps_slider.setValue(self._initial_fps)
        self._fps_slider.valueChanged.connect(self._on_fps_change)
        fps_row.addWidget(self._fps_slider)

        self._fps_value = QLabel(f"{self._initial_fps} fps")
        self._fps_value.setMinimumWidth(55)
        fps_row.addWidget(self._fps_value)
        rc_layout.addLayout(fps_row)

        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for label, val in [("2fps", 2), ("5fps", 5), ("10fps", 10), ("30fps", 30), ("60fps", 60)]:
            btn = QPushButton(label)
            btn.setObjectName("speedPreset")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _, v=val: self._set_fps(v))
            preset_row.addWidget(btn)
        rc_layout.addLayout(preset_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        rc_layout.addWidget(sep2)

        step_label = QLabel("⏩  Step Size")
        step_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(step_label)

        step_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 50)
        self._speed_slider.setValue(1)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        step_row.addWidget(self._speed_slider)

        self._speed_value = QLabel("x1")
        self._speed_value.setMinimumWidth(40)
        step_row.addWidget(self._speed_value)
        rc_layout.addLayout(step_row)

        self._replay_controls_container.setVisible(not sim)
        ctrl_layout.addWidget(self._replay_controls_container)

        # ── 仿真速度控件（仿真模式下可见） ──
        self._sim_speed_container = QWidget()
        ss_layout = QVBoxLayout(self._sim_speed_container)
        ss_layout.setContentsMargins(0, 0, 0, 0)
        ss_layout.setSpacing(6)

        speed_title = QLabel("⏱  Simulation Speed")
        speed_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ss_layout.addWidget(speed_title)

        self._sim_speed_label = QLabel("Speed: 1x")
        self._sim_speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sim_speed_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        ss_layout.addWidget(self._sim_speed_label)

        sim_preset_row = QHBoxLayout()
        sim_preset_row.setSpacing(4)
        for label, fps, step in _SIM_SPEED_PRESETS:
            btn = QPushButton(label)
            btn.setObjectName("speedPreset")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(
                lambda _, f=fps, s=step, l=label: self._set_sim_speed(f, s, l))
            sim_preset_row.addWidget(btn)
        ss_layout.addLayout(sim_preset_row)

        self._sim_speed_container.setVisible(sim)
        ctrl_layout.addWidget(self._sim_speed_container)

        layout.addWidget(ctrl_box)

        # 图表切换
        self._chart_btn = QPushButton("📊  Show Charts")
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
        if sim:
            self._recorded_label.setVisible(False)
        layout.addWidget(info_box)

        # ── 仪表盘指标（仿真模式） ──
        if sim:
            monitor_box = QGroupBox("Monitor")
            mon_layout = QVBoxLayout(monitor_box)
            self._util_label = QLabel("Utilization: 0.0%")
            self._active_pods_label = QLabel(
                f"Active Pods: 0 / {len(self._data.pod_homes)}")
            self._deliveries_label = QLabel("Est. Deliveries: 0")
            self._displacement_label = QLabel("Avg Movement: 0.00")
            self._status_dist_label = QLabel("IDLE: 0")
            self._status_dist_label.setWordWrap(True)
            for lbl in [self._util_label, self._active_pods_label,
                        self._deliveries_label, self._displacement_label,
                        self._status_dist_label]:
                lbl.setStyleSheet("font-size: 12px;")
                mon_layout.addWidget(lbl)
            layout.addWidget(monitor_box)

        layout.addStretch()

        hint_text = (
            "Space: Play/Pause  |  C: Charts"
            if sim else
            "Space: Play/Pause  |  Left/Right: Step  |  C: Charts"
        )
        hint = QLabel(hint_text)
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
            "#f59e0b", "#6b7280",
        ])

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)

        fig = plt.figure(figsize=(14, 3.5), facecolor=bg)
        self._chart_fig = fig
        n_charts = 3 if self._simulation_mode else 2
        axes = fig.subplots(1, n_charts)
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

        if self._simulation_mode:
            self._draw_timeline(self._chart_axes[0])
            self._draw_density(self._chart_axes[1])
            self._draw_estimated_throughput(self._chart_axes[2])
        else:
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
                  origin="upper", vmin=0, vmax=7, interpolation="nearest")

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


    def _draw_estimated_throughput(self, ax):
        history = self._last_metrics.get("delivery_history", [])
        if not history:
            return
        ticks = list(range(len(history)))
        ax.fill_between(ticks, history, alpha=0.15, color="#2ecc71")
        ax.plot(ticks, history, color="#2ecc71", linewidth=2)
        current = history[-1] if history else 0
        tick = self._replay_world.tick or 1
        rate = current / max(tick, 1)
        ax.text(0.98, 0.92, f"Deliveries: {current}\nRate: {rate:.2f}/tick",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#2ecc71", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self._chart_ax_bg,
                          edgecolor="#2ecc71", alpha=0.8))
        ax.set_xlabel("Tick", color=self._chart_tick_clr, fontsize=8)
        ax.set_ylabel("Est. Deliveries", color=self._chart_tick_clr, fontsize=8)
        ax.set_title("Estimated Throughput", color=self._chart_title_clr,
                      fontsize=10, pad=6)

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
                self._status_label.setText("\u23f9  STOPPED" if self._simulation_mode else "\u23f9  END")
                self._status_label.setStyleSheet(
                    "color: #e74c3c; font-size: 14px; font-weight: bold;")
            ws.set_frame(new_frame)
            self._last_advance_time = now

            # 计算仪表盘指标
            if self._simulation_mode:
                self._last_metrics = self._metrics.compute(ws.current_frame_index)
                self._update_metrics_display()

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
        if self._simulation_mode:
            self._frame_label.setText(f"Tick: {ws.tick}")
        else:
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
                if self._simulation_mode:
                    self._metrics.reset()
            self._play_btn.setText("\u23f8  Pause")
            self._status_label.setText("\u25b6  RUNNING" if self._simulation_mode else "\u25b6  PLAYING")
            self._status_label.setStyleSheet(
                "color: #2ecc71; font-size: 14px; font-weight: bold;")
            self._last_advance_time = time.time()

    def _on_frame_slider(self, value):
        self._replay_world.set_frame(value)
        if self._simulation_mode:
            self._last_metrics = self._metrics.recompute_to_frame(value)
            self._update_metrics_display()

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

    def _set_sim_speed(self, fps: int, step: int, label: str):
        self._frame_interval = 1.0 / fps
        self._speed = step
        self._sim_speed_label.setText(f"Speed: {label}")
        self._fps_slider.setValue(fps)
        self._speed_slider.setValue(step)

    def _update_metrics_display(self):
        if not self._simulation_mode or not self._last_metrics:
            return
        m = self._last_metrics
        self._util_label.setText(f"Utilization: {m['utilization']:.1f}%")
        self._active_pods_label.setText(
            f"Active Pods: {m['active_pods']} / {m['total_pods']}")
        self._deliveries_label.setText(
            f"Est. Deliveries: {m['cumulative_deliveries']}")
        self._displacement_label.setText(
            f"Avg Movement: {m['avg_displacement']:.2f}")
        parts = []
        for s in _STATUS_LABELS:
            cnt = m['status_counts'].get(s, 0)
            if cnt > 0:
                parts.append(f"{s}: {cnt}")
        self._status_dist_label.setText(
            " | ".join(parts) if parts else "All IDLE")

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
        if self._simulation_mode:
            print("=" * 60)
            print("MAS-RMFS Simulation (Replay-driven)")
            print(f"  Map: {self._data.rows}x{self._data.cols}")
            print(f"  Agents: {self._data.num_agents}")
            print(f"  Pods: {len(self._data.pod_homes)}")
            print(f"  Stations: {len(self._data.stations)}")
            print("=" * 60)
        else:
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
        simulation_mode: bool = True,
    ):
        self._qt_app = QApplication.instance() or QApplication([])
        super().__init__()
        self._datasets = datasets
        self._labels = labels
        self._layout = layout
        self._viz = visualizer
        self._night_mode = night_mode
        self._initial_fps = max(1, min(60, initial_fps))
        self._simulation_mode = simulation_mode

        # 回放状态
        from TrajectoryRecord.replay_state import ReplayWorldState
        self._replay_worlds = [ReplayWorldState(d) for d in datasets]
        self._max_frames = max(rw.total_frames for rw in self._replay_worlds)
        self._current_frame = 0
        self._paused = True
        self._speed = 1
        self._frame_interval = 1.0 / self._initial_fps
        self._last_advance_time = 0.0

        # 聚焦模式状态
        self._focused_slot_index: int | None = None
        self._chart_visible = False
        self._chart_last_frame = -1
        self._density: np.ndarray | None = None
        self._status_history: list[list[int]] = []

        # 仪表盘指标
        self._metrics_computers = [ReplayMetricsComputer(d) for d in datasets]
        self._last_metrics_list: list[dict] = [{} for _ in datasets]
        self._last_metrics: dict = {}

        self._build_ui()
        self.setStyleSheet(_DARK_STYLE if night_mode else _LIGHT_STYLE)

        # 点击检测（通过 Panda3D 鼠标事件回调）
        self._viz._on_click_callback = self._focus_on

        # 定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(16)

        self._panda_embedded = False

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        grid_rows, grid_cols = self._layout
        n = len(self._datasets)

        if self._simulation_mode:
            self.setWindowTitle(
                f"MAS-RMFS  \u2014  Multi-Agent Simulation {grid_rows}\u00d7{grid_cols}  "
                f"({n} scenarios)"
            )
        else:
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

        title = QLabel(
            "\U0001f916 Multi-Agent Simulation" if self._simulation_mode
            else "\U0001f3ac Multi-Replay"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lo.addWidget(title)

        # ── 回放状态 ──
        status_box = QGroupBox("Simulation" if self._simulation_mode else "Playback")
        status_layout = QVBoxLayout(status_box)

        self._status_label = QLabel("\u23f8  PAUSED")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._status_label)

        self._frame_label = QLabel("Tick: 0" if self._simulation_mode else "Frame: 0 / 0")
        self._frame_label.setObjectName("tickLabel")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._frame_label)

        info_row = QHBoxLayout()
        self._layout_label = QLabel(f"Layout: {grid_rows}\u00d7{grid_cols}")
        self._count_label = QLabel(f"Files: {n}")
        info_row.addWidget(self._layout_label)
        info_row.addWidget(self._count_label)
        status_layout.addLayout(info_row)

        import math as _math
        self._total_grid_rows = _math.ceil(n / grid_cols) if grid_cols > 0 else 1
        self._page_label = QLabel("")
        self._page_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._page_label.setStyleSheet("font-size: 11px; color: #6c63ff;")
        if self._total_grid_rows > grid_rows:
            self._page_label.setText(
                f"Rows 1-{grid_rows} of {self._total_grid_rows}  (scroll to navigate)")
        status_layout.addWidget(self._page_label)
        lo.addWidget(status_box)

        #── 控件 ──
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
        if self._simulation_mode:
            self._prev_btn.setVisible(False)
            self._next_btn.setVisible(False)
        ctrl_layout.addLayout(btn_row)

        # ── 回放控件容器（仿真模式下隐藏） ──
        self._replay_controls_container = QWidget()
        rc_layout = QVBoxLayout(self._replay_controls_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(6)

        slider_label = QLabel("\U0001f39e  Frame")
        slider_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(slider_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, self._max_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        rc_layout.addWidget(self._frame_slider)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        rc_layout.addWidget(sep)

        fps_label = QLabel("\u23f1  Playback FPS")
        fps_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(fps_label)

        fps_row = QHBoxLayout()
        self._fps_slider = QSlider(Qt.Orientation.Horizontal)
        self._fps_slider.setRange(1, 60)
        self._fps_slider.setValue(self._initial_fps)
        self._fps_slider.valueChanged.connect(self._on_fps_change)
        fps_row.addWidget(self._fps_slider)

        self._fps_value = QLabel(f"{self._initial_fps} fps")
        self._fps_value.setMinimumWidth(55)
        fps_row.addWidget(self._fps_value)
        rc_layout.addLayout(fps_row)

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
        rc_layout.addLayout(preset_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        rc_layout.addWidget(sep2)

        step_label = QLabel("\u23e9  Step Size")
        step_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(step_label)

        step_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 50)
        self._speed_slider.setValue(1)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        step_row.addWidget(self._speed_slider)

        self._speed_value = QLabel("x1")
        self._speed_value.setMinimumWidth(40)
        step_row.addWidget(self._speed_value)
        rc_layout.addLayout(step_row)

        self._replay_controls_container.setVisible(not self._simulation_mode)
        ctrl_layout.addWidget(self._replay_controls_container)

        # ── 仿真速度控件（仿真模式下可见） ──
        self._sim_speed_container = QWidget()
        ss_layout = QVBoxLayout(self._sim_speed_container)
        ss_layout.setContentsMargins(0, 0, 0, 0)
        ss_layout.setSpacing(6)

        speed_title = QLabel("\u23f1  Simulation Speed")
        speed_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ss_layout.addWidget(speed_title)

        self._sim_speed_label = QLabel("Speed: 1x")
        self._sim_speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sim_speed_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        ss_layout.addWidget(self._sim_speed_label)

        sim_preset_row = QHBoxLayout()
        sim_preset_row.setSpacing(4)
        for label_text, fps, step in _SIM_SPEED_PRESETS:
            btn = QPushButton(label_text)
            btn.setObjectName("speedPreset")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Fixed)
            btn.clicked.connect(
                lambda _, f=fps, s=step, l=label_text: self._set_sim_speed(f, s, l))
            sim_preset_row.addWidget(btn)
        ss_layout.addLayout(sim_preset_row)

        self._sim_speed_container.setVisible(self._simulation_mode)
        ctrl_layout.addWidget(self._sim_speed_container)

        lo.addWidget(ctrl_box)

        # ── 聚焦模式控件（默认隐藏）──
        self._back_btn = QPushButton("\u2b05  Back to Grid")
        self._back_btn.setObjectName("chartBtn")
        self._back_btn.setVisible(False)
        self._back_btn.clicked.connect(self._unfocus)
        lo.addWidget(self._back_btn)

        self._focus_label = QLabel("")
        self._focus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._focus_label.setStyleSheet(
            "font-size: 12px; color: #6c63ff; font-weight: bold;")
        self._focus_label.setVisible(False)
        lo.addWidget(self._focus_label)

        self._chart_btn = QPushButton("\U0001f4ca  Show Charts")
        self._chart_btn.setObjectName("chartBtn")
        self._chart_btn.setVisible(False)
        self._chart_btn.clicked.connect(self._toggle_charts)
        lo.addWidget(self._chart_btn)

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

        # ── 仪表盘指标（仿真模式） ──
        if self._simulation_mode:
            monitor_box = QGroupBox("Monitor")
            self._monitor_box = monitor_box
            mon_layout = QVBoxLayout(monitor_box)
            self._util_label = QLabel("Utilization: 0.0%")
            self._active_pods_label = QLabel("Active Pods: 0")
            self._deliveries_label = QLabel("Est. Deliveries: 0")
            self._displacement_label = QLabel("Avg Movement: 0.00")
            self._status_dist_label = QLabel("IDLE: 0")
            self._status_dist_label.setWordWrap(True)
            for lbl in [self._util_label, self._active_pods_label,
                        self._deliveries_label, self._displacement_label,
                        self._status_dist_label]:
                lbl.setStyleSheet("font-size: 12px;")
                mon_layout.addWidget(lbl)
            lo.addWidget(monitor_box)

        lo.addStretch()

        hint = QLabel(
            "Space: Play/Pause  |  Left/Right: Step\n"
            "Up/Down: Speed  |  Home/End: Jump\n"
            "Scroll: Zoom  |  Middle: Reset Zoom\n"
            "Click to focus  |  Esc: Back"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("font-size: 10px; color: #888;")
        lo.addWidget(hint)

        inner_splitter.addWidget(left_widget)

        # 右侧面板
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setMinimumSize(600, 400)

        self._panda_container = QWidget()
        self._panda_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._right_splitter.addWidget(self._panda_container)

        # Dashboard 面板（仿真模式下默认可见）
        self._dashboard = _DashboardPanel(night_mode=self._night_mode) if self._simulation_mode else None
        if self._dashboard is not None:
            self._right_splitter.addWidget(self._dashboard)

        # 图表面板（默认隐藏，聚焦时显示）
        self._charts_panel = self._build_charts_panel()
        self._charts_panel.setVisible(False)
        self._right_splitter.addWidget(self._charts_panel)
        self._right_splitter.setStretchFactor(0, 3)
        self._right_splitter.setStretchFactor(1, 2 if self._dashboard else 0)
        self._right_splitter.setStretchFactor(2 if self._dashboard else 1, 1)
        self._right_splitter.splitterMoved.connect(lambda *_: self._resize_panda())

        inner_splitter.addWidget(self._right_splitter)

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

    # ── 图表构建 ─────────────────────────────────────────────────

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
            "#f59e0b", "#6b7280",
        ])

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)

        fig = plt.figure(figsize=(14, 3.5), facecolor=bg)
        self._chart_fig = fig
        n_charts = 3 if self._simulation_mode else 2
        axes = fig.subplots(1, n_charts)
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
        """按当前帧计算聚焦轨迹的密度和时间线数据。"""
        si = self._focused_slot_index
        if si is None:
            return
        rw = self._replay_worlds[si]
        data = self._datasets[si]
        idx = rw.current_frame_index
        if idx == self._chart_last_frame:
            return
        self._chart_last_frame = idx

        rows, cols = data.rows, data.cols
        self._density = np.zeros((rows, cols), dtype=float)
        self._status_history = []
        for i in range(min(idx + 1, len(data.frames))):
            frame = data.frames[i]
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

        if self._simulation_mode:
            self._draw_timeline(self._chart_axes[0])
            self._draw_density(self._chart_axes[1])
            self._draw_estimated_throughput(self._chart_axes[2])
        else:
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
        if not self._status_history or self._focused_slot_index is None:
            return
        n_agents = self._datasets[self._focused_slot_index].num_agents
        n_ticks = len(self._status_history)
        mat = np.zeros((n_agents, n_ticks), dtype=int)
        for t, row in enumerate(self._status_history):
            for a, code in enumerate(row):
                if a < n_agents:
                    mat[a, t] = code

        ax.imshow(mat, cmap=self._timeline_cmap, aspect="auto",
                  origin="upper", vmin=0, vmax=7, interpolation="nearest")

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


    def _draw_estimated_throughput(self, ax):
        idx = self._focused_slot_index if self._focused_slot_index is not None else 0
        m = self._last_metrics_list[idx] if self._last_metrics_list else {}
        history = m.get("delivery_history", [])
        if not history:
            return
        ticks = list(range(len(history)))
        ax.fill_between(ticks, history, alpha=0.15, color="#2ecc71")
        ax.plot(ticks, history, color="#2ecc71", linewidth=2)
        current = history[-1] if history else 0
        rw = self._replay_worlds[idx]
        tick = rw.tick or 1
        rate = current / max(tick, 1)
        ax.text(0.98, 0.92, f"Deliveries: {current}\nRate: {rate:.2f}/tick",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#2ecc71", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self._chart_ax_bg,
                          edgecolor="#2ecc71", alpha=0.8))
        ax.set_xlabel("Tick", color=self._chart_tick_clr, fontsize=8)
        ax.set_ylabel("Est. Deliveries", color=self._chart_tick_clr, fontsize=8)
        ax.set_title("Estimated Throughput", color=self._chart_title_clr,
                      fontsize=10, pad=6)

    # ── 聚焦 ────────────────────────────────────────────────────

    def _focus_on(self, slot_index: int):
        """进入聚焦模式：仅显示选中的轨迹并启用图表。"""
        self._focused_slot_index = slot_index

        # 通知 Panda3D 可视化器
        self._viz.focus_slot(slot_index)

        # 重置图表数据
        self._density = None
        self._status_history = []
        self._chart_last_frame = -1
        self._chart_visible = True
        self._charts_panel.setVisible(True)
        self._chart_btn.setText("\U0001f4ca  Hide Charts")
        if self._dashboard is not None:
            self._dashboard.setVisible(False)

        # 调整帧滑块为聚焦轨迹的范围
        rw = self._replay_worlds[slot_index]
        self._frame_slider.setRange(0, max(0, rw.total_frames - 1))
        self._frame_slider.setValue(rw.current_frame_index)

        # 显示聚焦模式控件
        self._back_btn.setVisible(True)
        prefix = "Simulation:" if self._simulation_mode else "Focused:"
        self._focus_label.setText(
            f"{prefix} [{slot_index}] {self._labels[slot_index]}")
        self._focus_label.setVisible(True)
        self._chart_btn.setVisible(True)

        # 更新窗口标题
        self.setWindowTitle(
            f"MAS-RMFS  \u2014  Focused: {self._labels[slot_index]}")

        QTimer.singleShot(50, self._resize_panda)

    def _unfocus(self):
        """返回网格视图。"""
        if self._focused_slot_index is None:
            return

        # 取聚焦轨迹当前帧作为全局帧
        focused_rw = self._replay_worlds[self._focused_slot_index]
        self._current_frame = focused_rw.current_frame_index

        self._focused_slot_index = None

        # 恢复 Panda3D 网格布局
        self._viz.unfocus()

        # 隐藏图表
        self._chart_visible = False
        self._charts_panel.setVisible(False)
        if self._dashboard is not None:
            self._dashboard.setVisible(True)

        # 恢复帧滑块为全局范围
        self._frame_slider.setRange(0, max(0, self._max_frames - 1))
        self._frame_slider.setValue(self._current_frame)

        # 同步所有轨迹到聚焦轨迹所在的帧
        for rw in self._replay_worlds:
            rw.set_frame(min(self._current_frame, rw.total_frames - 1))

        # 隐藏聚焦模式控件
        self._back_btn.setVisible(False)
        self._focus_label.setVisible(False)
        self._chart_btn.setVisible(False)

        # 恢复窗口标题
        grid_rows, grid_cols = self._layout
        n = len(self._datasets)
        self.setWindowTitle(
            f"MAS-RMFS  \u2014  Multi-Replay {grid_rows}\u00d7{grid_cols}  "
            f"({n} trajectories)")

        QTimer.singleShot(50, self._resize_panda)

    def _set_sim_speed(self, fps: int, step: int, label: str):
        self._frame_interval = 1.0 / fps
        self._speed = step
        self._sim_speed_label.setText(f"Speed: {label}")
        self._fps_slider.setValue(fps)
        self._speed_slider.setValue(step)

    def _compute_aggregated_metrics(self) -> dict:
        valid = [m for m in self._last_metrics_list if m]
        if not valid:
            return {}
        total_agents = sum(m.get('num_agents', 0) for m in valid)
        if total_agents == 0:
            return {}
        util = sum(m['utilization'] * m['num_agents'] for m in valid) / total_agents
        deliveries = sum(m.get('cumulative_deliveries', 0) for m in valid)
        disp = sum(m.get('avg_displacement', 0) for m in valid) / len(valid)
        active = sum(m.get('active_pods', 0) for m in valid)
        total_pods = sum(m.get('total_pods', 0) for m in valid)
        merged_status: dict[str, int] = {}
        for m in valid:
            for s, cnt in m.get('status_counts', {}).items():
                merged_status[s] = merged_status.get(s, 0) + cnt
        return {
            'utilization': util,
            'cumulative_deliveries': deliveries,
            'avg_displacement': disp,
            'active_pods': active,
            'total_pods': total_pods,
            'status_counts': merged_status,
            'num_agents': total_agents,
        }

    def _update_metrics_display(self):
        if not self._simulation_mode or not self._last_metrics:
            return
        n = len(self._datasets)
        focused = self._focused_slot_index
        if focused is not None:
            self._monitor_box.setTitle(
                f"Monitor [{focused}] {self._labels[focused]}")
        else:
            self._monitor_box.setTitle(f"Monitor (All {n})")
        m = self._last_metrics
        self._util_label.setText(f"Utilization: {m['utilization']:.1f}%")
        self._active_pods_label.setText(
            f"Active Pods: {m['active_pods']} / {m.get('total_pods', '?')}")
        self._deliveries_label.setText(
            f"Est. Deliveries: {m['cumulative_deliveries']}")
        self._displacement_label.setText(
            f"Avg Movement: {m['avg_displacement']:.2f}")
        parts = []
        for s in _STATUS_LABELS:
            cnt = m['status_counts'].get(s, 0)
            if cnt > 0:
                parts.append(f"{s}: {cnt}")
        self._status_dist_label.setText(
            " | ".join(parts) if parts else "All IDLE")


    def _toggle_charts(self):
        self._chart_visible = not self._chart_visible
        self._charts_panel.setVisible(self._chart_visible)
        if self._chart_visible:
            self._chart_btn.setText("\U0001f4ca  Hide Charts")
            self._chart_last_frame = -1
            self._redraw_charts()
        else:
            self._chart_btn.setText("\U0001f4ca  Show Charts")
        QTimer.singleShot(50, self._resize_panda)

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
            if self._focused_slot_index is not None:
                # 聚焦模式：仅推进聚焦轨迹
                rw = self._replay_worlds[self._focused_slot_index]
                new_frame = rw.current_frame_index + self._speed
                if new_frame >= rw.total_frames:
                    new_frame = rw.total_frames - 1
                    self._paused = True
                    self._play_btn.setText("\u25b6  Play")
                    self._status_label.setText("\u23f9  STOPPED" if self._simulation_mode else "\u23f9  END")
                    self._status_label.setStyleSheet(
                        "color: #e74c3c; font-size: 14px; font-weight: bold;")
                rw.set_frame(new_frame)
            else:
                # 网格模式：推进所有轨迹
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

            # 计算仪表盘指标
            if self._simulation_mode:
                for i, rw in enumerate(self._replay_worlds):
                    if self._current_frame < rw.total_frames:
                        self._last_metrics_list[i] = self._metrics_computers[i].compute(rw.current_frame_index)
                if self._focused_slot_index is not None:
                    self._last_metrics = self._last_metrics_list[self._focused_slot_index]
                else:
                    self._last_metrics = self._compute_aggregated_metrics()
                self._update_metrics_display()
                if self._dashboard is not None:
                    self._dashboard.update(self._last_metrics_list, self._labels)

        # 驱动 Panda3D
        if self._focused_slot_index is not None:
            slot = self._viz._slots[self._focused_slot_index]
            self._viz._update_slot(slot)
        else:
            self._viz.update_all()
        self._viz.step()
        self._update_info()

        # 图表（仅聚焦模式）
        if self._focused_slot_index is not None and self._chart_visible:
            self._redraw_charts()

    def _step(self, delta: int):
        """手动步进帧。"""
        if self._focused_slot_index is not None:
            rw = self._replay_worlds[self._focused_slot_index]
            new_frame = max(0, min(rw.total_frames - 1,
                                   rw.current_frame_index + delta))
            rw.set_frame(new_frame)
            self._frame_slider.blockSignals(True)
            self._frame_slider.setValue(new_frame)
            self._frame_slider.blockSignals(False)
        else:
            new_frame = max(0, min(self._max_frames - 1,
                                   self._current_frame + delta))
            self._current_frame = new_frame
            for rw in self._replay_worlds:
                rw.set_frame(min(new_frame, rw.total_frames - 1))
            self._frame_slider.blockSignals(True)
            self._frame_slider.setValue(new_frame)
            self._frame_slider.blockSignals(False)

    def _update_info(self):
        if self._focused_slot_index is not None:
            rw = self._replay_worlds[self._focused_slot_index]
            idx = rw.current_frame_index
            if self._simulation_mode:
                self._frame_label.setText(f"Tick: {self._current_frame}")
            else:
                self._frame_label.setText(f"Frame: {idx} / {rw.total_frames - 1}")
        else:
            idx = self._current_frame
            if self._simulation_mode:
                self._frame_label.setText(f"Tick: {self._current_frame}")
            else:
                self._frame_label.setText(f"Frame: {idx} / {self._max_frames - 1}")
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)

        if hasattr(self, '_page_label') and hasattr(self, '_total_grid_rows'):
            grid_rows, grid_cols = self._layout
            if self._total_grid_rows > grid_rows:
                offset = self._viz.scroll_offset if self._viz._initialised else 0
                start = offset + 1
                end = min(offset + grid_rows, self._total_grid_rows)
                self._page_label.setText(
                    f"Rows {start}-{end} of {self._total_grid_rows}")
    # ── 控件回调 ──────────────────────────────────────────────────

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._play_btn.setText("\u25b6  Play")
            self._status_label.setText("\u23f8  PAUSED")
            self._status_label.setStyleSheet(
                "color: #f0a500; font-size: 14px; font-weight: bold;")
        else:
            if self._focused_slot_index is not None:
                rw = self._replay_worlds[self._focused_slot_index]
                if rw.current_frame_index >= rw.total_frames - 1:
                    rw.set_frame(0)
            else:
                if self._current_frame >= self._max_frames - 1:
                    self._current_frame = 0
                if self._simulation_mode:
                    for mc in self._metrics_computers:
                        mc.reset()
                    self._last_metrics_list = [{} for _ in self._datasets]
                    for rw in self._replay_worlds:
                        rw.set_frame(0)
            self._play_btn.setText("\u23f8  Pause")
            self._status_label.setText("\u25b6  RUNNING" if self._simulation_mode else "\u25b6  PLAYING")
            self._status_label.setStyleSheet(
                "color: #2ecc71; font-size: 14px; font-weight: bold;")
            self._last_advance_time = time.time()

    def _on_frame_slider(self, value):
        if self._focused_slot_index is not None:
            self._replay_worlds[self._focused_slot_index].set_frame(value)
        else:
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
        if event.key() == Qt.Key.Key_Escape:
            self._unfocus()
        elif event.key() == Qt.Key.Key_Space:
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
            if self._focused_slot_index is not None:
                self._replay_worlds[self._focused_slot_index].set_frame(0)
            else:
                self._current_frame = 0
                for rw in self._replay_worlds:
                    rw.set_frame(0)
        elif event.key() == Qt.Key.Key_End:
            if self._focused_slot_index is not None:
                rw = self._replay_worlds[self._focused_slot_index]
                rw.set_frame(rw.total_frames - 1)
            else:
                self._current_frame = self._max_frames - 1
                for rw in self._replay_worlds:
                    rw.set_frame(min(self._max_frames - 1, rw.total_frames - 1))
        elif event.key() == Qt.Key.Key_C:
            if self._focused_slot_index is not None:
                self._toggle_charts()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        """鼠标滚轮事件：网格模式下通知 Panda3D 滚动。"""
        if (self._focused_slot_index is None and self._viz._initialised
                and hasattr(self, '_total_grid_rows')):
            grid_rows, _ = self._layout
            if self._total_grid_rows > grid_rows:
                delta = event.angleDelta().y()
                if delta > 0:
                    self._viz.scroll_to(self._viz.scroll_offset - 1)
                elif delta < 0:
                    self._viz.scroll_to(self._viz.scroll_offset + 1)
                event.accept()
                return
        super().wheelEvent(event)

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


# ═══════════════════════════════════════════════════════════════════
# 列表模式 — 轨迹卡片 + 点击聚焦
# ═══════════════════════════════════════════════════════════════════

_CARD_DARK_STYLE = """
QFrame#trajCard {
    background-color: #1e1e3a;
    border: 1px solid #333355;
    border-radius: 8px;
    padding: 10px;
}
QFrame#trajCard:hover {
    border-color: #6c63ff;
    background-color: #24244a;
}
"""

_CARD_LIGHT_STYLE = """
QFrame#trajCard {
    background-color: #ffffff;
    border: 1px solid #ccccdd;
    border-radius: 8px;
    padding: 10px;
}
QFrame#trajCard:hover {
    border-color: #4361ee;
    background-color: #f0f0ff;
}
"""

# ─── 轨迹对比色板 ─────────────────────────────────────────────────
_TRAJ_COLORS = [
    "#4361ee", "#e07c24", "#2ecc71", "#e74c3c", "#9b59b6",
    "#1abc9c", "#f39c12", "#3498db", "#e91e63", "#00bcd4",
]


class _DashboardPanel(QWidget):
    """多轨迹实时仪表盘面板（2x2 matplotlib 图表）。"""

    def __init__(self, night_mode: bool = True):
        super().__init__()
        self._night_mode = night_mode
        self._last_redraw_time = 0.0
        self._build_ui()

    def _build_ui(self):
        nm = self._night_mode
        bg = "#0f0f1a" if nm else "#f5f5f8"
        ax_bg = "#16162a" if nm else "#ffffff"
        self._tick_clr = "#aaaaaa" if nm else "#333333"
        self._spine_clr = "#333355" if nm else "#bbbbcc"
        self._title_clr = "#e0e0e0" if nm else "#222222"
        self._ax_bg = ax_bg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        fig = plt.figure(figsize=(14, 7), facecolor=bg)
        self._fig = fig
        self._axes = fig.subplots(2, 2)
        for row in self._axes:
            for ax in row:
                ax.set_facecolor(ax_bg)
                ax.tick_params(colors=self._tick_clr, labelsize=7)
                for sp in ax.spines.values():
                    sp.set_color(self._spine_clr)

        self._canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(self._canvas)
        fig.tight_layout(pad=2.0)

    def update(self, metrics_list: list[dict], labels: list[str]):
        now = time.time()
        if now - self._last_redraw_time < 0.5:
            return
        self._last_redraw_time = now

        valid = [(i, m) for i, m in enumerate(metrics_list) if m]
        if not valid:
            return

        for row in self._axes:
            for ax in row:
                ax.clear()
                ax.set_facecolor(self._ax_bg)
                ax.tick_params(colors=self._tick_clr, labelsize=7)
                for sp in ax.spines.values():
                    sp.set_color(self._spine_clr)

        self._draw_utilization_trend(valid, labels)
        self._draw_delivery_trend(valid, labels)
        self._draw_comparison_bars(valid, labels)
        self._draw_status_distribution(valid)

        self._fig.tight_layout(pad=2.0)
        self._canvas.draw_idle()

    def _draw_utilization_trend(self, valid, labels):
        ax = self._axes[0][0]
        for idx, m in valid:
            hist = m.get('utilization_history', [])
            if hist:
                clr = _TRAJ_COLORS[idx % len(_TRAJ_COLORS)]
                ax.plot(range(len(hist)), hist, color=clr,
                        linewidth=1.5, alpha=0.85, label=labels[idx])
        ax.set_xlabel("Tick", color=self._tick_clr, fontsize=8)
        ax.set_ylabel("Utilization %", color=self._tick_clr, fontsize=8)
        ax.set_title("Utilization Trend", color=self._title_clr,
                      fontsize=10, pad=6)
        if len(valid) <= 6:
            ax.legend(fontsize=7, loc='upper left',
                      facecolor=self._ax_bg, edgecolor=self._spine_clr,
                      labelcolor=self._tick_clr)

    def _draw_delivery_trend(self, valid, labels):
        ax = self._axes[0][1]
        for idx, m in valid:
            hist = m.get('delivery_history', [])
            if hist:
                clr = _TRAJ_COLORS[idx % len(_TRAJ_COLORS)]
                ax.plot(range(len(hist)), hist, color=clr,
                        linewidth=1.5, alpha=0.85, label=labels[idx])
        ax.set_xlabel("Tick", color=self._tick_clr, fontsize=8)
        ax.set_ylabel("Cumulative Deliveries", color=self._tick_clr, fontsize=8)
        ax.set_title("Delivery Trend", color=self._title_clr,
                      fontsize=10, pad=6)
        if len(valid) <= 6:
            ax.legend(fontsize=7, loc='upper left',
                      facecolor=self._ax_bg, edgecolor=self._spine_clr,
                      labelcolor=self._tick_clr)

    def _draw_comparison_bars(self, valid, labels):
        ax = self._axes[1][0]
        n = len(valid)
        indices = list(range(n))
        bar_labels = [labels[idx] for idx, _ in valid]

        utils = [m.get('utilization', 0) for _, m in valid]
        deliveries = [m.get('cumulative_deliveries', 0) for _, m in valid]
        active = [m.get('active_pods', 0) for _, m in valid]

        max_d = max(deliveries) if deliveries and max(deliveries) > 0 else 1
        max_a = max(active) if active and max(active) > 0 else 1
        norm_d = [d / max_d * 100 for d in deliveries]
        norm_a = [a / max_a * 100 for a in active]

        w = 0.25
        x = np.arange(n)
        ax.bar(x - w, utils, w, color="#4361ee", alpha=0.8, label="Util %")
        ax.bar(x, norm_d, w, color="#2ecc71", alpha=0.8,
               label=f"Deliveries (norm, max={max_d})")
        ax.bar(x + w, norm_a, w, color="#e07c24", alpha=0.8,
               label=f"Active Pods (norm, max={max_a})")

        ax.set_xticks(x)
        tick_labels = [f"[{idx}]" for idx, _ in valid]
        ax.set_xticklabels(tick_labels, fontsize=7, color=self._tick_clr)
        ax.set_ylabel("Normalized %", color=self._tick_clr, fontsize=8)
        ax.set_title("Per-Trajectory Comparison", color=self._title_clr,
                      fontsize=10, pad=6)
        ax.legend(fontsize=6, loc='upper right',
                  facecolor=self._ax_bg, edgecolor=self._spine_clr,
                  labelcolor=self._tick_clr)

    def _draw_status_distribution(self, valid):
        ax = self._axes[1][1]
        merged: dict[str, int] = {}
        for _, m in valid:
            for s, cnt in m.get('status_counts', {}).items():
                merged[s] = merged.get(s, 0) + cnt

        status_colors = {
            "IDLE": "#555566" if self._night_mode else "#aaaaaa",
            "MOVING_TO_POD": "#4361ee",
            "CARRYING": "#f0a500",
            "DELIVERING": "#e07c24",
            "RETURNING": "#7b2cbf",
            "MOVING": "#2ec4b6",
        }

        filtered = [(s, merged.get(s, 0)) for s in _STATUS_LABELS if merged.get(s, 0) > 0]
        if not filtered:
            ax.set_title("Status Distribution", color=self._title_clr,
                          fontsize=10, pad=6)
            return

        labels_s = [s for s, _ in filtered]
        sizes = [c for _, c in filtered]
        colors = [status_colors.get(s, "#888888") for s in labels_s]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_s, colors=colors, autopct='%1.0f%%',
            textprops={'fontsize': 7, 'color': self._tick_clr},
            startangle=90, pctdistance=0.75)
        for t in autotexts:
            t.set_fontsize(7)
            t.set_color(self._title_clr)
        ax.set_title("Aggregated Status", color=self._title_clr,
                      fontsize=10, pad=6)


class _TrajectoryCard(QFrame):
    """列表模式中每个轨迹数据包的卡片控件。"""

    def __init__(self, index: int, data, label: str,
                 night_mode: bool = True, simulation_mode: bool = True,
                 on_click=None):
        super().__init__()
        self.setObjectName("trajCard")
        self._index = index
        self._on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._simulation_mode = simulation_mode
        self.setStyleSheet(_CARD_DARK_STYLE if night_mode else _CARD_LIGHT_STYLE)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # 标题行
        title_row = QHBoxLayout()
        idx_label = QLabel(f"[{index}]")
        idx_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        idx_label.setStyleSheet("color: #6c63ff;" if night_mode else "color: #4361ee;")
        title_row.addWidget(idx_label)

        name_label = QLabel(label)
        name_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title_row.addWidget(name_label)
        title_row.addStretch()

        if data.record_time:
            time_label = QLabel(data.record_time)
            time_label.setStyleSheet("font-size: 10px; color: #888;")
            if self._simulation_mode:
                time_label.setVisible(False)
            title_row.addWidget(time_label)
        layout.addLayout(title_row)

        # 静态指标行
        static_row = QHBoxLayout()
        clr = "#aaa" if night_mode else "#666"
        for text in [
            f"Map: {data.rows}×{data.cols}",
            f"Robots: {data.num_agents}",
            f"Frames: {data.total_ticks}",
            f"Stations: {len(data.stations)}",
            f"Pods: {len(data.pod_homes)}",
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet(f"font-size: 11px; color: {clr};")
            static_row.addWidget(lbl)
        static_row.addStretch()
        layout.addLayout(static_row)

        # 动态指标行
        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        layout.addWidget(self._frame_label)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(f"font-size: 10px; color: {clr};")
        layout.addWidget(self._status_label)

        if self._simulation_mode:
            self._metrics_label = QLabel("Util: — | Deliveries: —")
            self._metrics_label.setStyleSheet(
                "font-size: 11px; font-weight: bold; color: #2ecc71;")
            layout.addWidget(self._metrics_label)

    def update_metrics(self, frame_idx: int, total_frames: int,
                       status_counts: dict, carrying: int,
                       utilization: float | None = None,
                       deliveries: int | None = None):
        if self._simulation_mode:
            self._frame_label.setText(f"Tick: {frame_idx}")
        else:
            self._frame_label.setText(f"Frame: {frame_idx} / {total_frames - 1}")
        parts = []
        for s in ("IDLE", "MOVING_TO_POD", "CARRYING", "DELIVERING",
                  "RETURNING", "MOVING"):
            cnt = status_counts.get(s, 0)
            if cnt > 0:
                parts.append(f"{s}: {cnt}")
        if carrying > 0:
            parts.append(f"Pods carried: {carrying}")
        self._status_label.setText("  |  ".join(parts))
        if self._simulation_mode and utilization is not None:
            self._metrics_label.setText(
                f"Util: {utilization:.1f}%  |  Deliveries: {deliveries or 0}")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click(self._index)
        super().mousePressEvent(event)


class ListReplayUI(QMainWindow):
    """
    列表模式的多轨迹回放 UI。

    显示所有轨迹数据包的实时指标卡片列表，点击卡片进入
    Panda3D 聚焦视图。
    """

    def __init__(
        self,
        datasets: list,
        labels: list[str],
        night_mode: bool = True,
        initial_fps: int = 10,
        simulation_mode: bool = True,
    ):
        self._qt_app = QApplication.instance() or QApplication([])
        super().__init__()
        self._datasets = datasets
        self._labels = labels
        self._night_mode = night_mode
        self._initial_fps = max(1, min(60, initial_fps))
        self._simulation_mode = simulation_mode

        from TrajectoryRecord.replay_state import ReplayWorldState
        self._replay_worlds = [ReplayWorldState(d) for d in datasets]
        self._max_frames = max(rw.total_frames for rw in self._replay_worlds)
        self._current_frame = 0
        self._paused = True
        self._speed = 1
        self._frame_interval = 1.0 / self._initial_fps
        self._last_advance_time = 0.0

        # 聚焦模式
        self._focused_index: int | None = None
        self._viz = None
        self._panda_embedded = False
        self._chart_visible = False
        self._chart_last_frame = -1

        # 仪表盘指标
        self._metrics_computers = [ReplayMetricsComputer(d) for d in datasets]
        self._last_metrics_list: list[dict] = [{} for _ in datasets]
        self._last_metrics: dict = {}
        self._density = None
        self._status_history = []

        self._cards: list[_TrajectoryCard] = []
        self._build_ui()
        self.setStyleSheet(_DARK_STYLE if night_mode else _LIGHT_STYLE)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(16)

    # ── UI 构建 ───────────────────────────────────────────────

    def _build_ui(self):
        n = len(self._datasets)
        if self._simulation_mode:
            self.setWindowTitle(
                f"MAS-RMFS  —  Simulation ({n} scenarios)")
        else:
            self.setWindowTitle(
                f"MAS-RMFS  —  List Replay ({n} trajectories)")
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

        title = QLabel(
            "\U0001f916 Multi-Agent Simulation" if self._simulation_mode
            else "\U0001f3ac List Replay"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        lo.addWidget(title)

        # 回放状态
        status_box = QGroupBox("Simulation" if self._simulation_mode else "Playback")
        status_layout = QVBoxLayout(status_box)

        self._status_label = QLabel("⏸  PAUSED")
        self._status_label.setObjectName("statusLabel")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._status_label)

        self._frame_label = QLabel("Tick: 0" if self._simulation_mode else "Frame: 0 / 0")
        self._frame_label.setObjectName("tickLabel")
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._frame_label)

        self._count_label = QLabel(f"Files: {n}")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self._count_label)
        lo.addWidget(status_box)

        # 控件
        ctrl_box = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(ctrl_box)

        btn_row = QHBoxLayout()
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setObjectName("playBtn")
        self._play_btn.clicked.connect(self._toggle_pause)
        btn_row.addWidget(self._play_btn)

        self._prev_btn = QPushButton("⏮  Prev")
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn = QPushButton("Next  ⏭")
        self._next_btn.clicked.connect(lambda: self._step(1))
        btn_row.addWidget(self._prev_btn)
        btn_row.addWidget(self._next_btn)
        if self._simulation_mode:
            self._prev_btn.setVisible(False)
            self._next_btn.setVisible(False)
        ctrl_layout.addLayout(btn_row)

        # ── 回放控件容器（仿真模式下隐藏） ──
        self._replay_controls_container = QWidget()
        rc_layout = QVBoxLayout(self._replay_controls_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(6)

        slider_label = QLabel("\U0001f39e  Frame")
        slider_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(slider_label)

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(0, self._max_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider)
        rc_layout.addWidget(self._frame_slider)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        rc_layout.addWidget(sep)

        fps_label = QLabel("⏱  Playback FPS")
        fps_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(fps_label)

        fps_row = QHBoxLayout()
        self._fps_slider = QSlider(Qt.Orientation.Horizontal)
        self._fps_slider.setRange(1, 60)
        self._fps_slider.setValue(self._initial_fps)
        self._fps_slider.valueChanged.connect(self._on_fps_change)
        fps_row.addWidget(self._fps_slider)

        self._fps_value = QLabel(f"{self._initial_fps} fps")
        self._fps_value.setMinimumWidth(55)
        fps_row.addWidget(self._fps_value)
        rc_layout.addLayout(fps_row)

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
        rc_layout.addLayout(preset_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        rc_layout.addWidget(sep2)

        step_label = QLabel("⏩  Step Size")
        step_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        rc_layout.addWidget(step_label)

        step_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(1, 50)
        self._speed_slider.setValue(1)
        self._speed_slider.valueChanged.connect(self._on_speed_change)
        step_row.addWidget(self._speed_slider)

        self._speed_value = QLabel("x1")
        self._speed_value.setMinimumWidth(40)
        step_row.addWidget(self._speed_value)
        rc_layout.addLayout(step_row)

        self._replay_controls_container.setVisible(not self._simulation_mode)
        ctrl_layout.addWidget(self._replay_controls_container)

        # ── 仿真速度控件（仿真模式下可见） ──
        self._sim_speed_container = QWidget()
        ss_layout = QVBoxLayout(self._sim_speed_container)
        ss_layout.setContentsMargins(0, 0, 0, 0)
        ss_layout.setSpacing(6)

        speed_title = QLabel("⏱  Simulation Speed")
        speed_title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        ss_layout.addWidget(speed_title)

        self._sim_speed_label = QLabel("Speed: 1x")
        self._sim_speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sim_speed_label.setStyleSheet("font-size: 13px; font-weight: bold;")
        ss_layout.addWidget(self._sim_speed_label)

        sim_preset_row = QHBoxLayout()
        sim_preset_row.setSpacing(4)
        for label_text, fps, step in _SIM_SPEED_PRESETS:
            btn = QPushButton(label_text)
            btn.setObjectName("speedPreset")
            btn.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Fixed)
            btn.clicked.connect(
                lambda _, f=fps, s=step, l=label_text: self._set_sim_speed(f, s, l))
            sim_preset_row.addWidget(btn)
        ss_layout.addLayout(sim_preset_row)

        self._sim_speed_container.setVisible(self._simulation_mode)
        ctrl_layout.addWidget(self._sim_speed_container)

        lo.addWidget(ctrl_box)

        # 聚焦模式控件（默认隐藏）
        self._back_btn = QPushButton("⬅  Back to List")
        self._back_btn.setObjectName("chartBtn")
        self._back_btn.setVisible(False)
        self._back_btn.clicked.connect(self._unfocus)
        lo.addWidget(self._back_btn)

        self._focus_label = QLabel("")
        self._focus_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._focus_label.setStyleSheet(
            "font-size: 12px; color: #6c63ff; font-weight: bold;")
        self._focus_label.setVisible(False)
        lo.addWidget(self._focus_label)

        self._chart_btn = QPushButton("\U0001f4ca  Show Charts")
        self._chart_btn.setObjectName("chartBtn")
        self._chart_btn.setVisible(False)
        self._chart_btn.clicked.connect(self._toggle_charts)
        lo.addWidget(self._chart_btn)

        # ── 仪表盘指标（仿真模式） ──
        if self._simulation_mode:
            monitor_box = QGroupBox("Monitor")
            self._monitor_box = monitor_box
            mon_layout = QVBoxLayout(monitor_box)
            self._util_label = QLabel("Utilization: 0.0%")
            self._active_pods_label = QLabel("Active Pods: 0")
            self._deliveries_label = QLabel("Est. Deliveries: 0")
            self._displacement_label = QLabel("Avg Movement: 0.00")
            self._status_dist_label = QLabel("IDLE: 0")
            self._status_dist_label.setWordWrap(True)
            for lbl in [self._util_label, self._active_pods_label,
                        self._deliveries_label, self._displacement_label,
                        self._status_dist_label]:
                lbl.setStyleSheet("font-size: 12px;")
                mon_layout.addWidget(lbl)
            lo.addWidget(monitor_box)

        lo.addStretch()

        hint = QLabel(
            "Space: Play/Pause  |  Left/Right: Step\n"
            "Up/Down: Speed  |  Home/End: Jump\n"
            "Click card to focus  |  Esc: Back"
        )
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("font-size: 10px; color: #888;")
        lo.addWidget(hint)

        inner_splitter.addWidget(left_widget)

        # 右侧面板
        self._right_splitter = QSplitter(Qt.Orientation.Vertical)
        self._right_splitter.setMinimumSize(600, 400)

        # 卡片滚动区域
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_layout.setSpacing(10)
        for i, data in enumerate(self._datasets):
            card = _TrajectoryCard(
                index=i, data=data, label=self._labels[i],
                night_mode=self._night_mode,
                simulation_mode=self._simulation_mode,
                on_click=self._focus_on,
            )
            self._cards.append(card)
            scroll_layout.addWidget(card)

        scroll_layout.addStretch()
        self._scroll_area.setWidget(scroll_content)

        # Tab widget: Trajectories + Dashboard
        self._tab_widget = QTabWidget()
        self._tab_widget.addTab(self._scroll_area, "\U0001f4cb  Trajectories")
        self._dashboard = _DashboardPanel(night_mode=self._night_mode)
        self._tab_widget.addTab(self._dashboard, "\U0001f4ca  Dashboard")
        self._right_splitter.addWidget(self._tab_widget)

        self._panda_container = QWidget()
        self._panda_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._panda_container.setMinimumHeight(0)
        self._right_splitter.addWidget(self._panda_container)

        self._charts_panel = self._build_charts_panel()
        self._charts_panel.setMinimumHeight(0)
        self._right_splitter.addWidget(self._charts_panel)

        self._right_splitter.setCollapsible(0, True)
        self._right_splitter.setCollapsible(1, True)
        self._right_splitter.setCollapsible(2, True)
        self._right_splitter.setSizes([1, 0, 0])

        inner_splitter.addWidget(self._right_splitter)
        inner_splitter.setStretchFactor(0, 0)
        inner_splitter.setStretchFactor(1, 1)
        inner_splitter.setSizes([340, 1160])

    # ── Panda3D 嵌入 ─────────────────────────────────────────────

    def _ensure_visualizer(self):
        if self._viz is not None:
            return
        from Visualization.panda3d_visualizer import Panda3DVisualizer
        self._viz = Panda3DVisualizer(
            view_mode="2d", use_gpu=False, night_mode=self._night_mode)

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
        if (self._viz is None or self._viz._app is None
                or self._viz._app.win is None or not self._panda_embedded):
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
        if self._focused_index is not None:
            self._resize_panda()

    # ── 图表 ─────────────────────────────────────────────────────

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
            "#f59e0b", "#6b7280",
        ])

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)

        fig = plt.figure(figsize=(14, 3.5), facecolor=bg)
        self._chart_fig = fig
        n_charts = 3 if self._simulation_mode else 2
        axes = fig.subplots(1, n_charts)
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

    def _compute_chart_data(self):
        si = self._focused_index
        if si is None:
            return
        rw = self._replay_worlds[si]
        data = self._datasets[si]
        idx = rw.current_frame_index
        if idx == self._chart_last_frame:
            return
        self._chart_last_frame = idx

        rows, cols = data.rows, data.cols
        self._density = np.zeros((rows, cols), dtype=float)
        self._status_history = []
        for i in range(min(idx + 1, len(data.frames))):
            frame = data.frames[i]
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

        if self._simulation_mode:
            self._draw_timeline(self._chart_axes[0])
            self._draw_density(self._chart_axes[1])
            self._draw_estimated_throughput(self._chart_axes[2])
        else:
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
        if not self._status_history or self._focused_index is None:
            return
        n_agents = self._datasets[self._focused_index].num_agents
        n_ticks = len(self._status_history)
        mat = np.zeros((n_agents, n_ticks), dtype=int)
        for t, row in enumerate(self._status_history):
            for a, code in enumerate(row):
                if a < n_agents:
                    mat[a, t] = code

        ax.imshow(mat, cmap=self._timeline_cmap, aspect="auto",
                  origin="upper", vmin=0, vmax=7, interpolation="nearest")
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

    # ── 聚焦 / 取消聚焦 ──────────────────────────────────────────


    def _draw_estimated_throughput(self, ax):
        idx = self._focused_index if self._focused_index is not None else 0
        m = self._last_metrics_list[idx] if self._last_metrics_list else {}
        history = m.get("delivery_history", [])
        if not history:
            return
        ticks = list(range(len(history)))
        ax.fill_between(ticks, history, alpha=0.15, color="#2ecc71")
        ax.plot(ticks, history, color="#2ecc71", linewidth=2)
        current = history[-1] if history else 0
        rw = self._replay_worlds[idx]
        tick = rw.tick or 1
        rate = current / max(tick, 1)
        ax.text(0.98, 0.92, f"Deliveries: {current}\nRate: {rate:.2f}/tick",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="#2ecc71", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=self._chart_ax_bg,
                          edgecolor="#2ecc71", alpha=0.8))
        ax.set_xlabel("Tick", color=self._chart_tick_clr, fontsize=8)
        ax.set_ylabel("Est. Deliveries", color=self._chart_tick_clr, fontsize=8)
        ax.set_title("Estimated Throughput", color=self._chart_title_clr,
                      fontsize=10, pad=6)

    def _focus_on(self, index: int):
        self._focused_index = index

        self._density = None
        self._status_history = []
        self._chart_last_frame = -1
        self._chart_visible = True
        self._chart_btn.setText("\U0001f4ca  Hide Charts")

        total_h = self._right_splitter.height()
        chart_h = max(180, total_h // 4)
        self._right_splitter.setSizes([0, total_h - chart_h, chart_h])
        QApplication.processEvents()

        self._ensure_visualizer()
        if not self._panda_embedded:
            self._embed_panda()

        rw = self._replay_worlds[index]
        if not self._viz._initialised:
            self._viz.setup(rw)
        else:
            self._viz.switch_world(rw)
            self._resize_panda()

        self._frame_slider.setRange(0, max(0, rw.total_frames - 1))
        self._frame_slider.setValue(rw.current_frame_index)

        self._back_btn.setVisible(True)
        self._focus_label.setText(
            f"Focused: [{index}] {self._labels[index]}")
        self._focus_label.setVisible(True)
        self._chart_btn.setVisible(True)

        self.setWindowTitle(
            f"MAS-RMFS  —  Focused: {self._labels[index]}")

        for delay in (50, 200, 500):
            QTimer.singleShot(delay, self._resize_panda)

    def _unfocus(self):
        if self._focused_index is None:
            return
        focused_rw = self._replay_worlds[self._focused_index]
        self._current_frame = focused_rw.current_frame_index
        self._focused_index = None

        self._chart_visible = False

        total_h = self._right_splitter.height()
        self._right_splitter.setSizes([total_h, 0, 0])

        self._frame_slider.setRange(0, max(0, self._max_frames - 1))
        self._frame_slider.setValue(self._current_frame)

        for rw in self._replay_worlds:
            rw.set_frame(min(self._current_frame, rw.total_frames - 1))

        self._back_btn.setVisible(False)
        self._focus_label.setVisible(False)
        self._chart_btn.setVisible(False)

        n = len(self._datasets)
        self.setWindowTitle(
            f"MAS-RMFS  —  List Replay ({n} trajectories)")

    def _set_sim_speed(self, fps: int, step: int, label: str):
        self._frame_interval = 1.0 / fps
        self._speed = step
        self._sim_speed_label.setText(f"Speed: {label}")
        self._fps_slider.setValue(fps)
        self._speed_slider.setValue(step)

    def _compute_aggregated_metrics(self) -> dict:
        valid = [m for m in self._last_metrics_list if m]
        if not valid:
            return {}
        total_agents = sum(m.get('num_agents', 0) for m in valid)
        if total_agents == 0:
            return {}
        util = sum(m['utilization'] * m['num_agents'] for m in valid) / total_agents
        deliveries = sum(m.get('cumulative_deliveries', 0) for m in valid)
        disp = sum(m.get('avg_displacement', 0) for m in valid) / len(valid)
        active = sum(m.get('active_pods', 0) for m in valid)
        total_pods = sum(m.get('total_pods', 0) for m in valid)
        merged_status: dict[str, int] = {}
        for m in valid:
            for s, cnt in m.get('status_counts', {}).items():
                merged_status[s] = merged_status.get(s, 0) + cnt
        return {
            'utilization': util,
            'cumulative_deliveries': deliveries,
            'avg_displacement': disp,
            'active_pods': active,
            'total_pods': total_pods,
            'status_counts': merged_status,
            'num_agents': total_agents,
        }

    def _update_metrics_display(self):
        if not self._simulation_mode or not self._last_metrics:
            return
        n = len(self._datasets)
        focused = self._focused_index
        if focused is not None:
            self._monitor_box.setTitle(
                f"Monitor [{focused}] {self._labels[focused]}")
        else:
            self._monitor_box.setTitle(f"Monitor (All {n})")
        m = self._last_metrics
        self._util_label.setText(f"Utilization: {m['utilization']:.1f}%")
        self._active_pods_label.setText(
            f"Active Pods: {m['active_pods']} / {m.get('total_pods', '?')}")
        self._deliveries_label.setText(
            f"Est. Deliveries: {m['cumulative_deliveries']}")
        self._displacement_label.setText(
            f"Avg Movement: {m['avg_displacement']:.2f}")
        parts = []
        for s in _STATUS_LABELS:
            cnt = m['status_counts'].get(s, 0)
            if cnt > 0:
                parts.append(f"{s}: {cnt}")
        self._status_dist_label.setText(
            " | ".join(parts) if parts else "All IDLE")


    def _toggle_charts(self):
        self._chart_visible = not self._chart_visible
        if self._chart_visible:
            self._chart_btn.setText("\U0001f4ca  Hide Charts")
            total_h = self._right_splitter.height()
            chart_h = max(180, total_h // 4)
            self._right_splitter.setSizes([0, total_h - chart_h, chart_h])
            self._chart_last_frame = -1
            self._redraw_charts()
        else:
            self._chart_btn.setText("\U0001f4ca  Show Charts")
            total_h = self._right_splitter.height()
            self._right_splitter.setSizes([0, total_h, 0])
        QTimer.singleShot(50, self._resize_panda)

    # ── 指标计算 ──────────────────────────────────────────────────

    def _compute_card_metrics(self, index: int):
        rw = self._replay_worlds[index]
        data = self._datasets[index]
        frame_idx = rw.current_frame_index
        if frame_idx >= len(data.frames):
            return
        frame = data.frames[frame_idx]
        status_counts: dict[str, int] = {}
        carrying = 0
        for agent in frame["agents"]:
            s = agent.get("status", "IDLE")
            status_counts[s] = status_counts.get(s, 0) + 1
            if agent.get("pod") is not None:
                carrying += 1
        m = self._last_metrics_list[index] if self._last_metrics_list else {}
        self._cards[index].update_metrics(
            frame_idx, rw.total_frames, status_counts, carrying,
            utilization=m.get('utilization') if m else None,
            deliveries=m.get('cumulative_deliveries') if m else None)

    def _update_all_cards(self):
        for i in range(len(self._datasets)):
            self._compute_card_metrics(i)

    # ── 回放逻辑 ──────────────────────────────────────────────────

    def _on_timer(self):
        now = time.time()

        if not self._paused and now - self._last_advance_time >= self._frame_interval:
            if self._focused_index is not None:
                rw = self._replay_worlds[self._focused_index]
                new_frame = rw.current_frame_index + self._speed
                if new_frame >= rw.total_frames:
                    new_frame = rw.total_frames - 1
                    self._paused = True
                    self._play_btn.setText("▶  Play")
                    self._status_label.setText("⏹  END")
                    self._status_label.setStyleSheet(
                        "color: #e74c3c; font-size: 14px; font-weight: bold;")
                rw.set_frame(new_frame)
            else:
                new_frame = self._current_frame + self._speed
                if new_frame >= self._max_frames:
                    new_frame = self._max_frames - 1
                    self._paused = True
                    self._play_btn.setText("▶  Play")
                    self._status_label.setText("⏹  END")
                    self._status_label.setStyleSheet(
                        "color: #e74c3c; font-size: 14px; font-weight: bold;")
                self._current_frame = new_frame
                for rw in self._replay_worlds:
                    rw.set_frame(min(new_frame, rw.total_frames - 1))
            self._last_advance_time = now

            # 计算仪表盘指标
            if self._simulation_mode:
                for i, rw in enumerate(self._replay_worlds):
                    if self._current_frame < rw.total_frames:
                        self._last_metrics_list[i] = self._metrics_computers[i].compute(rw.current_frame_index)
                if self._focused_index is not None:
                    self._last_metrics = self._last_metrics_list[self._focused_index]
                else:
                    self._last_metrics = self._compute_aggregated_metrics()
                self._update_metrics_display()
                self._dashboard.update(self._last_metrics_list, self._labels)

        # 聚焦模式：驱动 Panda3D
        if self._focused_index is not None and self._viz is not None and self._viz._initialised:
            world = self._replay_worlds[self._focused_index]
            self._viz._update_agents(world)
            self._viz._update_pods(world)
            self._viz._update_hud(world)
            self._viz._app.taskMgr.step()
            if self._chart_visible:
                self._redraw_charts()
        else:
            self._update_all_cards()

        self._update_info()

    def _step(self, delta: int):
        if self._focused_index is not None:
            rw = self._replay_worlds[self._focused_index]
            new_frame = max(0, min(rw.total_frames - 1,
                                   rw.current_frame_index + delta))
            rw.set_frame(new_frame)
            self._frame_slider.blockSignals(True)
            self._frame_slider.setValue(new_frame)
            self._frame_slider.blockSignals(False)
        else:
            new_frame = max(0, min(self._max_frames - 1,
                                   self._current_frame + delta))
            self._current_frame = new_frame
            for rw in self._replay_worlds:
                rw.set_frame(min(new_frame, rw.total_frames - 1))
            self._frame_slider.blockSignals(True)
            self._frame_slider.setValue(new_frame)
            self._frame_slider.blockSignals(False)

    def _update_info(self):
        if self._focused_index is not None:
            rw = self._replay_worlds[self._focused_index]
            idx = rw.current_frame_index
            if self._simulation_mode:
                self._frame_label.setText(f"Tick: {self._current_frame}")
            else:
                self._frame_label.setText(f"Frame: {idx} / {rw.total_frames - 1}")
        else:
            idx = self._current_frame
            if self._simulation_mode:
                self._frame_label.setText(f"Tick: {self._current_frame}")
            else:
                self._frame_label.setText(f"Frame: {idx} / {self._max_frames - 1}")
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)

    # ── 控件回调 ──────────────────────────────────────────────────

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self._play_btn.setText("▶  Play")
            self._status_label.setText("⏸  PAUSED")
            self._status_label.setStyleSheet(
                "color: #f0a500; font-size: 14px; font-weight: bold;")
        else:
            if self._focused_index is not None:
                rw = self._replay_worlds[self._focused_index]
                if rw.current_frame_index >= rw.total_frames - 1:
                    rw.set_frame(0)
            else:
                if self._current_frame >= self._max_frames - 1:
                    self._current_frame = 0
                if self._simulation_mode:
                    for mc in self._metrics_computers:
                        mc.reset()
                    self._last_metrics_list = [{} for _ in self._datasets]
                    for rw in self._replay_worlds:
                        rw.set_frame(0)
            self._play_btn.setText("⏸  Pause")
            self._status_label.setText("▶  PLAYING")
            self._status_label.setStyleSheet(
                "color: #2ecc71; font-size: 14px; font-weight: bold;")
            self._last_advance_time = time.time()

    def _on_frame_slider(self, value):
        if self._focused_index is not None:
            self._replay_worlds[self._focused_index].set_frame(value)
        else:
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
        if event.key() == Qt.Key.Key_Escape:
            self._unfocus()
        elif event.key() == Qt.Key.Key_Space:
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
            if self._focused_index is not None:
                self._replay_worlds[self._focused_index].set_frame(0)
            else:
                self._current_frame = 0
                for rw in self._replay_worlds:
                    rw.set_frame(0)
        elif event.key() == Qt.Key.Key_End:
            if self._focused_index is not None:
                rw = self._replay_worlds[self._focused_index]
                rw.set_frame(rw.total_frames - 1)
            else:
                self._current_frame = self._max_frames - 1
                for rw in self._replay_worlds:
                    rw.set_frame(min(self._max_frames - 1, rw.total_frames - 1))
        elif event.key() == Qt.Key.Key_C:
            if self._focused_index is not None:
                self._toggle_charts()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        event.accept()

    # ── 公共 API ────────────────────────────────────────────────

    def run(self):
        n = len(self._datasets)
        print("=" * 60)
        print(f"MAS-RMFS List-Mode Replay")
        print(f"  Files: {n}")
        print("-" * 60)
        for i, data in enumerate(self._datasets):
            print(f"  [{i}] {self._labels[i]}")
            print(f"      Map: {data.rows}x{data.cols}, "
                  f"Agents: {data.num_agents}, Frames: {data.total_ticks}")
        print("=" * 60)
        self.show()
        self._qt_app.exec()