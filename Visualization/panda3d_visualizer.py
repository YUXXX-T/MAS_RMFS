"""
Panda3D Warehouse Visualizer
==============================
Real-time warehouse visualization using Panda3D.

Supports two camera modes controlled by ``view_mode``:

* ``"2d"`` — orthographic top-down.  Flat quads, no depth.
* ``"3d"`` — perspective camera at an isometric angle.
  Obstacles and stations are raised boxes; pods and robots
  are taller quads.  Mouse-orbit is enabled so the user can
  rotate, pan and zoom.

The window runs in the same process — the simulation engine
calls ``render()`` each tick, which updates node positions
and pumps the Panda3D event loop once.
"""

import math
from typing import TYPE_CHECKING

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    OrthographicLens,
    PerspectiveLens,
    CardMaker,
    TextNode,
    LVecBase4f,
    LMatrix4f,
    TransparencyAttrib,
    AntialiasAttrib,
    WindowProperties,
    NodePath,
    LineSegs,
    LPoint3f,
)

if TYPE_CHECKING:
    from WorldState.world import WorldState

from Visualization.visualizer import BaseVisualizer


# ─── colour palette ──────────────────────────────────────────────────
_CLR_BG       = LVecBase4f(0.08, 0.08, 0.14, 1)
_CLR_FREE     = LVecBase4f(0.12, 0.12, 0.20, 1)
_CLR_OBSTACLE = LVecBase4f(0.28, 0.28, 0.34, 1)
_CLR_STATION  = LVecBase4f(0.85, 0.22, 0.28, 1)
_CLR_POD_HOME = LVecBase4f(0.12, 0.22, 0.26, 1)
_CLR_POD      = LVecBase4f(0.18, 0.77, 0.71, 1)
_CLR_GRID     = LVecBase4f(0.20, 0.20, 0.33, 0.6)
_CLR_TEXT     = LVecBase4f(0.90, 0.90, 0.90, 1)

# Robot colours (tab10-like)
_ROBOT_COLOURS = [
    LVecBase4f(0.12, 0.47, 0.71, 1),   # blue
    LVecBase4f(1.00, 0.50, 0.05, 1),   # orange
    LVecBase4f(0.17, 0.63, 0.17, 1),   # green
    LVecBase4f(0.84, 0.15, 0.16, 1),   # red
    LVecBase4f(0.58, 0.40, 0.74, 1),   # purple
    LVecBase4f(0.55, 0.34, 0.29, 1),   # brown
    LVecBase4f(0.89, 0.47, 0.76, 1),   # pink
    LVecBase4f(0.50, 0.50, 0.50, 1),   # grey
    LVecBase4f(0.74, 0.74, 0.13, 1),   # olive
    LVecBase4f(0.09, 0.75, 0.81, 1),   # cyan
]

CELL = 1.0        # world-space size of one grid cell
PAD  = 0.02       # gap between cell quads


def _make_box(name: str, sx: float, sy: float, sz: float) -> NodePath:
    """Build a simple 6-face box from cards centred at origin."""
    root = NodePath(name)
    cm = CardMaker(name)
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    # top (+Z)
    cm.setFrame(-hx, hx, -hy, hy)
    top = root.attachNewNode(cm.generate())
    top.setP(-90)
    top.setPos(0, 0, hz)

    # bottom (-Z)
    bot = root.attachNewNode(cm.generate())
    bot.setP(90)
    bot.setPos(0, 0, -hz)

    # front (-Y)
    cm.setFrame(-hx, hx, -hz, hz)
    front = root.attachNewNode(cm.generate())
    front.setPos(0, -hy, 0)

    # back (+Y)
    back = root.attachNewNode(cm.generate())
    back.setH(180)
    back.setPos(0, hy, 0)

    # left (-X)
    cm.setFrame(-hy, hy, -hz, hz)
    left = root.attachNewNode(cm.generate())
    left.setH(90)
    left.setPos(-hx, 0, 0)

    # right (+X)
    right = root.attachNewNode(cm.generate())
    right.setH(-90)
    right.setPos(hx, 0, 0)

    return root


class Panda3DVisualizer(BaseVisualizer):
    """
    Warehouse visualizer with switchable 2D / 3D camera.

    Parameters
    ----------
    view_mode : str
        ``"2d"`` for orthographic top-down (default),
        ``"3d"`` for isometric perspective with mouse orbit.
    """

    def __init__(self, view_mode: str = "2d"):
        self._view_mode = view_mode.lower()
        self._app: ShowBase | None = None
        self._initialised = False

        # Scene-graph references kept across frames
        self._pod_nodes: dict[int, NodePath] = {}
        self._agent_nodes: dict[int, NodePath] = {}
        self._agent_labels: dict[int, NodePath] = {}
        self._glow_nodes: dict[int, NodePath] = {}
        self._hud_text: TextNode | None = None
        self._hud_np: NodePath | None = None

    # ── public API ────────────────────────────────────────────────────

    def render(self, world_state: "WorldState"):
        """Called once per tick by the simulation engine."""
        if not self._initialised:
            self._setup(world_state)

        self._update_pods(world_state)
        self._update_agents(world_state)
        self._update_hud(world_state)

        # Pump one frame so the window stays responsive
        self._app.taskMgr.step()

    # ── initialisation ────────────────────────────────────────────────

    def _setup(self, world_state: "WorldState"):
        from WorldState.map_state import CellType

        ms = world_state.map_state
        rows, cols = ms.rows, ms.cols
        is_3d = self._view_mode == "3d"

        # ---- ShowBase ----
        self._app = ShowBase()
        self._app.setBackgroundColor(_CLR_BG)

        wp = WindowProperties()
        mode_label = "3D" if is_3d else "2D"
        wp.setTitle(f"MAS-RMFS  —  Panda3D {mode_label}")
        wp.setSize(1024, 768) if is_3d else wp.setSize(900, 900)
        self._app.win.requestProperties(wp)

        self._app.render.setAntialias(AntialiasAttrib.MMultisample)

        # Centre of the grid in world coords
        cx = (cols - 1) * CELL / 2
        cz = -((rows - 1) * CELL / 2)

        if is_3d:
            # ---- 3D perspective with mouse orbit ----
            lens = PerspectiveLens()
            lens.setFov(50)
            lens.setNearFar(0.5, 500)
            self._app.cam.node().setLens(lens)

            dist = max(rows, cols) * CELL * 1.3

            # Position camera, then transfer transform to trackball
            self._app.disableMouse()
            self._app.camera.setPos(cx + dist * 0.7, -dist * 0.7, cz + dist * 0.8)
            self._app.camera.lookAt(cx, 0, cz)
            cam_mat = self._app.camera.getMat(self._app.render)
            self._app.enableMouse()
            inv_mat = LMatrix4f()
            inv_mat.invertFrom(cam_mat)
            self._app.mouseInterfaceNode.setMat(inv_mat)
        else:
            # ---- 2D orthographic ----
            self._app.disableMouse()
            lens = OrthographicLens()
            half_w = cols * CELL / 2 + 1.0
            half_h = rows * CELL / 2 + 1.0
            lens.setFilmSize(half_w * 2, half_h * 2)
            lens.setNearFar(-100, 100)
            self._app.cam.node().setLens(lens)

            self._app.cam.setPos(cx, -10, cz)
            self._app.cam.lookAt(cx, 0, cz)

        # ---- Static grid cells ----
        static_root = self._app.render.attachNewNode("static_grid")

        if is_3d:
            self._build_grid_3d(static_root, ms, rows, cols)
        else:
            self._build_grid_2d(static_root, ms, rows, cols)

        # ---- Station labels ----
        for sid, (sr, sc) in ms.station_positions.items():
            tn = TextNode(f"station_{sid}")
            tn.setText(f"S{sid}")
            tn.setTextColor(1, 0.42, 0.50, 1)
            tn.setAlign(TextNode.ACenter)
            tn.setCardColor(0, 0, 0, 0.5)
            tn.setCardAsMargin(0.05, 0.05, 0.05, 0.05)
            tn.setCardDecal(True)
            tnp = static_root.attachNewNode(tn)
            if is_3d:
                tnp.setPos(sc * CELL, 0, -sr * CELL + CELL * 0.35)
                tnp.setScale(0.25)
                tnp.setBillboardPointEye()
            else:
                tnp.setPos(sc * CELL, -0.1, -sr * CELL + CELL * 0.35)
                tnp.setScale(0.25)

        # ---- Grid lines (3D floor lines) ----
        if is_3d:
            self._draw_grid_lines(static_root, rows, cols)

        # ---- Pod nodes (dynamic) ----
        pod_root = self._app.render.attachNewNode("pods")
        for pod in world_state.pod_state.pods.values():
            pr, pc = pod.current_position
            if is_3d:
                np = _make_box("pod", CELL * 0.6, CELL * 0.6, CELL * 0.35)
                np.reparentTo(pod_root)
                np.setPos(pc * CELL, 0, -pr * CELL + CELL * 0.175)
            else:
                pod_cm = CardMaker("pod")
                s = CELL * 0.35
                pod_cm.setFrame(-s, s, -s, s)
                np = pod_root.attachNewNode(pod_cm.generate())
                np.setPos(pc * CELL, -0.2, -pr * CELL)
            np.setColor(_CLR_POD)
            np.setTransparency(TransparencyAttrib.MAlpha)
            self._pod_nodes[pod.pod_id] = np

        # ---- Agent nodes (dynamic) ----
        agent_root = self._app.render.attachNewNode("agents")
        for agent in world_state.agents:
            clr = _ROBOT_COLOURS[agent.agent_id % len(_ROBOT_COLOURS)]

            if is_3d:
                # Glow (larger translucent box)
                gnp = _make_box("glow", CELL * 0.75, CELL * 0.75, CELL * 0.55)
                gnp.reparentTo(agent_root)
                gnp.setColor(clr[0], clr[1], clr[2], 0.2)
                gnp.setTransparency(TransparencyAttrib.MAlpha)
                gnp.hide()
                self._glow_nodes[agent.agent_id] = gnp

                # Agent box
                anp = _make_box("agent", CELL * 0.55, CELL * 0.55, CELL * 0.45)
                anp.reparentTo(agent_root)
                anp.setColor(clr)
                self._agent_nodes[agent.agent_id] = anp

                # Billboard label
                tn = TextNode(f"agent_{agent.agent_id}")
                tn.setText(str(agent.agent_id))
                tn.setTextColor(1, 1, 1, 1)
                tn.setAlign(TextNode.ACenter)
                lnp = agent_root.attachNewNode(tn)
                lnp.setScale(0.25)
                lnp.setBillboardPointEye()
                self._agent_labels[agent.agent_id] = lnp
            else:
                # 2D — flat cards
                glow_cm = CardMaker("glow")
                g = CELL * 0.55
                glow_cm.setFrame(-g, g, -g, g)
                gnp = agent_root.attachNewNode(glow_cm.generate())
                gnp.setColor(clr[0], clr[1], clr[2], 0.25)
                gnp.setTransparency(TransparencyAttrib.MAlpha)
                gnp.hide()
                self._glow_nodes[agent.agent_id] = gnp

                agent_cm = CardMaker("agent")
                a = CELL * 0.4
                agent_cm.setFrame(-a, a, -a, a)
                anp = agent_root.attachNewNode(agent_cm.generate())
                anp.setColor(clr)
                self._agent_nodes[agent.agent_id] = anp

                tn = TextNode(f"agent_{agent.agent_id}")
                tn.setText(str(agent.agent_id))
                tn.setTextColor(1, 1, 1, 1)
                tn.setAlign(TextNode.ACenter)
                lnp = agent_root.attachNewNode(tn)
                lnp.setScale(0.22)
                self._agent_labels[agent.agent_id] = lnp

        # ---- HUD text ----
        self._hud_text = TextNode("hud")
        self._hud_text.setTextColor(_CLR_TEXT)
        self._hud_text.setAlign(TextNode.ALeft)
        self._hud_text.setShadow(0.05, 0.05)
        self._hud_text.setShadowColor(0, 0, 0, 0.8)
        self._hud_np = self._app.aspect2d.attachNewNode(self._hud_text)
        self._hud_np.setScale(0.05)
        self._hud_np.setPos(-1.3, 0, 0.92)

        self._initialised = True
        self._is_3d = is_3d

    # ── grid builders ─────────────────────────────────────────────────

    def _build_grid_2d(self, parent, ms, rows, cols):
        from WorldState.map_state import CellType
        cm = CardMaker("cell")
        cm.setFrame(-CELL / 2 + PAD, CELL / 2 - PAD,
                     -CELL / 2 + PAD, CELL / 2 - PAD)
        for r in range(rows):
            for c in range(cols):
                cell = ms.grid[r][c]
                if cell == CellType.OBSTACLE:
                    clr = _CLR_OBSTACLE
                elif cell == CellType.STATION:
                    clr = _CLR_STATION
                elif cell == CellType.POD_HOME:
                    clr = _CLR_POD_HOME
                else:
                    clr = _CLR_FREE
                np = parent.attachNewNode(cm.generate())
                np.setPos(c * CELL, 0, -r * CELL)
                np.setColor(clr)

    def _build_grid_3d(self, parent, ms, rows, cols):
        from WorldState.map_state import CellType

        # Floor tiles (thin cards in the XY plane, rotated flat)
        floor_cm = CardMaker("floor")
        floor_cm.setFrame(-CELL / 2 + PAD, CELL / 2 - PAD,
                          -CELL / 2 + PAD, CELL / 2 - PAD)

        for r in range(rows):
            for c in range(cols):
                cell = ms.grid[r][c]

                if cell == CellType.OBSTACLE:
                    # Raised box
                    box = _make_box("obs", CELL - PAD * 2, CELL - PAD * 2, CELL * 0.5)
                    box.reparentTo(parent)
                    box.setPos(c * CELL, 0, -r * CELL + CELL * 0.25)
                    box.setColor(_CLR_OBSTACLE)
                elif cell == CellType.STATION:
                    # Flat red tile on the floor + a small raised marker
                    np = parent.attachNewNode(floor_cm.generate())
                    np.setP(-90)
                    np.setPos(c * CELL, 0, -r * CELL)
                    np.setColor(_CLR_STATION)
                elif cell == CellType.POD_HOME:
                    np = parent.attachNewNode(floor_cm.generate())
                    np.setP(-90)
                    np.setPos(c * CELL, 0, -r * CELL)
                    np.setColor(_CLR_POD_HOME)
                else:
                    np = parent.attachNewNode(floor_cm.generate())
                    np.setP(-90)
                    np.setPos(c * CELL, 0, -r * CELL)
                    np.setColor(_CLR_FREE)

    def _draw_grid_lines(self, parent, rows, cols):
        """Draw thin white grid lines on the floor for 3D view."""
        ls = LineSegs("grid_lines")
        ls.setColor(0.25, 0.25, 0.40, 0.4)
        ls.setThickness(1.0)

        min_x = -CELL / 2
        max_x = (cols - 1) * CELL + CELL / 2
        min_z = -((rows - 1) * CELL + CELL / 2)
        max_z = CELL / 2

        # Horizontal lines
        for r in range(rows + 1):
            z = CELL / 2 - r * CELL
            ls.moveTo(min_x, 0.01, z)
            ls.drawTo(max_x, 0.01, z)

        # Vertical lines
        for c in range(cols + 1):
            x = -CELL / 2 + c * CELL
            ls.moveTo(x, 0.01, max_z)
            ls.drawTo(x, 0.01, min_z)

        parent.attachNewNode(ls.create())

    # ── per-frame updates ────────────────────────────────────────────

    def _update_pods(self, world_state):
        for pod in world_state.pod_state.pods.values():
            np = self._pod_nodes.get(pod.pod_id)
            if np is None:
                continue
            if pod.is_carried:
                np.hide()
            else:
                np.show()
                pr, pc = pod.current_position
                if self._is_3d:
                    np.setPos(pc * CELL, 0, -pr * CELL + CELL * 0.175)
                else:
                    np.setPos(pc * CELL, -0.2, -pr * CELL)

    def _update_agents(self, world_state):
        for agent in world_state.agents:
            aid = agent.agent_id
            r, c = agent.position
            x = c * CELL
            z = -r * CELL

            anp = self._agent_nodes[aid]
            gnp = self._glow_nodes[aid]
            lnp = self._agent_labels[aid]

            if self._is_3d:
                anp.setPos(x, 0, z + CELL * 0.225)
                gnp.setPos(x, 0, z + CELL * 0.275)
                lnp.setPos(x, 0, z + CELL * 0.55)
            else:
                anp.setPos(x, -0.5, z)
                gnp.setPos(x, -0.4, z)
                lnp.setPos(x, -0.6, z + CELL * 0.02)

            if agent.carried_pod_id is not None:
                gnp.show()
            else:
                gnp.hide()

    def _update_hud(self, world_state):
        os = world_state.order_state
        completed = os.total_completed
        total = os.total_orders
        rate = completed / max(world_state.tick, 1)
        self._hud_text.setText(
            f"Tick: {world_state.tick}   "
            f"Orders: {completed}/{total}   "
            f"Rate: {rate:.2f}/tick"
        )
