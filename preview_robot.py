"""
Preview the Jackal robot model in a standalone Panda3D window.
Left-drag: orbit, right-drag: pan, scroll: zoom.
"""

import math
import os
import sys

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    PerspectiveLens, TextNode, LVecBase4f,
    AntialiasAttrib, NodePath, LineSegs, Filename,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Visualization.stl_loader import load_stl
from Config.config_loader import load_config

_JACKAL_BASE_CLR = LVecBase4f(0.2, 0.2, 0.2, 1)
_JACKAL_FENDER_CLR = LVecBase4f(0.12, 0.47, 0.71, 1)
_WHEEL_CLR = LVecBase4f(0.25, 0.25, 0.28, 1)


def _make_box(name, sx, sy, sz):
    from panda3d.core import CardMaker
    root = NodePath(name)
    root.setTwoSided(True)
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    cm = CardMaker(name + "_top")
    cm.setFrame(-hx, hx, -hy, hy)
    top = root.attachNewNode(cm.generate())
    top.setP(-90); top.setPos(0, 0, hz)
    bot = root.attachNewNode(cm.generate())
    bot.setP(90); bot.setPos(0, 0, -hz)
    cm2 = CardMaker(name + "_fwd")
    cm2.setFrame(-hx, hx, -hz, hz)
    front = root.attachNewNode(cm2.generate())
    front.setPos(0, -hy, 0)
    back = root.attachNewNode(cm2.generate())
    back.setH(180); back.setPos(0, hy, 0)
    cm3 = CardMaker(name + "_side")
    cm3.setFrame(-hy, hy, -hz, hz)
    left = root.attachNewNode(cm3.generate())
    left.setH(90); left.setPos(-hx, 0, 0)
    right = root.attachNewNode(cm3.generate())
    right.setH(-90); right.setPos(hx, 0, 0)
    return root


def _make_robot(name, rm_cfg, base_model=None, fenders_model=None, wheel_model=None):
    root = NodePath(name)

    if base_model is not None and fenders_model is not None:
        offset_z = rm_cfg.body_offset_z
        hpr = rm_cfg.body_hpr
        wheel_r = rm_cfg.wheel_radius
        wheel_pos = rm_cfg.wheel_positions

        base_pivot = root.attachNewNode(name + "_base")
        base_pivot.setPos(0, 0, offset_z)
        base_pivot.setHpr(*hpr)
        b = base_model.copyTo(base_pivot)
        b.setColor(_JACKAL_BASE_CLR)

        fenders_pivot = root.attachNewNode(name + "_fenders")
        fenders_pivot.setPos(0, 0, offset_z)
        fenders_pivot.setHpr(*hpr)
        f = fenders_model.copyTo(fenders_pivot)
        f.setColor(_JACKAL_FENDER_CLR)

        for i, (wx, wy, wz) in enumerate(wheel_pos):
            if wheel_model is not None:
                pivot = root.attachNewNode(f"{name}_wp{i}")
                pivot.setPos(wx, wy, wz)
                pivot.setScale(wheel_r / 217.0)
                pivot.setR(90)
                w = wheel_model.copyTo(pivot)
                w.setPos(0, 0, -80.5)
    else:
        body = _make_box(name + "_body", 0.50, 0.50, 0.30)
        body.reparentTo(root)
        body.setPos(0, 0, 0.07 + 0.15)
        body.setColor(_JACKAL_FENDER_CLR)
    return root


class PreviewApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(0.08, 0.08, 0.14, 1)
        self.render.setAntialias(AntialiasAttrib.MMultisample)

        self.disableMouse()
        lens = PerspectiveLens()
        lens.setFov(45)
        lens.setNearFar(0.01, 100)
        self.cam.node().setLens(lens)

        self._pivot = (0.0, 0.0, 0.1)
        self._heading = -135.0
        self._pitch = 25.0
        self._dist = 0.8
        self._mouse_btn = 0
        self._mouse_prev = None
        self._update_camera()

        self.accept("mouse1", self._on_down, [1])
        self.accept("mouse1-up", self._on_up, [1])
        self.accept("mouse3", self._on_down, [3])
        self.accept("mouse3-up", self._on_up, [3])
        self.accept("wheel_up", self._on_zoom, [-1])
        self.accept("wheel_down", self._on_zoom, [1])
        self.taskMgr.add(self._orbit_task, "orbit")

        # ground grid
        ls = LineSegs("grid")
        ls.setColor(0.25, 0.25, 0.40, 0.5)
        ls.setThickness(1.0)
        for i in range(-4, 5):
            v = i * 0.1
            ls.moveTo(-0.4, v, 0); ls.drawTo(0.4, v, 0)
            ls.moveTo(v, -0.4, 0); ls.drawTo(v, 0.4, 0)
        self.render.attachNewNode(ls.create())

        # load config
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(base_dir, "Config", "default_config.json")
        cfg = load_config(cfg_path)
        rm_cfg = cfg.robot_model

        # load models
        vis_dir = os.path.join(base_dir, "Visualization")
        base_model = None
        fenders_model = None
        wheel_model = None

        if rm_cfg.use_model:
            base_stl = os.path.join(vis_dir, "models", "jackal", "jackal-base.stl")
            fenders_stl = os.path.join(vis_dir, "models", "jackal", "jackal-fenders.stl")
            if os.path.isfile(base_stl) and os.path.isfile(fenders_stl):
                base_model = load_stl(base_stl, "jackal_base")
                fenders_model = load_stl(fenders_stl, "jackal_fenders")

            egg_path = os.path.join(vis_dir, "models", "car_wheel", "meshes", "car_wheel.egg")
            tex_path = os.path.join(vis_dir, "models", "car_wheel", "materials", "textures", "car_wheel.png")
            if os.path.isfile(egg_path):
                wheel_model = self.loader.loadModel(Filename.fromOsSpecific(egg_path))
                if os.path.isfile(tex_path):
                    tex = self.loader.loadTexture(Filename.fromOsSpecific(tex_path))
                    wheel_model.setTexture(tex, 1)
                    wheel_model.setMaterialOff()

        robot = _make_robot("robot", rm_cfg, base_model, fenders_model, wheel_model)
        robot.reparentTo(self.render)

        # axis lines
        for label, end, clr in [
            ("X", (0.3, 0, 0), (1, 0.3, 0.3, 1)),
            ("Y", (0, 0.3, 0), (0.3, 1, 0.3, 1)),
            ("Z", (0, 0, 0.3), (0.4, 0.5, 1, 1)),
        ]:
            ax = LineSegs(label)
            ax.setColor(*clr)
            ax.setThickness(2)
            ax.moveTo(0, 0, 0)
            ax.drawTo(*end)
            self.render.attachNewNode(ax.create())

        # HUD
        tn = TextNode("info")
        tn.setText("Left-drag: Orbit | Right-drag: Pan | Scroll: Zoom")
        tn.setTextColor(0.7, 0.7, 0.8, 0.8)
        tn.setAlign(TextNode.ACenter)
        tnp = self.aspect2d.attachNewNode(tn)
        tnp.setScale(0.04)
        tnp.setPos(0, 0, -0.95)

    def _update_camera(self):
        h = math.radians(self._heading)
        p = math.radians(self._pitch)
        d = self._dist
        cp = math.cos(p)
        cx = self._pivot[0] + d * cp * math.sin(h)
        cy = self._pivot[1] - d * cp * math.cos(h)
        cz = self._pivot[2] + d * math.sin(p)
        self.camera.setPos(cx, cy, cz)
        self.camera.lookAt(*self._pivot)

    def _on_down(self, btn):
        self._mouse_btn = btn
        self._mouse_prev = None

    def _on_up(self, btn):
        if self._mouse_btn == btn:
            self._mouse_btn = 0
            self._mouse_prev = None

    def _on_zoom(self, d):
        self._dist *= 1.15 if d > 0 else 1 / 1.15
        self._dist = max(0.1, min(20.0, self._dist))
        self._update_camera()

    def _orbit_task(self, task):
        if not self.mouseWatcherNode.hasMouse():
            return task.cont
        mx = self.mouseWatcherNode.getMouseX()
        my = self.mouseWatcherNode.getMouseY()
        if self._mouse_btn == 0 or self._mouse_prev is None:
            self._mouse_prev = (mx, my)
            return task.cont
        dx = mx - self._mouse_prev[0]
        dy = my - self._mouse_prev[1]
        self._mouse_prev = (mx, my)
        if self._mouse_btn == 1:
            self._heading += dx * 150
            self._pitch += dy * 100
            self._pitch = max(5, min(85, self._pitch))
        elif self._mouse_btn == 3:
            h = math.radians(self._heading)
            s = self._dist * 0.5
            rx, ry = math.cos(h), math.sin(h)
            self._pivot = (
                self._pivot[0] - dx * s * rx,
                self._pivot[1] - dx * s * ry,
                self._pivot[2] + dy * s,
            )
        self._update_camera()
        return task.cont


if __name__ == "__main__":
    app = PreviewApp()
    app.run()
