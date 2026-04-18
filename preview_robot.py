"""
Preview the robot model (body + 4 wheels) in a standalone Panda3D window.
Left-drag: orbit, right-drag: pan, scroll: zoom.
"""

import math
import os

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    PerspectiveLens, CardMaker, TextNode, LVecBase4f,
    TransparencyAttrib, AntialiasAttrib, NodePath, LineSegs,
    GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomNode, GeomTriangles, Filename,
)

CELL = 1.0
_WHEEL_CLR = LVecBase4f(0.25, 0.25, 0.28, 1)
_BODY_CLR = LVecBase4f(0.12, 0.47, 0.71, 1)


def _make_box(name, sx, sy, sz):
    root = NodePath(name)
    root.setTwoSided(True)
    hx, hy, hz = sx / 2, sy / 2, sz / 2

    cm_top = CardMaker(name + "_top")
    cm_top.setFrame(-hx, hx, -hy, hy)
    top = root.attachNewNode(cm_top.generate())
    top.setP(-90); top.setPos(0, 0, hz)
    bot = root.attachNewNode(cm_top.generate())
    bot.setP(90); bot.setPos(0, 0, -hz)

    cm_fwd = CardMaker(name + "_fwd")
    cm_fwd.setFrame(-hx, hx, -hz, hz)
    front = root.attachNewNode(cm_fwd.generate())
    front.setPos(0, -hy, 0)
    back = root.attachNewNode(cm_fwd.generate())
    back.setH(180); back.setPos(0, hy, 0)

    cm_side = CardMaker(name + "_side")
    cm_side.setFrame(-hy, hy, -hz, hz)
    left = root.attachNewNode(cm_side.generate())
    left.setH(90); left.setPos(-hx, 0, 0)
    right = root.attachNewNode(cm_side.generate())
    right.setH(-90); right.setPos(hx, 0, 0)
    return root


def _make_cylinder(name, radius, height, segments=12):
    fmt = GeomVertexFormat.getV3t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    n = segments
    vdata.setNumRows(n * 2 + 2)
    vertex = GeomVertexWriter(vdata, "vertex")
    texcoord = GeomVertexWriter(vdata, "texcoord")

    for i in range(n):
        a = 2.0 * math.pi * i / n
        x = radius * math.cos(a)
        y = radius * math.sin(a)
        u = i / n
        vertex.addData3(x, y, height / 2)
        texcoord.addData2(u, 1)
        vertex.addData3(x, y, -height / 2)
        texcoord.addData2(u, 0)

    vertex.addData3(0, 0, height / 2)
    texcoord.addData2(0.5, 0.5)
    vertex.addData3(0, 0, -height / 2)
    texcoord.addData2(0.5, 0.5)

    tris = GeomTriangles(Geom.UHStatic)
    tc, bc = n * 2, n * 2 + 1
    for i in range(n):
        ni = (i + 1) % n
        t0, b0 = i * 2, i * 2 + 1
        t1, b1 = ni * 2, ni * 2 + 1
        tris.addVertices(t0, b0, b1)
        tris.addVertices(t0, b1, t1)
        tris.addVertices(tc, t0, t1)
        tris.addVertices(bc, b1, b0)

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def _make_robot(name, wheel_model=None):
    root = NodePath(name)
    body_w = CELL * 0.50
    body_d = CELL * 0.50
    body_h = CELL * 0.30
    wheel_r = CELL * 0.07

    body = _make_box(name + "_body", body_w, body_d, body_h)
    body.reparentTo(root)
    body.setPos(0, 0, wheel_r + body_h / 2)

    wx = body_w * 0.42
    wy = body_d * 0.42
    for i, (dx, dy) in enumerate([(-wx, -wy), (wx, -wy), (-wx, wy), (wx, wy)]):
        if wheel_model is not None:
            pivot = root.attachNewNode(f"{name}_wp{i}")
            pivot.setScale(wheel_r / 217.0)
            pivot.setR(90)
            pivot.setPos(dx, dy, wheel_r)
            w = wheel_model.copyTo(pivot)
            w.setPos(0, 0, -80.5)
        else:
            w = _make_cylinder(f"{name}_w{i}", wheel_r, CELL * 0.05)
            w.reparentTo(root)
            w.setR(90)
            w.setPos(dx, dy, wheel_r)
            w.setColor(_WHEEL_CLR)
    return root


class PreviewApp(ShowBase):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(0.08, 0.08, 0.14, 1)
        self.render.setAntialias(AntialiasAttrib.MMultisample)

        self.disableMouse()
        lens = PerspectiveLens()
        lens.setFov(45)
        lens.setNearFar(0.1, 100)
        self.cam.node().setLens(lens)

        # orbit state
        self._pivot = (0.0, 0.0, 0.15)
        self._heading = -135.0
        self._pitch = 25.0
        self._dist = 1.5
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
        for i in range(-2, 3):
            v = i * 0.25
            ls.moveTo(-0.5, v, 0); ls.drawTo(0.5, v, 0)
            ls.moveTo(v, -0.5, 0); ls.drawTo(v, 0.5, 0)
        self.render.attachNewNode(ls.create())

        # wheel model
        wheel_model = None
        base_dir = os.path.dirname(os.path.abspath(__file__))
        egg_path = os.path.join(
            base_dir, "Visualization", "models", "car_wheel",
            "meshes", "car_wheel.egg",
        )
        tex_path = os.path.join(
            base_dir, "Visualization", "models", "car_wheel",
            "materials", "textures", "car_wheel.png",
        )
        if os.path.isfile(egg_path):
            wheel_model = self.loader.loadModel(Filename.fromOsSpecific(egg_path))
            if os.path.isfile(tex_path):
                tex = self.loader.loadTexture(Filename.fromOsSpecific(tex_path))
                wheel_model.setTexture(tex, 1)
                wheel_model.setMaterialOff()

        # robot model
        robot = _make_robot("robot", wheel_model=wheel_model)
        robot.reparentTo(self.render)
        body_np = robot.find("**/robot_body")
        if body_np:
            body_np.setColor(_BODY_CLR)

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
        self._dist = max(0.3, min(20.0, self._dist))
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
