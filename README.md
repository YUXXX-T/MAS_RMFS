# 🤖 MAS-RMFS：多智能体机器人移动履行系统仿真

多智能体仓储机器人仿真平台，支持通过 JSON 配置文件灵活切换订单生成、路径规划与任务分配算法，无需修改源代码。

---

## 📑 目录

- [🏗️ 系统结构](#-系统结构)
- [🚀 运行方式](#-运行方式)
- [⚙️ 配置文件说明](#️-配置文件说明)
- [🔄 系统运行流程](#-系统运行流程)
- [🧩 如何集成自定义算法](#-如何集成自定义算法)
- [📦 已有算法列表](#-已有算法列表)
- [🎮 Panda3D 可视化](#-panda3d-可视化)

---

## 🏗️ 系统结构

```
MAS_RMFS/
├── main.py                       # 程序入口：解析命令行参数、加载配置、实例化策略、启动引擎
├── Config/
│   ├── config_loader.py          # 配置加载器：解析 JSON → 数据类
│   └── default_config.json       # 默认配置文件
├── Engine/
│   └── simulation_engine.py      # 仿真引擎：主循环与 tick 调度
├── WorldState/                   # 世界状态（数据模型层）
│   ├── world.py                  # WorldState：聚合所有子状态
│   ├── map_state.py              # 地图：网格、障碍物、工作站、货架区
│   ├── agent_state.py            # 智能体：位置、路径、等待计数器
│   ├── pod_state.py              # 货架（Pod）：位置、归属、搬运状态
│   ├── order_state.py            # 订单：包含的 Pod、目标工作站
│   └── task_state.py             # 任务：PICK / DELIVER / RETURN
├── Policies/                     # 策略层（算法可插拔）
│   ├── policy_registry.py        # 策略注册中心
│   ├── OrderGenerator/           # 订单生成器
│   │   ├── base_order_generator.py
│   │   ├── RandomOrderGenerator/
│   │   └── ZipfOrderGenerator/
│   ├── PathPlanner/              # 路径规划器
│   │   ├── base_path_planner.py
│   │   ├── AStarPathPlanner/
│   │   └── PrioritizedPathPlanner/
│   └── TaskAssigner/             # 任务分配器
│       ├── base_task_assigner.py
│       └── GreedyTaskAssigner/
├── Visualization/                # 可视化
│   ├── visualizer.py             # 终端 ASCII / Matplotlib 仪表盘
│   └── panda3d_visualizer.py     # Panda3D 2D/3D 可视化
└── Debug/
    └── logger.py                 # 日志工具
```

### 📋 核心模块职责

| 模块 | 📌 职责 |
|------|------|
| `Config` | 从 JSON 文件加载并验证配置参数 |
| `WorldState` | 维护仿真的全部状态（地图、智能体、货架、订单、任务） |
| `Engine` | 驱动仿真主循环，按固定顺序调用各策略 |
| `Policies` | 提供算法接口（抽象基类）和具体实现，通过注册中心按名称查找 |
| `Visualization` | 可选的实时可视化渲染 |

---

## 🚀 运行方式

```bash
# 使用默认配置运行（无可视化）
python main.py

# 使用 Matplotlib 仪表盘
python main.py --mpl

# 使用 Panda3D 可视化（2D 正交或 3D 透视，由配置决定）
python main.py --p3d

# 使用终端 ASCII 可视化
python main.py --visualize

# 指定自定义配置文件
python main.py --config path/to/my_config.json
```

> `--mpl`、`--p3d`、`--visualize` 三者互斥，只能选择其一。

按 `Ctrl+C` 可安全停止仿真并输出统计摘要。

---

## ⚙️ 配置文件说明

配置文件为 JSON 格式，包含四个顶层节点：

```jsonc
{
    "map": {
        "rows": 10,                     // 网格行数
        "cols": 10,                     // 网格列数
        "obstacles": [[2,5], [3,5]],    // 障碍物坐标 [row, col]
        "stations": [                   // 工作站列表
            {"id": 1, "row": 0, "col": 4}
        ],
        "pod_zones": [                  // 货架区定义
            {"origin_row": 3, "origin_col": 1, "num_rows": 2, "num_cols": 3}
        ]
    },
    "robots": {
        "num_robots": 3,                // 机器人数量
        "starts": [[0,0], [0,1], [0,2]],// 各机器人初始位置
        "speed": 1
    },
    "simulation": {
        "order_interval": 5,            // 每隔 N tick 生成订单
        "max_items_per_order": 2,       // 每个订单最多包含的 Pod 数
        "pickup_duration": 2,           // 拾取货架暂停 tick 数
        "dropoff_duration": 2,          // 放下货架暂停 tick 数
        "station_process_duration": 5,  // 工作站处理暂停 tick 数
        "tick_delay": 0.5,              // 每 tick 间隔秒数（0=全速）
        "p3d_view_mode": "3d",          // Panda3D 视角："2d" 或 "3d"
        "log_level": "INFO"
    },
    "policies": {
        // 支持两种写法：
        // 写法一：仅指定算法名（使用默认参数）
        "task_assigner": "GreedyTaskAssigner",

        // 写法二：同时指定算法名和参数
        "order_generator": {
            "name": "ZipfOrderGenerator",
            "params": { "zipf_param": 1.5 }
        },
        "path_planner": {
            "name": "PrioritizedPathPlanner",
            "params": { "max_horizon": 100, "goal_reserve": 10 }
        }
    }
}
```

---

## 🔄 系统运行流程

### 🟢 启动阶段（`main.py`）

```
1. 解析命令行参数（--config, --mpl, --visualize）
2. 调用 load_config() 加载 JSON 配置
3. import Policies      ← 触发所有算法的自动注册
4. get_policy() 按名称查找算法类
5. 实例化三大策略：OrderGenerator, TaskAssigner, PathPlanner
6. 创建 SimulationEngine 并调用 engine.run()
```

### 🔁 仿真主循环（`SimulationEngine._tick()`）

引擎按固定顺序在每个 tick 执行以下 8 个步骤：

```
┌─────────────────────────────────────────────────┐
│                 SimulationEngine.run()           │
│         while not shutdown:  _tick()             │
└────────────────────┬────────────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 1: 生成订单                    │
    │  OrderGenerator.generate(world)      │
    │  → 返回新订单列表，加入 OrderState    │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 2: 分配任务                    │
    │  TaskAssigner.assign(world)          │
    │  → 将订单拆分为 PICK/DELIVER/RETURN  │
    │    任务，分配给空闲智能体             │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 3: 路径规划与任务激活           │
    │  _plan_and_activate(tick)            │
    │  → 对每个有任务但无路径的智能体调用    │
    │    PathPlanner.plan(agent, goal, world)│
    │  → 跳过等待中(is_waiting)的智能体     │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 4: 移动智能体                  │
    │  _move_agents(tick)                  │
    │  → 每个智能体沿路径前进一步           │
    │  → 等待中的智能体不移动               │
    │  → 搬运中的货架跟随移动               │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 5: 冲突检测                    │
    │  _detect_conflicts(tick)             │
    │  → 检测顶点冲突（两个智能体同位置）   │
    │  → 检测对向冲突（两个智能体交换位置）  │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 6: 执行动作（含延时机制）       │
    │  _handle_actions(tick)               │
    │  → 到达目标后开始倒计时等待           │
    │  → 倒计时结束后执行 PICK/DELIVER/RETURN│
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 7: 检查订单完成                │
    │  _check_order_completion(tick)        │
    └────────────────┬────────────────────┘
                     │
    ┌────────────────▼────────────────────┐
    │  Step 8: 可视化渲染（可选）           │
    │  visualizer.render(world)            │
    └────────────────┬────────────────────┘
                     │
                world.advance_tick()
                     │
              ───回到 Step 1───
```

### 📦 任务生命周期

每个订单会被拆解为一组任务链，按顺序执行：

```
Order(pods=[A, B], station=S)
  │
  ├─→ Agent #1: PICK(pod=A) → DELIVER(pod=A, station=S) → RETURN(pod=A)
  └─→ Agent #2: PICK(pod=B) → DELIVER(pod=B, station=S) → RETURN(pod=B)
```

每个任务阶段：
1. **PICK（拾取）**：机器人移动到货架位置 → 暂停 `pickup_duration` tick → 拾起货架
2. **DELIVER（配送）**：机器人搬运货架到工作站 → 暂停 `station_process_duration` tick → 完成配送
3. **RETURN（归还）**：机器人将货架搬回原位 → 暂停 `dropoff_duration` tick → 放下货架

---

## 🧩 如何集成自定义算法

系统支持三种策略的自定义扩展：**订单生成器**、**路径规划器**、**任务分配器**。以下以添加新的路径规划器为例。

### 📁 第一步：创建算法子目录

在对应的类别文件夹下创建子目录：

```
Policies/PathPlanner/
  └── MyNewPlanner/
      ├── __init__.py
      └── my_new_planner.py
```

### ✏️ 第二步：实现算法类

继承对应的抽象基类，实现所有抽象方法：

```python
# Policies/PathPlanner/MyNewPlanner/my_new_planner.py

from Policies.PathPlanner.base_path_planner import BasePathPlanner

class MyNewPlanner(BasePathPlanner):
    """自定义路径规划器。"""

    def __init__(self, my_param: float = 1.0):
        self.my_param = my_param

    def plan(self, agent, goal, world_state):
        """
        计算从 agent.position 到 goal 的路径。

        参数：
            agent      - AgentState 对象（包含 position, carried_pod_id 等）
            goal       - 目标坐标 (row, col)
            world_state - WorldState 对象（包含地图、所有智能体状态等）

        返回：
            list[tuple[int, int]] - 路径坐标列表（不含起点），空列表表示未找到路径
        """
        # 在此实现你的路径规划算法
        path = []
        # ...
        return path
```

### 📤 第三步：创建 `__init__.py` 并导出

```python
# Policies/PathPlanner/MyNewPlanner/__init__.py

from .my_new_planner import MyNewPlanner
__all__ = ["MyNewPlanner"]
```

### 🔗 第四步：在类别 `__init__.py` 中注册

编辑 `Policies/PathPlanner/__init__.py`，添加导入和注册：

```python
from .MyNewPlanner import MyNewPlanner          # ← 新增

from Policies.policy_registry import register
register("path_planner", "MyNewPlanner", MyNewPlanner)  # ← 新增
```

### ✅ 第五步：在配置文件中启用

```json
{
    "policies": {
        "path_planner": {
            "name": "MyNewPlanner",
            "params": { "my_param": 2.5 }
        }
    }
}
```

完成！无需修改 `main.py` 或 `simulation_engine.py` 中的任何代码。

### 📐 三种策略的接口汇总

| 策略类型 | 基类 | 需实现的方法 | 方法签名 |
|---------|------|-------------|---------|
| 订单生成器 | `BaseOrderGenerator` | `generate()` | `generate(world_state) → list[Order]` |
| 任务分配器 | `BaseTaskAssigner` | `assign()` | `assign(world_state) → list[Task]` |
| 路径规划器 | `BasePathPlanner` | `plan()` | `plan(agent, goal, world_state) → list[tuple]` |

### 🏭 策略注册中心工作原理

```python
# Policies/policy_registry.py 提供三个函数：

register(category, name, cls)    # 注册算法类
get_policy(category, name)       # 按名称查找算法类（找不到时抛出 ValueError 并列出可用选项）
list_policies(category=None)     # 列出已注册的算法
```

当 `main.py` 执行 `import Policies` 时，各 `__init__.py` 中的 `register()` 调用会自动触发，将所有算法注册到中央注册表。之后通过 `get_policy()` 按配置文件中的名称查找对应的类。

---

## 📦 已有算法列表

### 🛒 订单生成器（OrderGenerator）

| 算法名 | 说明 | 可配参数 |
|--------|------|---------|
| `RandomOrderGenerator` | 均匀随机选择货架生成订单 | — |
| `ZipfOrderGenerator` | 按 Zipf 分布选择货架（模拟热门商品） | `zipf_param`（偏斜度，默认 1.5） |

### 🗺️ 路径规划器（PathPlanner）

| 算法名 | 说明 | 可配参数 |
|--------|------|---------|
| `AStarPathPlanner` | 单智能体 A* 算法 | — |
| `PrioritizedPathPlanner` | 优先级规划（时空 A* + 预留表） | `max_horizon`（搜索深度，默认 100）, `goal_reserve`（目标占用缓冲，默认 10） |

### 📋 任务分配器（TaskAssigner）

| 算法名 | 说明 | 可配参数 |
|--------|------|---------|
| `GreedyTaskAssigner` | 贪心分配：按距离选择最近的空闲智能体 | — |

---

## 🎮 Panda3D 可视化

使用 `--p3d` 标志启动 Panda3D 可视化窗口，支持 **2D 正交投影** 和 **3D 透视投影** 两种模式。

### 🔀 视角模式切换

通过配置文件中的 `p3d_view_mode` 参数切换：

```json
"simulation": {
    "p3d_view_mode": "3d",   // "2d" = 正交俯视，"3d" = 透视立体
    "tick_delay": 0.5         // 控制仿真速度（秒/tick）
}
```

### 📷 2D 模式

| 特性 | 说明 |
|------|------|
| 相机 | 正交投影，俯视全局 |
| 网格 | 平面色块（障碍物=灰色、工作站=红色、货架区=青影） |
| 货架 | 青色小方块，搬运时隐藏 |
| 机器人 | 彩色圆形 + ID 标签，搬运时显示发光环 |
| 鼠标 | 无交互 |

### 🌍 3D 模式

| 特性 | 说明 |
|------|------|
| 相机 | 透视投影，等距角度俯瞰 |
| 鼠标控制 | ⭐ **左键拖动** = 旋转，**右键拖动** = 平移，**滚轮** = 缩放 |
| 障碍物 | 立体方块，有高度感 |
| 地板 | 平铺色块 + 网格线 |
| 货架 | 3D 方块，浮在地板之上 |
| 机器人 | 3D 方块 + 发光环（搬运时） |
| 标签 | 🏷️ Billboard 效果，始终面向相机 |
| HUD | 📊 左上角显示 Tick / 订单数 / 完成率 |
| 坐标轴 | 🧭 左下角 3D 坐标系指示器（同步旋转） |
| 键盘快捷键 | ⌨️ `1`=俯视 `2`=正视 `3`=侧视 `4`=等距 `R`=重置 |



### 📝 TODO List
超大规模下——多进程解耦架构
```
如果仿真和渲染会互相拖慢，可以用 ZeroMQ 做进程间通信：
┌──────────────────┐    ZeroMQ (TCP/IPC)    ┌────────────────────┐
│ Python 仿真进程   │ ────────────────────▶ │ 渲染进程            │
│ (event_engine)   │    序列化 world_state   │ (Panda3D / Godot)  │
│ 纯逻辑计算        │ ◀───────────────────── │ GPU 渲染           │
│                  │    用户输入/控制命令     │ 原生桌面窗口        │
└──────────────────┘                        └─────────────────────┘

渲染端未来可以换成 任何引擎（Panda3D、Godot、甚至 C++ 自定义），只要它能读 ZeroMQ 消息。
```

### 🎬 DEMO (Current)
🔷 Simulation in 3D:
![demo_3d](./demo/demo_3d.png)
🔶 Simulation in 2D (achieved by matplotlib):
![demo_mpl](./demo/demo_mpl.png)

## 📄 License

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This project is licensed under the **Apache License 2.0**.

You are free to use, modify, and distribute this code for academic and commercial purposes.
However, you must include the original copyright notice, state any significant changes made
to the files, and include a copy of the license. This license also provides an express grant
of patent rights from contributors.

For more details, please refer to the [LICENSE](./LICENSE.txt) file in this repository.