# default_config.json 配置参数说明

本文档对 `Config/default_config.json` 中的所有配置段和参数进行解析说明。

---

## 1. `map` — 仓库地图配置

定义仓库网格的尺寸、障碍物、工作站和货架存储区。

| 参数 | 类型 | 说明 |
|------|------|------|
| `rows` | `int` | 网格行数 |
| `cols` | `int` | 网格列数 |
| `obstacles` | `list[[row, col]]` | 障碍物坐标列表（不可通行格子） |

### 1.1 `stations` — 工作站列表

每个工作站是一个对象：

| 参数 | 类型 | 说明 |
|------|------|------|
| `id` | `int` | 工作站唯一 ID |
| `row` | `int` | 工作站所在行 |
| `col` | `int` | 工作站所在列 |

> 当前配置有 4 个工作站，分别位于地图四角附近。

### 1.2 货架存储片区 — 两种配置方式

Pod 存储区支持两种互斥的配置方式，使用其中一种即可：

#### 方式一：`pod_zones` — 显式列出每个片区（适合少量 zone）

每个片区定义一个矩形区域，区域内每个格子放置一个 Pod。

| 参数 | 类型 | 说明 |
|------|------|------|
| `origin_row` | `int` | 片区左上角行号 |
| `origin_col` | `int` | 片区左上角列号 |
| `num_rows` | `int` | 片区行数 |
| `num_cols` | `int` | 片区列数（建议 ≤ 2，否则内部 Pod 无法取出） |

```json
"pod_zones": [
    {"origin_row": 3, "origin_col": 2, "num_rows": 6, "num_cols": 2},
    {"origin_row": 3, "origin_col": 8, "num_rows": 6, "num_cols": 2}
]
```

#### 方式二：`pod_layout` — 紧凑参数自动生成（适合大规模场景）

用 5 个参数自动铺满整个地图，无需手动列出每个 zone。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_rows` | `int` | `10` | 每个 zone 的行数 |
| `num_cols` | `int` | `2` | 每个 zone 的列数（建议 ≤ 2） |
| `row_step` | `int` | `12` | 行方向周期 = `num_rows` + 行过道宽度 |
| `col_step` | `int` | `6` | 列方向周期 = `num_cols` + 列过道宽度 |
| `margin` | `int` | `3` | Pod 区域距地图边缘的格数 |
| `max_pods` | `int` | `0` | Pod 总数上限（0 = 不限制，自动铺满） |

```json
"pod_layout": {
    "num_rows": 10,
    "num_cols": 2,
    "row_step": 12,
    "col_step": 6,
    "margin": 4
}
```

> 上例中列过道宽度 = `col_step - num_cols` = 6 - 2 = **4 格**，行过道宽度 = `row_step - num_rows` = 12 - 10 = **2 格**。

布局示意（列方向）：
```
col:  margin  [P][P]  4格过道  [P][P]  4格过道  [P][P] ...
        ↑     ←2列→  ←col_step=6→
```

> **注意**：`pod_layout` 和 `pod_zones` 二选一。同时存在时优先使用 `pod_layout`。

---

## 2. `robots` — 机器人配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_robots` | `int` | — | 机器人数量 |
| `starts` | `list[[row, col]]` | — | 每个机器人的初始位置，长度应 ≥ `num_robots` |
| `speed` | `int` | `1` | 机器人移动速度（每 tick 移动的格子数） |

---

## 3. `pods` — Pod 初始化配置

控制 Pod 的类型划分、SKU 分配和库存初始化，由 `PodInitializer` 策略使用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pod_types` | `list[str]` | `["A", "B", "C"]` | 可用的 Pod 类型列表。`DefaultPodInitializer` 按轮询将 `pod_zones` 分配给这些类型 |
| `skus_per_pod` | `int` | `3` | 每个 Pod 持有的 SKU 种类数 |
| `sku_pool_size_per_type` | `int` | `10` | 每种 Pod 类型对应的 SKU 种类数（用于生成 SKU 标识符池） |
| `items_per_sku` | `int` | `20` | 每个 Pod 初始化时每种 SKU 的物品数量 |

> **约束**：同类型 Pod 位于同一 pod_zone 内；同类型 SKU 只会出现在对应类型的 Pod 上。
>
> **空配置**：当 `pods` 为空字典 `{}` 时，所有 Pod 视为 `"default"` 类型，`skus_per_pod=1`，`sku_pool_size=1`。

---

## 4. `simulation` — 仿真运行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order_interval` | `int` | `5` | 每隔 N 个 tick 生成一个新订单 |
| `max_items_per_order` | `int` | `2` | 每个订单最多包含的 SKU 种类数 |
| `pickup_duration` | `int` | `2` | 机器人到达 Pod 位置后拣货等待 tick 数 |
| `dropoff_duration` | `int` | `2` | 机器人归还 Pod 时放下等待 tick 数 |
| `station_process_duration` | `int` | `5` | 在工作站处理（配送）等待 tick 数。Pod 到达工作站时会扣减对应 SKU 库存 |
| `tick_delay` | `float` | `0.0` | 每个 tick 之间的间隔时间（秒），用于控制仿真速度 |
| `p3d_view_mode` | `str` | `"2d"` | Panda3D 可视化摄像机模式：`"2d"`（正交）或 `"3d"`（透视） |
| `p3d_use_gpu` | `bool` | `false` | 是否启用 Panda3D 的 GPU 批处理/实例化 |
| `night_mode` | `bool` | `true` | `true` = 暗色主题，`false` = 亮色主题 |
| `log_level` | `str` | `"INFO"` | 日志级别（`DEBUG` / `INFO` / `WARNING` / `ERROR`） |
| `log_file` | `str\|null` | `null` | 日志输出文件路径，`null` 表示仅输出到控制台 |
| `fixed_order_size` | `bool` | `false` | `true` = 每个订单固定包含 `max_items_per_order` 种 SKU；`false` = 随机 1 ~ `max_items_per_order` |
| `task_execution_mode` | `str` | `"parallel"` | 任务执行模式：`"parallel"` = 多机器人并行处理同一订单；`"serial"` = 单机器人串行处理 |
| `max_items_per_sku` | `int` | `5` | 订单生成时每种 SKU 需求的物品数量上限，实际数量取 `randint(1, max_items_per_sku)` |

---

## 5. `policies` — 策略选择配置

指定各模块使用的算法实现。每个字段可以是：
- **字符串**：仅指定算法名称（无额外参数）
- **对象**：`{"name": "算法名", "params": {...}}` 指定算法名称和额外参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order_generator` | `str\|obj` | `"RandomOrderGenerator"` | 订单生成策略（生成 SKU 需求） |
| `task_assigner` | `str\|obj` | `"GreedyTaskAssigner"` | 任务分配策略 |
| `path_planner` | `str\|obj` | `"AStarPathPlanner"` | 路径规划策略 |
| `pod_return_planner` | `str\|obj` | `"HomeReturnPlanner"` | Pod 归还位置规划策略 |
| `pod_initializer` | `str\|obj` | `"DefaultPodInitializer"` | Pod 初始化策略 |
| `pod_retriever` | `str\|obj` | `"DefaultPodRetriever"` | Pod 检索策略（将订单 SKU 需求映射为 Pod 列表） |

### 可用算法

| 策略类别 | 可用实现 | 额外参数 |
|---------|---------|---------|
| `order_generator` | `RandomOrderGenerator`, `ZipfOrderGenerator` | `ZipfOrderGenerator`: `zipf_param`（Zipf 指数，越大越偏向热门 SKU） |
| `task_assigner` | `GreedyTaskAssigner` | — |
| `path_planner` | `AStarPathPlanner`, `PrioritizedPathPlanner` | `PrioritizedPathPlanner`: `max_horizon`（搜索步数上限）, `goal_reserve`（目标保留 tick 数） |
| `pod_return_planner` | `HomeReturnPlanner` | — |
| `pod_initializer` | `DefaultPodInitializer` | — |
| `pod_retriever` | `DefaultPodRetriever` | — |

---

## 6. `replay` — 回放模式配置

控制多轨迹回放时的显示模式和布局方式。仅在使用 `--replay` 参数加载多个轨迹文件时生效。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | `str` | `"window"` | 回放模式：`"window"` = Panda3D 网格窗口模式；`"list"` = 列表卡片模式 |
| `layout` | `str` | `"auto"` | 网格布局：`"auto"` = 自动计算；或显式指定如 `"2x2"`、`"1x3"` |

```json
"replay": {
    "mode": "window",
    "layout": "auto"
}
```

### 模式说明

- **`"window"`（窗口模式）**：所有轨迹在同一窗口的 RxC 网格中独立渲染，支持鼠标滚轮滚动（当文件数超过布局容量时）。点击某个子窗口可进入聚焦查看模式。
- **`"list"`（列表模式）**：以卡片列表形式展示所有轨迹的实时指标（当前帧、机器人状态分布、搬运 Pod 数等），点击卡片进入 Panda3D 聚焦视图。

### 布局优先级

当 `mode = "window"` 时，布局的确定优先级为：

1. CLI 参数 `--replay-layout`（最高优先级）
2. 配置文件中 `replay.layout` 的值
3. 自动计算（`"auto"`）

> **注意**：当文件数量超过布局格子数（如 6 个文件使用 `2x2` 布局），窗口模式会自动启用滚动功能，无需手动调整布局。
> 单文件回放不受此配置影响，始终使用 `ReplayUI`。

---

## 完整结构总览

```
default_config.json
├── map
│   ├── rows, cols              # 网格尺寸
│   ├── obstacles               # 障碍物
│   ├── stations[]              # 工作站 (id, row, col)
│   ├── pod_zones[]             # [方式一] 显式 Pod 片区列表
│   └── pod_layout{}            # [方式二] 紧凑参数 (num_rows, num_cols, row_step, col_step, margin)
├── robots
│   ├── num_robots              # 机器人数量
│   ├── starts[]                # 初始位置
│   └── speed                   # 移动速度
├── pods
│   ├── pod_types               # Pod 类型列表
│   ├── skus_per_pod            # 每 Pod SKU 种类数
│   ├── sku_pool_size_per_type  # 每类型 SKU 池大小
│   └── items_per_sku           # 每种 SKU 初始物品数量
├── simulation
│   ├── order_interval          # 订单生成间隔
│   ├── max_items_per_order     # 订单最大 SKU 种类数
│   ├── pickup/dropoff/station_process_duration  # 操作等待
│   ├── tick_delay              # Tick 间隔
│   ├── p3d_view_mode, p3d_use_gpu, night_mode   # 可视化
│   ├── log_level, log_file     # 日志
│   ├── fixed_order_size        # 固定订单大小开关
│   ├── task_execution_mode     # 串/并行模式
│   └── max_items_per_sku       # 每种 SKU 需求数量上限
├── policies
│   ├── order_generator         # 订单生成算法（生成 SKU 需求）
│   ├── task_assigner           # 任务分配算法
│   ├── path_planner            # 路径规划算法
│   ├── pod_return_planner      # Pod 归还算法
│   ├── pod_initializer         # Pod 初始化算法
│   └── pod_retriever           # Pod 检索算法（SKU 需求 → Pod 列表）
└── replay
    ├── mode                    # 回放模式（window / list）
    └── layout                  # 网格布局（auto / RxC）
```
