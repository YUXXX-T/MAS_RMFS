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

### 1.2 `pod_zones` — 货架存储片区列表

每个片区定义一个矩形区域，区域内每个格子放置一个 Pod。

| 参数 | 类型 | 说明 |
|------|------|------|
| `origin_row` | `int` | 片区左上角行号 |
| `origin_col` | `int` | 片区左上角列号 |
| `num_rows` | `int` | 片区行数 |
| `num_cols` | `int` | 片区列数 |

> 当前配置有 8 个片区，每个 6×2 = 12 个位置，共 96 个 Pod。

---

## 2. `robots` — 机器人配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_robots` | `int` | — | 机器人数量 |
| `starts` | `list[[row, col]]` | — | 每个机器人的初始位置，长度应 ≥ `num_robots` |
| `speed` | `int` | `1` | 机器人移动速度（每 tick 移动的格子数） |

---

## 3. `pods` — Pod 初始化配置

控制 Pod 的类型划分和 SKU 分配，由 `PodInitializer` 策略使用。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `pod_types` | `list[str]` | `["A", "B", "C"]` | 可用的 Pod 类型列表。`DefaultPodInitializer` 按轮询将 `pod_zones` 分配给这些类型 |
| `skus_per_pod` | `int` | `3` | 每个 Pod 持有的 SKU 数量 |
| `sku_pool_size_per_type` | `int` | `10` | 每种 Pod 类型对应的 SKU 种类数（用于生成 SKU 标识符池） |

> **约束**：同类型 Pod 位于同一 pod_zone 内；同类型 SKU 只会出现在对应类型的 Pod 上。

---

## 4. `simulation` — 仿真运行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order_interval` | `int` | `5` | 每隔 N 个 tick 生成一个新订单 |
| `max_items_per_order` | `int` | `2` | 每个订单最多包含的 Pod 数量 |
| `pickup_duration` | `int` | `2` | 机器人到达 Pod 位置后拣货等待 tick 数 |
| `dropoff_duration` | `int` | `2` | 机器人归还 Pod 时放下等待 tick 数 |
| `station_process_duration` | `int` | `5` | 在工作站处理（配送）等待 tick 数 |
| `tick_delay` | `float` | `0.0` | 每个 tick 之间的间隔时间（秒），用于控制仿真速度 |
| `p3d_view_mode` | `str` | `"2d"` | Panda3D 可视化摄像机模式：`"2d"`（正交）或 `"3d"`（透视） |
| `p3d_use_gpu` | `bool` | `false` | 是否启用 Panda3D 的 GPU 批处理/实例化 |
| `night_mode` | `bool` | `true` | `true` = 暗色主题，`false` = 亮色主题 |
| `log_level` | `str` | `"INFO"` | 日志级别（`DEBUG` / `INFO` / `WARNING` / `ERROR`） |
| `log_file` | `str\|null` | `null` | 日志输出文件路径，`null` 表示仅输出到控制台 |
| `fixed_order_size` | `bool` | `false` | `true` = 每个订单固定包含 `max_items_per_order` 个 Pod；`false` = 随机 1 ~ `max_items_per_order` |
| `task_execution_mode` | `str` | `"parallel"` | 任务执行模式：`"parallel"` = 多机器人并行处理同一订单；`"serial"` = 单机器人串行处理 |

---

## 5. `policies` — 策略选择配置

指定各模块使用的算法实现。每个字段可以是：
- **字符串**：仅指定算法名称（无额外参数）
- **对象**：`{"name": "算法名", "params": {...}}` 指定算法名称和额外参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order_generator` | `str\|obj` | `"RandomOrderGenerator"` | 订单生成策略 |
| `task_assigner` | `str\|obj` | `"GreedyTaskAssigner"` | 任务分配策略 |
| `path_planner` | `str\|obj` | `"AStarPathPlanner"` | 路径规划策略 |
| `pod_return_planner` | `str\|obj` | `"HomeReturnPlanner"` | Pod 归还位置规划策略 |
| `pod_initializer` | `str\|obj` | `"DefaultPodInitializer"` | Pod 初始化策略 |

### 可用算法

| 策略类别 | 可用实现 | 额外参数 |
|---------|---------|---------|
| `order_generator` | `RandomOrderGenerator`, `ZipfOrderGenerator` | `ZipfOrderGenerator`: `zipf_param`（Zipf 指数，越大越偏向热门 Pod） |
| `task_assigner` | `GreedyTaskAssigner` | — |
| `path_planner` | `AStarPathPlanner`, `PrioritizedPathPlanner` | `PrioritizedPathPlanner`: `max_horizon`（搜索步数上限）, `goal_reserve`（目标保留 tick 数） |
| `pod_return_planner` | `HomeReturnPlanner` | — |
| `pod_initializer` | `DefaultPodInitializer` | — |

---

## 完整结构总览

```
default_config.json
├── map
│   ├── rows, cols              # 网格尺寸
│   ├── obstacles               # 障碍物
│   ├── stations[]              # 工作站 (id, row, col)
│   └── pod_zones[]             # Pod 存储片区 (origin_row, origin_col, num_rows, num_cols)
├── robots
│   ├── num_robots              # 机器人数量
│   ├── starts[]                # 初始位置
│   └── speed                   # 移动速度
├── pods
│   ├── pod_types               # Pod 类型列表
│   ├── skus_per_pod            # 每 Pod SKU 数
│   └── sku_pool_size_per_type  # 每类型 SKU 池大小
├── simulation
│   ├── order_interval          # 订单生成间隔
│   ├── max_items_per_order     # 订单最大 Pod 数
│   ├── pickup/dropoff/station_process_duration  # 操作等待
│   ├── tick_delay              # Tick 间隔
│   ├── p3d_view_mode, p3d_use_gpu, night_mode   # 可视化
│   ├── log_level, log_file     # 日志
│   ├── fixed_order_size        # 固定订单大小开关
│   └── task_execution_mode     # 串/并行模式
└── policies
    ├── order_generator         # 订单生成算法
    ├── task_assigner           # 任务分配算法
    ├── path_planner            # 路径规划算法
    ├── pod_return_planner      # Pod 归还算法
    └── pod_initializer         # Pod 初始化算法
```
