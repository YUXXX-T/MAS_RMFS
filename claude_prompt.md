# Prompt: MAS_RMFS 仿真系统 — 四项功能增强

## 背景

你正在为一个仓储仿真系统（MAS_RMFS）实现四项功能增强。该系统使用 Python 编写，核心架构如下：

- **`Policies/`** 下存放所有可插拔策略，每个策略有独立子文件夹，包含一个 `base_*.py` 抽象基类和若干具体实现。策略通过 [policy_registry.py](file:///f:/MAS_RMFS/Policies/policy_registry.py) 注册/查找（`register(category, name, cls)` / `get_policy(category, name)`）。
- **[WorldState/pod_state.py](file:///f:/MAS_RMFS/WorldState/pod_state.py)** 定义了 [Pod](file:///f:/MAS_RMFS/WorldState/pod_state.py#L10-L53) 类（`pod_id`, `home_position`, `current_position`, `is_carried`）和 [PodState](file:///f:/MAS_RMFS/WorldState/pod_state.py#L56-L97) 容器。**当前 Pod 没有 `pod_type` 和 `skus` 属性。**
- **[WorldState/world.py](file:///f:/MAS_RMFS/WorldState/world.py)** 中 [WorldState.__init__()](file:///f:/MAS_RMFS/WorldState/world.py#L39-L60) 遍历 `map_state.pod_home_positions` 逐个创建 Pod 对象（硬编码逻辑）。
- **[WorldState/map_state.py](file:///f:/MAS_RMFS/WorldState/map_state.py)** 中 `MapState` 根据 `default_config.json` 的 `pod_zones`（8 个矩形片区）展开所有 pod 的 home position。
- **[WorldState/order_state.py](file:///f:/MAS_RMFS/WorldState/order_state.py)** 定义了 `Order` 类（属性：`order_id`, `pod_ids: List[int]`, `station_id`, `status`）。一个订单可能需要 1 ~ `max_items_per_order` 个 pod。
- **[WorldState/task_state.py](file:///f:/MAS_RMFS/WorldState/task_state.py)** 定义了 `Task` 类（`PICK` → `DELIVER` → `RETURN` 三种类型），每个 task 关联一个 `agent_id` 和 `pod_id`。
- **[WorldState/agent_state.py](file:///f:/MAS_RMFS/WorldState/agent_state.py)** 定义了 `AgentState`（状态机：`IDLE → MOVING_TO_POD → CARRYING → DELIVERING → RETURNING`）和 `AgentStatus` 枚举。
- **[Config/config_loader.py](file:///f:/MAS_RMFS/Config/config_loader.py)** 使用 dataclass 解析 JSON 配置（`MapConfig`, `RobotConfig`, `SimulationParams`, `PolicyConfig`），入口函数 `load_config(path)`。
- **[Config/default_config.json](file:///f:/MAS_RMFS/Config/default_config.json)** 包含 `map`（含 `pod_zones`）、`robots`、`simulation`（含 `max_items_per_order`）、`policies` 四个顶层段。
- **[main.py](file:///f:/MAS_RMFS/main.py)** 从配置实例化各策略并注入 `SimulationEngine`。
- **[Engine/simulation_engine.py](file:///f:/MAS_RMFS/Engine/simulation_engine.py)** 执行 tick 循环：生成订单 → 分配任务 → 规划路径 → 移动 → 检测冲突 → 处理 pickup/delivery/return → 检查订单完成。

### 当前 GreedyTaskAssigner 行为

[GreedyTaskAssigner](file:///f:/MAS_RMFS/Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py) 的 `assign()` 方法：

- 遍历每个 pending order 的每个 `pod_id`
- 为**每个 pod** 独立找最近的空闲机器人
- 为该机器人创建 `PICK → DELIVER → RETURN` 三个 task
- 这意味着**一个包含多个 pod 的订单会被多个机器人并行完成**（每个 pod 由不同的机器人负责）

### 当前 OrderGenerator 行为

[ZipfOrderGenerator](file:///f:/MAS_RMFS/Policies/OrderGenerator/ZipfOrderGenerator/zipf_order_generator.py) 的 `generate()` 方法：

- 按 `order_interval` 间隔生成订单
- 每个订单的 pod 数量为 `random.randint(1, max_items_per_order)`（随机 1 到上限）
- 使用 Zipf 分布抽取 pod

---

## 需求一：Pod 初始化策略（PodInitializer）

### 1.1 给 Pod 增加 `pod_type` 和 `skus` 属性

在 [Pod](file:///f:/MAS_RMFS/WorldState/pod_state.py#L10-L53) 类中新增：

- `pod_type: str` — Pod 类型标识（如 `"A"`, `"B"`, `"C"`），初始化时赋值，之后不变
- `skus: List[str]` — 该 Pod 持有的 SKU 标识符列表

修改 `Pod.__init__()` 签名来接受这两个新参数（可以有默认值以保证向后兼容）。

### 1.2 同类型 Pod 必须初始化在同一个 pod_zone

- 一个 `pod_zone` 内所有 Pod 的 `pod_type` 相同
- 不同 `pod_zone` 可以有相同的 `pod_type`（多个片区可存放同类型 Pod）
- 具体分配策略由 `PodInitializer` 实现决定

### 1.3 同类型 SKU 只能在同类型 Pod 上

`pod_type="A"` 的所有 Pod 只持有属于类型 A 的 SKU 池中的 SKU，不会出现类型 B 的 SKU。每个 Pod 持有 `n` 个 SKU，`n` 从配置读取。

### 1.4 创建 PodInitializer 策略

在 `Policies/` 下创建 `PodInitializer/` 子文件夹，遵循现有策略目录结构：

```
Policies/
└── PodInitializer/
    ├── __init__.py                    # 导入 + 注册到 policy_registry
    ├── base_pod_initializer.py        # 抽象基类
    └── DefaultPodInitializer/
        └── __init__.py                # 具体默认实现
```

#### 抽象基类 `BasePodInitializer`

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState

class BasePodInitializer(ABC):
    @abstractmethod
    def initialize_pods(self, world_state: "WorldState") -> None:
        """
        根据 map_state 中的 pod_zones 和 config 中的参数，
        初始化所有 Pod 的 pod_type 和 skus 属性，
        并将 Pod 对象注册到 world_state.pod_state 中。
        
        该方法在仿真开始前调用一次。
        """
        ...
```

#### 默认实现 `DefaultPodInitializer`

- 从 config 读取 `pod_types` 列表和 `skus_per_pod`
- 将 `pod_zones` 按轮询或均匀方式分配给各 `pod_type`
- 为每种 `pod_type` 生成专属的 SKU 池（大小由 `sku_pool_size_per_type` 控制），每个 Pod 从对应类型的 SKU 池中分配 `n` 个 SKU

### 1.5 配置扩展

在 [default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) 中新增：

```json
{
    "pods": {
        "pod_types": ["A", "B", "C"],
        "skus_per_pod": 3,
        "sku_pool_size_per_type": 10
    },
    "policies": {
        "pod_initializer": "DefaultPodInitializer",
        ...已有策略配置保留...
    }
}
```

同步修改 [config_loader.py](file:///f:/MAS_RMFS/Config/config_loader.py)，新增 `PodsConfig` dataclass，并在 `SimulationConfig` 和 `load_config()` 中解析 `pods` 段。同时在 `PolicyConfig` 中增加 `pod_initializer` 字段。

### 1.6 集成到 WorldState 初始化流程

修改 [WorldState.__init__()](file:///f:/MAS_RMFS/WorldState/world.py#L39-L60)，将现在硬编码的 Pod 创建逻辑：

```python
# 旧逻辑（移除）：
for idx, pos in enumerate(self.map_state.pod_home_positions):
    pod = Pod(pod_id=idx, home_position=pos)
    self.pod_state.add_pod(pod)
```

替换为调用 PodInitializer 策略：

```python
# 新逻辑：
from Policies.policy_registry import get_policy
PodInitializerCls = get_policy("pod_initializer", config.policies.pod_initializer[0])
pod_initializer = PodInitializerCls()
pod_initializer.initialize_pods(self)
```

---

## 需求二：固定订单 pod 数量参数

### 当前行为

在 [ZipfOrderGenerator](file:///f:/MAS_RMFS/Policies/OrderGenerator/ZipfOrderGenerator/zipf_order_generator.py#L82-L85) 中：

```python
num_items = min(
    random.randint(1, self.max_items_per_order),
    len(available_pods),
)
```

每个订单的 pod 数量是 **1 ~ max_items_per_order** 之间的随机数。

### 需要做的

在 [default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) 的 `simulation` 段增加一个布尔参数：

```json
{
    "simulation": {
        ...已有参数...
        "fixed_order_size": false
    }
}
```

- `fixed_order_size: false`（默认）— 保持现有行为，`random.randint(1, max_items_per_order)`
- `fixed_order_size: true` — 每个订单**固定**包含 `max_items_per_order` 个 pod

#### 修改文件

1. **[Config/default_config.json](file:///f:/MAS_RMFS/Config/default_config.json)** — `simulation` 段增加 `"fixed_order_size": false`
2. **[Config/config_loader.py](file:///f:/MAS_RMFS/Config/config_loader.py)** — `SimulationParams` dataclass 增加 `fixed_order_size: bool = False`，`load_config()` 中解析该字段
3. **[ZipfOrderGenerator](file:///f:/MAS_RMFS/Policies/OrderGenerator/ZipfOrderGenerator/zipf_order_generator.py)** — 接受 `fixed_order_size` 参数，修改 `num_items` 计算逻辑：

```python
# 新逻辑：
if self.fixed_order_size:
    num_items = min(self.max_items_per_order, len(available_pods))
else:
    num_items = min(
        random.randint(1, self.max_items_per_order),
        len(available_pods),
    )
```

4. **[main.py](file:///f:/MAS_RMFS/main.py#L76-L80)** — 实例化 OrderGenerator 时传入 `fixed_order_size` 参数：

```python
order_generator = OrderGeneratorCls(
    order_interval=config.simulation.order_interval,
    max_items_per_order=config.simulation.max_items_per_order,
    fixed_order_size=config.simulation.fixed_order_size,  # 新增
    **og_params,
)
```

5. 同时检查 [RandomOrderGenerator](file:///f:/MAS_RMFS/Policies/OrderGenerator/RandomOrderGenerator/) 和 [BaseOrderGenerator](file:///f:/MAS_RMFS/Policies/OrderGenerator/base_order_generator.py) 是否也需要同步修改，保持接口一致。

---

## 需求三：串行任务分配模式

### 当前行为（并行模式）

[GreedyTaskAssigner](file:///f:/MAS_RMFS/Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py) 的 `assign()` 方法中，对于一个包含多个 pod 的订单，**每个 pod 分别分配给不同的空闲机器人**，多个机器人并行完成同一订单。

核心逻辑（[第 55-136 行](file:///f:/MAS_RMFS/Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py#L55-L136)）：

```python
for order in pending_orders:
    for pod_id in order.pod_ids:
        # 为每个 pod 找最近的空闲机器人
        idle_agents = world_state.get_idle_agents()
        agent = idle_agents[0]  # 最近的
        # 创建 PICK → DELIVER → RETURN 三个 task，都分配给 agent
```

### 需要做的

在 [default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) 的 `simulation` 段增加一个参数：

```json
{
    "simulation": {
        ...已有参数...
        "task_execution_mode": "parallel"
    }
}
```

- `"parallel"`（默认）— 保持现有行为（多个机器人并行处理同一订单的不同 pod）
- `"serial"` — **同一个订单的所有 pod 由同一台机器人串行完成**

#### 串行模式的具体行为

当 `task_execution_mode == "serial"` 时，`GreedyTaskAssigner.assign()` 应改为：

1. 遍历 pending order
2. 找到**一个**最近的空闲机器人
3. 将该订单的**所有** pod 的 `PICK → DELIVER → RETURN` task **全部**分配给这一台机器人
4. task 按顺序排列：`PICK_pod1 → DELIVER_pod1 → RETURN_pod1 → PICK_pod2 → DELIVER_pod2 → RETURN_pod2 → ...`
5. SimulationEngine 已有的 `_plan_and_activate()` 方法会按 `ASSIGNED` 状态依次激活 task，因此**不需要修改引擎逻辑**，只需确保 task 的创建顺序正确且都分配给同一个 agent

#### 修改文件

1. **[Config/default_config.json](file:///f:/MAS_RMFS/Config/default_config.json)** — `simulation` 段增加 `"task_execution_mode": "parallel"`
2. **[Config/config_loader.py](file:///f:/MAS_RMFS/Config/config_loader.py)** — `SimulationParams` 增加 `task_execution_mode: str = "parallel"`
3. **[GreedyTaskAssigner](file:///f:/MAS_RMFS/Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py)** — 在 `assign()` 中根据 `world_state.config.simulation.task_execution_mode` 选择并行/串行逻辑。**在同一个文件内实现**，不需要新建文件。建议实现方式：

```python
class GreedyTaskAssigner(BaseTaskAssigner):
    def assign(self, world_state) -> List[Task]:
        mode = world_state.config.simulation.task_execution_mode
        if mode == "serial":
            return self._assign_serial(world_state)
        else:
            return self._assign_parallel(world_state)
    
    def _assign_parallel(self, world_state) -> List[Task]:
        """现有的并行分配逻辑（每个 pod 分配给不同机器人）"""
        ...  # 将当前 assign() 的逻辑移到这里
        
    def _assign_serial(self, world_state) -> List[Task]:
        """串行分配逻辑（同一订单所有 pod 由同一机器人依次完成）"""
        new_tasks = []
        pending_orders = world_state.order_state.get_pending_orders()
        reserved_pods = {
            t.pod_id for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }
        
        for order in pending_orders:
            # 检查该订单的所有 pod 是否都可用
            order_pods = []
            for pod_id in order.pod_ids:
                pod = world_state.pod_state.get_pod(pod_id)
                if pod is None or pod.is_carried or pod_id in reserved_pods:
                    break
                order_pods.append(pod)
            else:
                # 所有 pod 都可用
                if not order_pods:
                    continue
                    
                # 找一个空闲机器人
                idle_agents = world_state.get_idle_agents()
                if not idle_agents:
                    continue
                
                # 按到第一个 pod 的距离排序
                idle_agents.sort(
                    key=lambda a: _manhattan_distance(a.position, order_pods[0].current_position)
                )
                agent = idle_agents[0]
                
                station_pos = world_state.map_state.station_positions.get(order.station_id)
                if station_pos is None:
                    continue
                
                # 为所有 pod 串行创建 PICK→DELIVER→RETURN
                for pod in order_pods:
                    pick_task = Task(...)
                    deliver_task = Task(...)
                    return_task = Task(...)
                    # 全部分配给同一个 agent
                    for t in [pick_task, deliver_task, return_task]:
                        t.agent_id = agent.agent_id
                        t.status = TaskStatus.ASSIGNED
                        world_state.task_state.add_task(t)
                        new_tasks.append(t)
                    reserved_pods.add(pod.pod_id)
                
                agent.status = AgentStatus.MOVING_TO_POD
                agent.assigned_task_id = new_tasks[0].task_id  # 第一个 task
                order.status = OrderStatus.IN_PROGRESS
        
        return new_tasks
```

> **注意**：串行模式中需要确认 [SimulationEngine._plan_and_activate()](file:///f:/MAS_RMFS/Engine/simulation_engine.py#L147-L188) 的 `get_next_task_for_agent()` 能按正确顺序逐个激活 task（当前 `TaskState.get_next_task_for_agent()` 返回第一个 `ASSIGNED` 状态的 task，这依赖于 dict 插入顺序，Python 3.7+ 保证 dict 保持插入顺序，所以只要 task 按正确顺序创建即可）。

---

## 编码规范

1. **语言风格**：代码中的注释和 docstring 使用中英混合（与现有代码风格一致，参考 [policy_registry.py](file:///f:/MAS_RMFS/Policies/policy_registry.py)、[world.py](file:///f:/MAS_RMFS/WorldState/world.py) 和 [greedy_task_assigner.py](file:///f:/MAS_RMFS/Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py)）。
2. **类型注解**：所有函数签名和类属性使用 Python type hints。
3. **注册机制**：新策略在 `__init__.py` 中通过 `register(category, name, cls)` 注册。
4. **配置加载**：新增配置段时同步修改 `config_loader.py`（参照 `MapConfig`、`RobotConfig` 的 dataclass 模式）。
5. **不要破坏现有功能**：OrderGenerator、PathPlanner 等现有策略不应受影响。现有测试和命令行参数需要保持兼容。

---

## 涉及文件总览

| 操作 | 文件路径 | 修改内容 |
|------|---------|---------|
| MODIFY | [WorldState/pod_state.py](file:///f:/MAS_RMFS/WorldState/pod_state.py) | 给 `Pod` 增加 `pod_type` 和 `skus` 属性 |
| MODIFY | [WorldState/world.py](file:///f:/MAS_RMFS/WorldState/world.py) | 替换 Pod 初始化逻辑为调用 PodInitializer 策略 |
| MODIFY | [Config/default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) | 增加 `pods` 配置段、`policies.pod_initializer`、`simulation.fixed_order_size`、`simulation.task_execution_mode` |
| MODIFY | [Config/config_loader.py](file:///f:/MAS_RMFS/Config/config_loader.py) | 增加 `PodsConfig` dataclass、`SimulationParams` 新字段、`PolicyConfig.pod_initializer` |
| MODIFY | [main.py](file:///f:/MAS_RMFS/main.py) | 传入 `fixed_order_size` 参数给 OrderGenerator |
| MODIFY | [ZipfOrderGenerator](file:///f:/MAS_RMFS/Policies/OrderGenerator/ZipfOrderGenerator/zipf_order_generator.py) | 支持 `fixed_order_size` 参数 |
| MODIFY | [GreedyTaskAssigner](file:///f:/MAS_RMFS/Policies/TaskAssigner/GreedyTaskAssigner/greedy_task_assigner.py) | 增加串行模式分配逻辑 |
| NEW | `Policies/PodInitializer/__init__.py` | 导入 + 注册 |
| NEW | `Policies/PodInitializer/base_pod_initializer.py` | 抽象基类 |
| NEW | `Policies/PodInitializer/DefaultPodInitializer/__init__.py` | 默认实现 |
