# Prompt: 实现 Pod 初始化策略（PodInitializer）

## 背景

你正在为一个仓储仿真系统（MAS_RMFS）添加 **Pod 初始化策略**。该系统使用 Python 编写，核心架构如下：

- **`Policies/`** 目录下存放所有可插拔策略，每个策略有独立子文件夹，包含一个 `base_*.py` 抽象基类和若干具体实现。策略通过 [Policies/policy_registry.py](file:///f:/MAS_RMFS/Policies/policy_registry.py) 注册/查找。
- **[WorldState/pod_state.py](file:///f:/MAS_RMFS/WorldState/pod_state.py)** 定义了 [Pod](file:///f:/MAS_RMFS/WorldState/pod_state.py#10-54) 类（属性：`pod_id`, `home_position`, `current_position`, `is_carried`）和 [PodState](file:///f:/MAS_RMFS/WorldState/pod_state.py#56-98) 容器。
- **[WorldState/map_state.py](file:///f:/MAS_RMFS/WorldState/map_state.py)** 中 [MapState](file:///f:/MAS_RMFS/WorldState/map_state.py#21-95) 根据 [default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) 的 `pod_zones` 配置展开所有 pod 的 home position，每个 `pod_zone` 定义一个矩形片区。
- **[WorldState/world.py](file:///f:/MAS_RMFS/WorldState/world.py)** 中 `WorldState.__init__()` 目前遍历 `map_state.pod_home_positions` 逐个创建 [Pod](file:///f:/MAS_RMFS/WorldState/pod_state.py#10-54) 对象。
- **[Config/default_config.json](file:///f:/MAS_RMFS/Config/default_config.json)** 包含 `map.pod_zones`（8 个片区）、`robots`、`simulation`、[policies](file:///f:/MAS_RMFS/Policies/policy_registry.py#87-99) 等配置。
- 当前系统 **没有 SKU 概念**，Pod 仅是一个空壳（只有位置和 ID）。

## 需求

### 1. 给 Pod 增加"类型"（pod_type）属性

- 在 [Pod](file:///f:/MAS_RMFS/WorldState/pod_state.py#10-54) 类中新增 `pod_type: str` 属性（如 `"A"`, `"B"`, `"C"`）。
- `pod_type` 在初始化时赋值，创建后不变。

### 2. 同类型 Pod 必须初始化在同一个片区（pod_zone）

- 一个 `pod_zone` 内所有 Pod 的 `pod_type` 相同。
- 不同 `pod_zone` 可以有相同的 `pod_type`（即多个片区可存放同类型 Pod）。
- 哪些 `pod_zone` 分配哪种 `pod_type` 由初始化策略决定。

### 3. 每个 Pod 持有 n 个 SKU

- 在 [Pod](file:///f:/MAS_RMFS/WorldState/pod_state.py#10-54) 类中新增 `skus: List[str]` 属性（存放 SKU 标识符列表）。
- `n`（每个 Pod 持有的 SKU 数量）从 [default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) 中读取（新增配置参数）。
- **同类型 SKU 只能出现在同类型的 Pod 上**。即 `pod_type="A"` 的所有 Pod 只会持有属于类型 A 的 SKU，不会出现类型 B 的 SKU。

### 4. 创建 PodInitializer 策略接口

在 `Policies/` 下创建 `PodInitializer/` 子文件夹，遵循现有策略的目录结构：

```
Policies/
└── PodInitializer/
    ├── __init__.py                    # 导入 + 注册到 policy_registry
    ├── base_pod_initializer.py        # 抽象基类 (interface)
    └── DefaultPodInitializer/
        └── __init__.py                # 具体默认实现
```

#### 抽象基类 `BasePodInitializer`

```python
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

- 从 config 读取 `pod_type` 列表和每个 Pod 的 SKU 数量 `n`。
- 将 `pod_zones` 按轮询或均匀方式分配给各 `pod_type`。
- 为每种 `pod_type` 生成专属的 SKU 池，每个 Pod 从对应类型的 SKU 池中分配 `n` 个 SKU。

### 5. 配置扩展

在 [default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) 中新增以下参数：

```json
{
    "pods": {
        "pod_types": ["A", "B", "C"],
        "skus_per_pod": 3,
        "sku_pool_size_per_type": 10
    },
    "policies": {
        "pod_initializer": "DefaultPodInitializer",
        ...
    }
}
```

- `pod_types`: 可用的 Pod 类型列表。
- `skus_per_pod`: 每个 Pod 持有的 SKU 数量（即 `n`）。
- `sku_pool_size_per_type`: 每种 pod_type 对应的 SKU 种类数（用于生成 SKU 标识符）。

### 6. 集成到 WorldState 初始化流程

修改 `WorldState.__init__()`，将当前写死的 Pod 创建逻辑替换为调用 `PodInitializer` 策略：

```python
# 旧逻辑（移除）：
# for idx, pos in enumerate(self.map_state.pod_home_positions):
#     pod = Pod(pod_id=idx, home_position=pos)
#     self.pod_state.add_pod(pod)

# 新逻辑：
pod_initializer = get_policy("pod_initializer", config.policies.pod_initializer)()
pod_initializer.initialize_pods(self)
```

## 编码规范

1. **语言风格**：代码中的注释和 docstring 使用中英混合（与现有代码风格一致，参考 [policy_registry.py](file:///f:/MAS_RMFS/Policies/policy_registry.py) 和 [world.py](file:///f:/MAS_RMFS/WorldState/world.py)）。
2. **类型注解**：所有函数签名和类属性使用 Python type hints。
3. **注册机制**：在 `PodInitializer/__init__.py` 中通过 [register("pod_initializer", "DefaultPodInitializer", DefaultPodInitializer)](file:///f:/MAS_RMFS/Policies/policy_registry.py#31-45) 注册。
4. **配置加载**：需要同步修改 [Config/config_loader.py](file:///f:/MAS_RMFS/Config/config_loader.py) 以支持解析新的 [pods](file:///f:/MAS_RMFS/WorldState/pod_state.py#91-94) 配置段（参照现有 `MapConfig`、`RobotsConfig` 的 dataclass 模式）。
5. **不要破坏现有测试和功能**——OrderGenerator 等现有策略不应受影响。

## 涉及文件

| 操作 | 文件路径 |
|------|---------|
| MODIFY | [WorldState/pod_state.py](file:///f:/MAS_RMFS/WorldState/pod_state.py) — 给 [Pod](file:///f:/MAS_RMFS/WorldState/pod_state.py#10-54) 增加 `pod_type` 和 `skus` 属性 |
| MODIFY | [WorldState/world.py](file:///f:/MAS_RMFS/WorldState/world.py) — 替换 Pod 初始化逻辑为调用策略 |
| MODIFY | [Config/default_config.json](file:///f:/MAS_RMFS/Config/default_config.json) — 增加 [pods](file:///f:/MAS_RMFS/WorldState/pod_state.py#91-94) 配置段和 `policies.pod_initializer` |
| MODIFY | [Config/config_loader.py](file:///f:/MAS_RMFS/Config/config_loader.py) — 增加 `PodsConfig` dataclass |
| NEW | `Policies/PodInitializer/__init__.py` |
| NEW | `Policies/PodInitializer/base_pod_initializer.py` |
| NEW | `Policies/PodInitializer/DefaultPodInitializer/__init__.py` |
