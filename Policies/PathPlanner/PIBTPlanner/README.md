# PIBTPlanner — 优先级继承回溯路径规划器

基于 PIBT（Priority Inheritance with Backtracking）算法的多智能体路径规划器。
与 A\*/Prioritized 等一次性计算完整路径的规划器不同，PIBT 每个 tick 仅决策一步，通过引擎每 tick 重复调用 `plan()` 实现持续导航。

> **论文**：Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2019).
> *Priority Inheritance with Backtracking for Iterative Multi-agent Path Finding.* IJCAI-19.

---

## 1. 算法原理

### 1.1 核心流程

每个 tick 的首次 `plan()` 调用触发一次 **批量 PIBT**（Bulk PIBT）：

```
1. 按动态优先级降序排列所有活跃 agent
2. 对每个未决策的 agent 执行 PIBT 递归：
   a. 生成候选位置 = 邻居格 + 原地不动
   b. 按 BFS 距离排序（贪心：优先靠近目标）
   c. 对每个候选格：
      - 若已被占用 → 跳过
      - 若有未决策 agent 占据 → 递归推开（优先级继承）
      - 推开失败 → 回溯，尝试下一候选
   d. 所有候选耗尽 → 原地等待
3. 根据决策结果更新优先级（老化机制）
```

### 1.2 关键机制

| 机制 | 说明 |
|------|------|
| **优先级继承** | 高优先级 agent A 要移动到 agent B 所在格时，B 继承 A 的调度权，递归为 B 寻找可行位置 |
| **回溯** | 若被推开的 agent 无法找到合法位置，撤销本次决策，尝试下一候选 |
| **优先级老化** | 每 tick 未取得进展的 agent 优先级 +1；取得进展则重置为 0；已到达目标的 agent 固定为 -1 |
| **BFS 距离启发** | 使用从目标反向 BFS 的真实最短路径距离（非 Manhattan），对有障碍物的地图至关重要 |
| **Pod 感知 BFS** | 搬运 pod 的 agent 使用独立的 BFS 缓存，将非搬运 pod 位置视为不可通行，引导 agent 走过道 |
| **随机打破平局** | 相同 BFS 距离的候选格随机排序，避免周期性振荡死锁 |
| **到达感知优先级** | 记录 agent 到达的具体目标；任务切换后自动清除到达状态，恢复正常优先级 |

---

## 2. 与其他规划器对比

| 特性 | AStarPathPlanner | PrioritizedPathPlanner | **PIBTPlanner** |
|------|-----------------|----------------------|-----------------|
| 规划粒度 | 一次性完整路径 | 一次性完整路径 | 每 tick 一步 |
| 冲突处理 | 无（独立规划） | 时空预留表 | 优先级继承 + 回溯 |
| 冲突保证 | 可能冲突 | 无冲突 | **无冲突** |
| 计算复杂度 | O(V log V) per agent | O(NV log V) | O(N) per tick |
| 适用场景 | 低密度 | 中密度 | 高密度 / 实时 |
| RMFS 仿真 | 基准吞吐量 | 最优吞吐量 | 83-116% of A\* |

---

## 3. 使用方式

### 3.1 配置切换

在 `default_config.json` 中：

```json
"policies": {
    "path_planner": "PIBTPlanner"
}
```

或在 `benchmark_config.json` 中用于 MAPF benchmark：

```json
"policies": {
    "path_planner": "PIBTPlanner"
}
```

### 3.2 命令行

```bash
# 主仿真模式
python main.py --config Config/default_config.json

# MAPF benchmark 模式
python main.py --benchmark --config Config/benchmark_config.json
```

### 3.3 代码调用

```python
from Policies.PathPlanner.PIBTPlanner import PIBTPlanner

planner = PIBTPlanner()

# 主仿真：由 SimulationEngine 每 tick 自动调用
path = planner.plan(agent, goal, world_state)  # 返回 [next_pos]

# Benchmark：提前设置所有 agent 的目标
planner.set_goals({0: (5, 10), 1: (3, 7), ...})
```

---

## 4. 实现细节

### 4.1 文件结构

```
PIBTPlanner/
├── __init__.py              # 导出 PIBTPlanner
├── pibt_path_planner.py     # 核心实现（~350 行）
└── README.md                # 本文档
```

### 4.2 关键数据结构

| 字段 | 类型 | 生命周期 | 说明 |
|------|------|---------|------|
| `_goals` | `Dict[int, (int,int)]` | 跨 tick | agent → 目标位置缓存 |
| `_priorities` | `Dict[int, int]` | 跨 tick | agent → 动态优先级（-1, 0, 1, 2, ...） |
| `_arrived` | `Dict[int, (int,int)]` | 跨 tick | agent → 到达的目标（任务切换时自动清除） |
| `_dist_cache` | `Dict[(int,int), Dict]` | 跨 tick | 目标 → BFS 距离表（不考虑 pod） |
| `_carry_dist_cache` | `Dict[(int,int), Dict]` | 每 tick 重建 | 目标 → BFS 距离表（考虑 pod） |
| `_decided` | `Dict[int, (int,int)]` | 每 tick | agent → 本 tick 决策位置 |
| `_next_occupied` | `Dict[(int,int), int]` | 每 tick | 位置 → 占据该位置的 agent |
| `_undecided` | `Set[int]` | 每 tick | 尚未决策的 agent 集合 |

### 4.3 BFS 距离缓存策略

- **非搬运 agent**：使用 `_dist_cache`，仅考虑地图障碍物，跨 tick 持久化（地图不变）
- **搬运 agent**：使用 `_carry_dist_cache`，额外将非搬运 pod 位置视为障碍，每 tick 重建（因 pod 位置随搬运操作变化）

### 4.4 与 SimulationEngine 的交互

```
Engine._tick():
  ├── _plan_and_activate()
  │     └── for agent in agents:
  │           if not agent.has_path:
  │               path = planner.plan(agent, goal, world)  ← 第一个 agent 触发 bulk PIBT
  │               agent.assign_path(path)                  ← [next_pos] 或 [current_pos]
  ├── _move_agents()
  │     └── agent.advance()                                ← 消耗 1 步路径，has_path → False
  └── 下一 tick → 重新调用 plan()
```

---

## 5. Benchmark 性能

### 5.1 MAPF Benchmark（纯路径规划，无 pod/订单）

| 地图 | Agent 数 | 完成率 | 冲突 | Makespan |
|------|---------|--------|------|----------|
| empty-32-32 | 100 | 99.0% | 0 | 500 |
| empty-32-32 | 200 | 98.5% | 0 | 500 |
| random-32-32-10 | 50 | 100% | 0 | 49 |
| random-64-64-10 | 100 | 100% | 0 | 108 |
| warehouse-10-20-10-2-1 | 50 | 82.0% | 0 | 500 |
| warehouse-10-20-10-2-1 | 100 | 54.0% | 0 | 500 |
| maze-32-32-2 | 20 | 100% | 0 | 138 |

### 5.2 RMFS 主仿真（4 agent，20x20 地图）

| Tick 数 | A\* 订单 | PIBT 订单 | PIBT/A\* | PIBT move% |
|---------|---------|----------|----------|------------|
| 200 | 11 | 9 | 82% | 94.4% |
| 500 | 25 | 29 | 116% | 94.3% |
| 1000 | 59 | 49 | 83% | 93.8% |

> warehouse 地图完成率较低是 vanilla PIBT 在窄走廊环境的已知局限，更高级的算法（LaCAM/winPIBT）可改善此问题。

---

## 6. 已知局限

1. **窄走廊死锁**：单格宽走廊中 agent 面对面时，PIBT 的单步贪心决策难以协调让路，需要更高级的旋转操作（rotation）来解决
2. **非最优路径**：贪心局部决策不保证全局最优路径长度，makespan 通常高于 A\*/CBS
3. **高密度 pod 区域**：搬运 agent 在 pod 密集区域可行走空间受限，可能需要多步绕行

---

## 7. 参考文献

- Okumura, K., Machida, M., Défago, X., & Tamura, Y. (2019). *Priority Inheritance with Backtracking for Iterative Multi-agent Path Finding.* IJCAI-19.
- Okumura, K. (2023). *Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding.* IJCAI-23.
