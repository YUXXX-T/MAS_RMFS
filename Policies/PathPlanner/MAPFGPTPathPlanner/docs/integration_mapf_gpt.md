# MAPF-GPT 集成说明（精简版）

本文只说明 `MAS_RMFS` 主项目中的改动，不展开 `MAPF-GPT-main/` 内部实现。

## 1. 改了哪些文件

- `Config/default_config.json`：默认路径规划器改为 `MAPFGPTPathPlanner`。
- `Policies/PathPlanner/__init__.py`：注册 `MAPFGPTPathPlanner`。
- `Policies/PathPlanner/MAPFGPTPathPlanner/`：新增适配器实现。
- `README.md`：补充 MAPF-GPT 配置示例和说明。

## 2. 关键配置

```json
"path_planner": {
  "name": "MAPFGPTPathPlanner",
  "params": {
    "mapf_gpt_root": "MAPF-GPT-main",
    "model": "2M",
    "device": "cuda",
    "min_joint_agents": 2
  }
}
```

常用参数：

- `mapf_gpt_root`：MAPF-GPT 目录。
- `model`：`2M` / `6M` / `85M`。
- `device`：`cuda` / `cpu` / `mps`。
- `min_joint_agents`：联合推理最少活跃机器人数量。
- `avoid_agents`：A* 回退时是否避让其他机器人。
- `cost2go_radius`：cost2go 半径（默认 5）。
- `num_previous_actions`：历史动作长度（默认 5）。

## 3. 适配器做了什么

`MAPFGPTPathPlanner` 主要做四件事：

1. **每 tick 联合推理一次**：同 tick 内其余 `plan()` 调用直接读缓存。
2. **cost2go 缓存复用**：首次计算后跨 tick 复用，避免重复 C++ 预计算。
3. **维护动作历史**：在适配器层维护每个 agent 的历史动作。
4. **地图 padding**：为小地图补边，避免 `cost2go` 边界 `KeyError`。

推理链为：

`generate_input()` -> `encoder.encode()` -> `net.act()`

（有意绕过 `MAPFGPTInference.act()`，避免其内部状态副作用覆盖历史。）

## 4. 回退规则

- MAPF-GPT 初始化失败：整体回退 A*。
- 活跃 agent 数不足 `min_joint_agents`：当前 tick 使用 A*。
- MAPF-GPT 推理异常：当前 tick 使用 A*。
- MAPF-GPT 输出非法下一步（越界/障碍/货架占用）：该 agent 改用 A* 单步。

## 5. 没有改动的核心模块

- `main.py`
- `Engine/simulation_engine.py`
- `Policies/PathPlanner/base_path_planner.py`
- `Policies/PathPlanner/AStarPathPlanner/`
- `Policies/PathPlanner/PrioritizedPathPlanner/`

即：本次接入是通过新增路径规划器实现的，未改动仿真主循环。

## 6. 运行与确认

运行：

```bash
python main.py --config Config/default_config.json
```

确认日志：

```text
path_planner=MAPFGPTPathPlanner
MAPF-GPT backend ready: ...
```

若失败会看到：

```text
MAPF-GPT backend init failed; fallback to A*: ...
```

补充 -- YUING
```
根据以上适配内容，对 MAPFGPTPathPlanner 类中的_plan_joint_one_step()方法进行了调整
增加了对于 is_wait 的额外处理，将 waiting agent 作为"目标=当前位置"的 agent 纳入观测，以避免影响其他 agent 的规划
但是发现依然存在冲突问题，考虑 将 waiting agent 的位置加入障碍矩阵 (即在构建 padded_obstacles 时，将 waiting agent 的格子标为 1)
```
