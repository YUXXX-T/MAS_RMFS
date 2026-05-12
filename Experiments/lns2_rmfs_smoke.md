# LNS2 × RMFS Smoke Test

最小验证：把 LNS2Planner 接入 RMFS 全栈（订单 / 任务分配 / pod / station），跑短时长，确认能产出 snapshot 训练数据。

## 环境参数

| 项 | 值 |
|---|---|
| 地图 | 20×20（`Config/default_config.json` 的默认布局，8 个 pod_zone × 6×2，4 个 station 在四角附近） |
| 机器人数 | 4（starts: (0,0), (0,1), (0,2), (19,2)） |
| Pod 总数 | 96（3 种类型 × pod_zone 容量，每 pod 3 SKU） |
| 订单生成 | `ZipfOrderGenerator`，`order_interval=2`，`max_items_per_order=2`，`fixed_order_size=true`，`max_items_per_sku=3` |
| 任务分配 | `GreedyTaskAssigner`，`task_execution_mode=serial`（与 4 robot 量级匹配） |
| 路径规划 | `LNS2Planner`，`timeout=10s`，`neighbor_size=8`，`initAlgo=PP`，`replanAlgo=PP` |
| Pod 归还 | `HomeReturnPlanner` |
| Pod 检索 | `DefaultPodRetriever` |
| 订单录制 | `use_recorded_orders=false`（实时生成，便于多 seed 重跑） |
| `tick_delay` | 0.05 s |
| 终止方式 | `timeout 30` 外部 SIGTERM（引擎本身无 tick 上限，~500 tick） |
| Snapshot | 启用，每 tick 一行 JSONL |
| 输出目录 | `DataGen/snapshots/lns2_rmfs_smoke/` |

配置文件：`Experiments/configs/lns2_rmfs_smoke.json`

## 复现指令

```bash
cd /home/cnc/MAS_RMFS_wm
conda activate flat_lora_lab

# 首次准备：给三个外部 solver 二进制加执行权限（仅需一次）
chmod +x Policies/PathPlanner/ExternalSolverPlanner/MAPF-LNS2/build/lns \
         Policies/PathPlanner/ExternalSolverPlanner/EECBS/build/eecbs \
         Policies/PathPlanner/ExternalSolverPlanner/lacam2/build/main

# 30 秒 smoke test
timeout 30 python main.py --config Experiments/configs/lns2_rmfs_smoke.json
```

## 本次实测结果（2026-05-12）

- 运行时长：30 s（外部超时终止）
- 完成 tick 数：505
- Snapshot 文件大小：~11 MB
- LNS2 二进制：能正常调用（修复 `chmod +x` 后）
- 观测到的冲突：~10 次（包括 ONCOMING swap 与同 cell 占用）—— 因 LNS2 每 tick 独立批量求解，相邻 tick 间无 reservation 一致性，引擎执行层撞到。该问题独立于本测试，需要单独修。

## Snapshot 字段（已扩展）

**Episode 头部**（首行，`_header: true`）：
```json
{
  "_header": true, "episode_id": "...", "timestamp": "...", "git_commit": "...",
  "map":   {"rows", "cols", "obstacles": [[r,c]...], "stations": [{id, pos}...], "pod_homes": [[r,c]...]},
  "agents":{"count", "starts": [[r,c]...]},
  "mode":  "rmfs" | "mapf_benchmark",
  "planner": {"name", "params"}, "task_assigner": {...}, "order_generator": {...},
  "pod_return_planner", "pod_retriever",
  "num_robots", "order_interval", "max_items_per_order",
  "use_recorded_orders", "task_execution_mode"
}
```

**Per-tick** （每行一条）：
```json
{
  "tick": 0,
  "agent_positions":   [[id, [r,c]], ...],
  "agent_goals":       [[id, [r,c] | null], ...],
  "agent_statuses":    [[id, "IDLE|MOVING|WAITING|..."], ...],
  "planned_paths":     [[id, [[r,c],...剩余规划路径]], ...],
  "task_phases":       [[id, "PICK|DELIVER|RETURN" | null], ...],
  "replans":           [agent_id, ...],                            // 本 tick 路径签名变化的 agent
  "pod_positions":     [[pod_id, [r,c], is_carried], ...],
  "pending_orders":    [{order_id, sku_demands, station_id, status, created_at, pod_ids, delivered_pod_ids}, ...],
  "in_progress_orders":[...],
  "conflicts": {
      "vertex": [{"pos": [r,c], "agents": [id, ...]}, ...],
      "swap":   [{"a_prev": [r,c], "b_prev": [r,c], "agents": [aid_a, aid_b]}, ...]
  },
  "metrics": {                                                     // 等同 MetricsTracker.history[-1]
      "throughput_cumulative", "throughput_delta",
      "queue_length", "idle_agents",
      "congestion_count", "deadlock_agents",
      "planning_ms", "assignment_ms"
  }
}
```

## 下一步

- 数据格式 OK，可以开始按 demand.md §2.1 跑 baseline 网格（robot_counts × planners × assigners）
- 单独修：LNS2/EECBS/LaCAM2 在 RMFS 引擎下的 reservation 一致性（避免引擎执行层撞车）
- 可选：JSONL → Parquet/HDF5 转换工具（11 MB/500 tick × 多 run 仍可控，但训练时按 numpy 张量加载会更快）
