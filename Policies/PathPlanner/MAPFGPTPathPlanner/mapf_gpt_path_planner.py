"""
MAPF-GPT Path Planner Adapter
=============================
Adapter that plugs MAPF-GPT inference into the MAS_RMFS path planner interface.

调用链（修复版）
--------------
MAS_RMFS 每 tick 对有任务但无路径的 agent 逐个调用 plan()。
MAPF-GPT 需要所有活跃 agent 的联合观测才能推理。

本适配器的解决方案：
  1. 每个 tick 第一次调用 plan() 时触发 _plan_joint_one_step()，对全部
     活跃 agent 做一次联合推理，把结果（下一步坐标）存入 _step_cache。
  2. 同 tick 内后续的 plan() 调用直接读缓存，不再推理。
  3. 推理时不调用 act()，而是直接调用 generate_input() + encoder + net：
       - act() 的 else 分支会用 position_history 计算上一步动作并强制写入
         actions_history，破坏我们精心维护的历史，必须绕过。
       - generate_input() 只需要 cost2go_data + actions_history + 观测坐标，
         完全不需要 position_history。
  4. cost2go 仅依赖静态障碍图，第一次计算后跨 tick 复用。
  5. actions_history 按 agent_id 在 adapter 层持久化，每次推理后
     根据实际执行的动作更新。
  6. 依赖/权重加载失败时整体回退到 AStarPathPlanner。
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from Debug.logger import SimLogger
from Policies.PathPlanner.AStarPathPlanner.astar_path_planner import AStarPathPlanner
from Policies.PathPlanner.base_path_planner import BasePathPlanner

# MAPF-GPT action index → (Δrow, Δcol)
# 与 gpt/inference.py 中的 moves 字典严格对齐：
#   0=wait(0,0)  1=up(-1,0)  2=down(1,0)  3=left(0,-1)  4=right(0,1)
_ACTION_TO_DELTA: Dict[int, Tuple[int, int]] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}
# (Δrow, Δcol) → 动作字符（与 gpt/inference.py moves 字典对齐）
_DELTA_TO_STR: Dict[Tuple[int, int], str] = {
    (0,  0):  "w",
    (-1, 0):  "u",
    (1,  0):  "d",
    (0, -1):  "l",
    (0,  1):  "r",
}
# 动作字符集（用于 actions_history 初始化和 vocab 校验）
_STR_TO_DELTA: Dict[str, Tuple[int, int]] = {v: k for k, v in _DELTA_TO_STR.items()}


class MAPFGPTPathPlanner(BasePathPlanner):
    """
    MAPF-GPT 联合推理适配器（绕过 act()，直接调用推理链）。

    参数
    ----
    mapf_gpt_root : str
        MAPF-GPT-main 目录路径（相对或绝对均可）。
    model : str
        模型规模："2M" / "6M" / "85M"。
    device : str
        推理设备："cuda" / "cpu" / "mps"。
    min_joint_agents : int
        触发联合推理所需的最少活跃 agent 数；不足时每个 agent 单独走 A*。
    avoid_agents : bool
        A* 回退时是否绕开其他 agent 当前位置。
    cost2go_radius : int
        与 MAPF-GPT tokenizer 保持一致的观测半径（默认 5）。
    num_previous_actions : int
        历史动作窗口长度，需与训练参数一致（默认 5）。
    """

    def __init__(
        self,
        mapf_gpt_root: str = "MAPF-GPT-main",
        model: str = "2M",
        device: str = "cpu",
        min_joint_agents: int = 2,
        avoid_agents: bool = True,
        cost2go_radius: int = 5,
        num_previous_actions: int = 5,
    ):
        self.mapf_gpt_root = mapf_gpt_root
        self.model = model
        self.device = device
        self.min_joint_agents = max(1, int(min_joint_agents))
        self.avoid_agents = avoid_agents
        self.cost2go_radius = int(cost2go_radius)
        self.num_previous_actions = int(num_previous_actions)

        self.logger = SimLogger("MAPFGPTPathPlanner")
        self.fallback_planner = AStarPathPlanner(avoid_agents=avoid_agents)

        # 后端组件（懒初始化）
        self._backend_ready: bool = False
        self._backend_init_failed: bool = False
        self._mapf_algo = None      # MAPFGPTInference 实例
        self._cost2go_mod = None    # tokenizer.cost2go C++ 模块
        self._torch = None          # torch 模块

        # cost2go 缓存（只算一次）
        # key: (padded_rows, padded_cols) 元组，值: precompute_cost2go 返回的字典
        self._cost2go_cache = None

        # 每个 agent 的动作历史（持久化，按 agent_id 索引）
        # 格式：list[str]，长度 <= num_previous_actions，元素为 "n"/"w"/"u"/"d"/"l"/"r"
        self._agent_history: Dict[int, List[str]] = {}

        # 每 tick 的单步缓存
        self._step_cache: Dict[int, List[Tuple[int, int]]] = {}
        self._last_tick: int = -1

    # ------------------------------------------------------------------
    # 后端懒初始化
    # ------------------------------------------------------------------

    def _try_init_backend(self) -> bool:
        if self._backend_ready:
            return True
        if self._backend_init_failed:
            return False

        try:
            root_abs = os.path.abspath(self.mapf_gpt_root)
            if not os.path.isdir(root_abs):
                raise FileNotFoundError(f"MAPF-GPT root not found: {root_abs}")

            if root_abs not in sys.path:
                sys.path.insert(0, root_abs)

            # 导入推理模块
            infer_mod = importlib.import_module("gpt.inference")
            cfg_cls = infer_mod.MAPFGPTInferenceConfig
            algo_cls = infer_mod.MAPFGPTInference

            model_name = self.model.upper()
            if model_name not in {"2M", "6M", "85M"}:
                raise ValueError("model must be one of: 2M, 6M, 85M")

            weights_path = os.path.join(root_abs, "weights", f"model-{model_name}.pt")
            cfg = cfg_cls(
                path_to_weights=weights_path,
                device=self.device,
                cost2go_radius=self.cost2go_radius,
                num_previous_actions=self.num_previous_actions,
            )
            self._mapf_algo = algo_cls(cfg)

            # 导入 C++ cost2go 模块（已被 gpt.inference 触发编译/加载）
            self._cost2go_mod = importlib.import_module("tokenizer.cost2go")

            # 导入 torch（此时一定已安装）
            import torch as _torch
            self._torch = _torch

            self._backend_ready = True
            self.logger.info(
                f"MAPF-GPT backend ready: model={model_name}, device={self.device}"
            )
            return True

        except Exception as e:
            self._backend_init_failed = True
            self.logger.warning(f"MAPF-GPT backend init failed; fallback to A*: {e}")
            return False

    # ------------------------------------------------------------------
    # 地图工具
    # ------------------------------------------------------------------

    def _build_padded_obstacles(self, world_state) -> Tuple[np.ndarray, int]:
        """
        构建带 border 的障碍矩阵并返回 (padded_grid, border)。

        padding 方式与 MAPF-GPT 数据集生成完全一致：
          - 原始地图嵌入 (rows+2b) × (cols+2b) 的全障碍矩阵中心
          - 这样所有内部可走格的 padded 坐标都在 cost2go 的有效键范围内
        """
        ms = world_state.map_state
        inner = np.zeros((ms.rows, ms.cols), dtype=np.int32)
        for r in range(ms.rows):
            for c in range(ms.cols):
                if not ms.is_walkable(r, c):
                    inner[r][c] = 1

        b = self.cost2go_radius
        padded = np.ones((ms.rows + 2 * b, ms.cols + 2 * b), dtype=np.int32)
        padded[b : b + ms.rows, b : b + ms.cols] = inner
        return padded, b

    # ------------------------------------------------------------------
    # 辅助：解析 agent 当前目标
    # ------------------------------------------------------------------

    def _resolve_goal(self, world_state, agent) -> Optional[Tuple[int, int]]:
        """从 task_state 中找到该 agent 的当前目标坐标。"""
        task = world_state.task_state.get_active_task_for_agent(agent.agent_id)
        if task is not None:
            return task.destination
        task = world_state.task_state.get_next_task_for_agent(agent.agent_id)
        if task is not None:
            return task.destination
        return None

    # ------------------------------------------------------------------
    # 辅助：校验下一步位置（原始坐标系）
    # ------------------------------------------------------------------

    def _valid_move(self, agent, next_pos: Tuple[int, int], world_state) -> bool:
        ms = world_state.map_state
        r, c = next_pos
        if not ms.in_bounds(r, c) or not ms.is_walkable(r, c):
            return False
        # 携带货架时不能踩踏其他静止货架格
        if agent.carried_pod_id is not None:
            for pod in world_state.pod_state.pods.values():
                if not pod.is_carried and pod.current_position == next_pos:
                    return False
        return True

    # ------------------------------------------------------------------
    # 核心推理：每 tick 联合推理一次
    # ------------------------------------------------------------------

    def _plan_joint_one_step(self, world_state) -> None:
        """
        对当前所有活跃（非 idle、非 waiting）且有目标的 agent 联合推理一步。
        结果存入 self._step_cache[agent_id] = [(next_row, next_col)]。

        ⚠️ 不调用 MAPFGPTInference.act()，因为 act() 的 else 分支会用
           position_history 倒推上一步动作并强制写入 actions_history，破坏
           我们在 adapter 层精心维护的历史。
           改为直接调用 generate_input() → encoder.encode() → net.act()。
        """
        self._step_cache = {}

        # ── 1. 收集活跃 agent ──────────────────────────────────────────
        # 注意：is_waiting 的 agent 也参与联合观测（goal=当前位置），
        # 这样 MAPF-GPT 能感知它们的存在并为其他 agent 规划避让路径。
        controllable: List[Tuple] = []
        for agent in world_state.agents:
            if agent.is_idle:
                continue
            goal = self._resolve_goal(world_state, agent)
            if goal is None:
                continue
            controllable.append((agent, goal))

        if len(controllable) < self.min_joint_agents:
            # 活跃 agent 不足，不做联合推理；plan() 里会各自用 A* 回退
            return

        border = self.cost2go_radius  # 常量，等于 padding 宽度

        # ── 2. 首次构建 cost2go（后续复用缓存）────────────────────────
        if self._cost2go_cache is None:
            padded_obs, b = self._build_padded_obstacles(world_state)
            assert b == border, "border mismatch"
            grid_list = padded_obs.astype(int).tolist()
            self._cost2go_cache = self._cost2go_mod.precompute_cost2go(
                grid_list, self.cost2go_radius
            )
            self.logger.info(
                f"cost2go computed and cached "
                f"(padded grid: {padded_obs.shape[0]}×{padded_obs.shape[1]})"
            )

        # ── 3. 构造 observations（只需要 global_xy / global_target_xy）─
        observations = []
        for agent, goal in controllable:
            gr = agent.position[0] + border
            gc = agent.position[1] + border
            tr = goal[0] + border
            tc = goal[1] + border
            observations.append(
                {
                    "global_xy": (gr, gc),
                    "global_target_xy": (tr, tc),
                    # global_obstacles 不传给 generate_input()，这里省略
                }
            )

        # ── 4. 准备 actions_history（我们管理的历史）──────────────────
        ordered_history: List[List[str]] = []
        for agent, _ in controllable:
            aid = agent.agent_id
            if aid not in self._agent_history:
                self._agent_history[aid] = ["n"] * self.num_previous_actions
            ordered_history.append(list(self._agent_history[aid]))

        # ── 5. 设置 MAPF-GPT 内部状态（仅 cost2go_data + actions_history）
        self._mapf_algo.cost2go_data = self._cost2go_cache
        self._mapf_algo.actions_history = ordered_history
        # position_history 不需要：generate_input() 完全不读取该字段
        self._mapf_algo.position_history = None

        # ── 6. 调用推理链（绕过 act()）────────────────────────────────
        try:
            # 6a. 生成结构化观测（使用 cost2go_data + actions_history）
            inputs = self._mapf_algo.generate_input(observations)
        except Exception as e:
            self.logger.warning(f"MAPF-GPT generate_input failed: {e}")
            return

        try:
            # 6b. 编码为 token 序列
            torch = self._torch
            tensor_obs = torch.tensor(
                [self._mapf_algo.encoder.encode(inp) for inp in inputs],
                dtype=torch.long,
                device=self.device,
            )
            # 6c. 模型前向推理
            with torch.no_grad():
                raw_actions = self._mapf_algo.net.act(tensor_obs)

            # 6d. 解析输出（squeeze 处理单 agent 时的标量情形）
            squeezed = torch.squeeze(raw_actions)
            if squeezed.dim() == 0:
                actions: List[int] = [int(squeezed.item())]
            else:
                actions = [int(a) for a in squeezed.tolist()]
        except Exception as e:
            self.logger.warning(f"MAPF-GPT net.act failed: {e}")
            return

        if len(actions) != len(controllable):
            self.logger.warning(
                f"MAPF-GPT returned {len(actions)} actions "
                f"for {len(controllable)} agents; fallback this tick."
            )
            return

        # ── 7. 把结果转为坐标，写缓存，更新历史──────────────────────
        for idx, (agent, goal) in enumerate(controllable):
            # waiting agent 仅参与观测，不执行动作、不写缓存
            if agent.is_waiting:
                continue

            action_idx = actions[idx]
            dr, dc = _ACTION_TO_DELTA.get(action_idx, (0, 0))
            next_pos = (agent.position[0] + dr, agent.position[1] + dc)

            if not self._valid_move(agent, next_pos, world_state):
                # MAPF-GPT 建议的动作不合法（障碍/越界/货架占用）
                # 用 A* 单步替代
                fb = self.fallback_planner.plan(agent, goal, world_state)
                if fb:
                    next_pos = fb[0]
                    dr = next_pos[0] - agent.position[0]
                    dc = next_pos[1] - agent.position[1]
                    self._step_cache[agent.agent_id] = [next_pos]
                # else: A* 也找不到路，不写缓存；plan() 里再尝试完整 A*
            else:
                self._step_cache[agent.agent_id] = [next_pos]

            # 更新持久化动作历史（用实际执行的 delta）
            act_str = _DELTA_TO_STR.get((dr, dc), "w")
            hist = self._agent_history.setdefault(
                agent.agent_id, ["n"] * self.num_previous_actions
            )
            hist.append(act_str)
            self._agent_history[agent.agent_id] = hist[-self.num_previous_actions:]

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def plan(
        self,
        agent,
        goal: Tuple[int, int],
        world_state,
    ) -> List[Tuple[int, int]]:
        """
        返回下一步路径（MAPF-GPT 给出 1 步；A* 回退时给完整路径）。

        engine 只在 agent 无路径时调用 plan()，所以 MAPF-GPT 的"每 tick
        只给 1 步"模式会让 engine 每 tick 重规划，保持与 MAPF-GPT
        联合推理的节奏一致。
        """
        tick = world_state.tick

        # 每 tick 第一次被调用时执行联合推理
        if tick != self._last_tick:
            self._last_tick = tick
            self._step_cache = {}
            if self._try_init_backend():
                self._plan_joint_one_step(world_state)

        if self._backend_ready:
            cached = self._step_cache.get(agent.agent_id)
            if cached is not None:
                return cached
            # 该 agent 未进入联合推理（如活跃数不足 min_joint_agents）
            # 回退到完整 A* 路径
            return self.fallback_planner.plan(agent, goal, world_state)

        # 后端初始化失败：完整 A*
        return self.fallback_planner.plan(agent, goal, world_state)
