基于PlaNet
1. 接入基准的Baseline 传统极速次优算法: PIBT (Priority-based Iterative BT) 传统最优流派： CBS (Conflict-Based Search) 或 LNS2 (Large Neighborhood Search，目前工业级 MAPF 的神级基线)
2. 构建算法核心 metrics: system throughout, deadlock, computation time(on GPUS) ...
3. 尝试用图神经网络表征(+因果方法？, ST-GNN)，离散隐空间的表达，以及梯度回传。
4. “全局”与“局部感受野”进行潜状态推断，重构热力图。如果没有订单信息，潜空间能否有效学习呢？
5. 热力图参与底层planning算法的启发项。
6. 证明引入了深度学习启发式惩罚后，底层算法依然能保证找到次优解的下界，且 100% 完备。


设计architecture
· Upper Level (预言机大脑 - Spatiotemporal Oracle)：
    Encoder: 采用 GNN（图神经网络）或局部 CNN，提取当前网格中 AGV 和任务目标的热力分布。
    Latent Rollout: 核心亮点为强调在低维潜空间 (Latent Space) 中自回归展开未来 K 步，全程矩阵相乘，耗时仅几毫秒，彻底摆脱物理引擎的缓慢。
    Decoder: 输出未来 t_1, t_2, ..., t_k 时刻的物理网格拥堵概率张量 P_WM(x, y, t)。
· Lower Level (执行小脑 - Proactive MAPF Planner)：
    将时空热力图融合到底层启发式搜索中。例如，魔改 Space-Time A* 的代价函数：Cost(n) = g(n) + h(n) + λ*Σ_τ P_WM(n_x, n_y, τ)



=========================================================================================================================================
 - 1. 传统最优、次优的路径规划保底
 - 2. PlaNet 动态赋能路径规划

Planet作为时空预言机，利用时空因果表征方法，对拥堵热力图进行表征和潜空间学习；或者参考大系统的结构（解耦的订单层也参与优化，利用前一刻的最优解参与计算）。复原未来的热力图，化为启发项作用到传统路径规划算法上。

对于离散事件的处理方法：
 - 分类离散潜在变量
 - CEM（交叉熵）连续动作规划器 替换为 Discrete CEM (离散交叉熵) 或者结合 MCTS (类似 MuZero 的搜索)
 - 图神经网络表达节点和量化

