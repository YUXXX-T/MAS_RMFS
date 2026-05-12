import sys, logging, json, os, tempfile, copy
logging.disable(logging.CRITICAL)

from WorldState.order_state import Order
from WorldState.task_state import Task, TaskType, TaskStatus
from WorldState.agent_state import AgentStatus
from Config.config_loader import load_config
from Engine.simulation_engine import SimulationEngine
from Policies.policy_registry import get_policy
from Policies.PathPlanner.base_path_planner import BasePathPlanner
import Policies

passed = failed = 0
def check(name, cond, msg=""):
    global passed, failed
    if cond: passed += 1; print(f"  [PASS] {name}")
    else:    failed += 1; print(f"  [FAIL] {name} -- {msg}")

def build_engine(planner, ta_name="GreedyTaskAssigner", cfg_path="Config/default_config.json"):
    Task._next_id = 0; Order._next_id = 0
    cfg = load_config(cfg_path)
    cfg.simulation.tick_delay = 0
    cfg.simulation.log_level = "CRITICAL"
    on, op = cfg.policies.order_generator
    rn, rp_ = cfg.policies.pod_return_planner
    prn, prp = cfg.policies.pod_retriever
    _, ta_p = cfg.policies.task_assigner
    if cfg.simulation.use_recorded_orders:
        on = "RecordedOrderGenerator"
        op = {"recorded_orders_path": cfg.simulation.recorded_orders_path,
              "immediate_dispatch": cfg.simulation.immediate_dispatch}
    og = get_policy("order_generator", on)(
        order_interval=cfg.simulation.order_interval,
        max_items_per_order=cfg.simulation.max_items_per_order,
        fixed_order_size=cfg.simulation.fixed_order_size,
        max_items_per_sku=cfg.simulation.max_items_per_sku, **op)
    rp = get_policy("pod_return_planner", rn)(**rp_)
    pr = get_policy("pod_retriever", prn)(**prp)
    ta = get_policy("task_assigner", ta_name)(**ta_p)
    ta.pod_return_planner = rp; ta.pod_retriever = pr
    return SimulationEngine(config=cfg, order_generator=og, task_assigner=ta,
                            path_planner=planner, visualizer=None)

class BatchTracker(BasePathPlanner):
    def __init__(self):
        self.inner = get_policy("path_planner", "AStarPathPlanner")()
        self.plan_calls = 0; self.batch_calls = 0; self.batch_agent_counts = []
    def plan(self, agent, goal, ws):
        self.plan_calls += 1; return self.inner.plan(agent, goal, ws)
    def plan_batch(self, agents_with_goals, ws):
        self.batch_calls += 1; self.batch_agent_counts.append(len(agents_with_goals))
        result = {}
        for agent, goal in agents_with_goals:
            p = self.inner.plan(agent, goal, ws)
            if p: result[agent.agent_id] = p
        return result

class PartialBatchPlanner(BasePathPlanner):
    def __init__(self):
        self.inner = get_policy("path_planner", "AStarPathPlanner")()
    def plan(self, a, g, ws): return self.inner.plan(a, g, ws)
    def plan_batch(self, awg, ws):
        result = {}
        for i, (a, g) in enumerate(awg):
            if i % 2 == 0:
                p = self.inner.plan(a, g, ws)
                if p: result[a.agent_id] = p
        return result

# ================================================================
print("=" * 60)
print("TEST 1: Regression -- individual planners unchanged")
print("=" * 60)

for pp_name in ("AStarPathPlanner", "PrioritizedPathPlanner"):
    pp_cls = get_policy("path_planner", pp_name)
    pp_params = {"max_horizon": 100, "goal_reserve": 6} if pp_name == "PrioritizedPathPlanner" else {}
    pp = pp_cls(**pp_params)
    check(f"{pp_name} has no plan_batch", not hasattr(pp, "plan_batch"))
    eng = build_engine(pp)
    for _ in range(100): eng._tick()
    t = len(eng.world.task_state.tasks)
    check(f"{pp_name}: tasks created", t > 0)
    check(f"{pp_name}: metrics recorded", eng.metrics.summarize()["total_ticks"] == 100)

# ================================================================
print()
print("=" * 60)
print("TEST 2: Batch path exercised correctly")
print("=" * 60)

bt = BatchTracker()
eng = build_engine(bt)
for _ in range(100): eng._tick()
check("plan_batch was called", bt.batch_calls > 0, str(bt.batch_calls))
check("plan() was NOT called (batch overrides)", bt.plan_calls == 0, str(bt.plan_calls))
check("batch got >1 agent at once", any(n > 1 for n in bt.batch_agent_counts),
      str(bt.batch_agent_counts))
check("tasks created", len(eng.world.task_state.tasks) > 0)
check("metrics plan_ms >= 0", eng.metrics.summarize()["avg_plan_ms"] >= 0)

# ================================================================
print()
print("=" * 60)
print("TEST 3: Edge cases")
print("=" * 60)

# 3a: partial results
part = PartialBatchPlanner()
eng_p = build_engine(part)
crashed = False
try:
    for _ in range(200): eng_p._tick()
except Exception as e: crashed = True; print(f"    CRASH: {e}")
check("Partial batch: no crash 200 ticks", not crashed)
check("Partial batch: some tasks", len(eng_p.world.task_state.tasks) > 0)

# 3b: empty results
class EmptyBatch(BasePathPlanner):
    def plan(self, a, g, ws): return []
    def plan_batch(self, awg, ws): return {}
eng_e = build_engine(EmptyBatch())
crashed = False
try:
    for _ in range(50): eng_e._tick()
except Exception as e: crashed = True
check("Empty batch: no crash 50 ticks", not crashed)

# 3c: no orders -> planner never called
class NeverCalledBatch(BasePathPlanner):
    def __init__(self): self.called = False
    def plan(self, a, g, ws): self.called = True; return []
    def plan_batch(self, awg, ws): self.called = True; return {}
nc = NeverCalledBatch()
with open("Config/default_config.json") as f: cj = json.load(f)
cj["simulation"]["order_interval"] = 99999
cj["simulation"]["use_recorded_orders"] = False
cj["simulation"]["log_level"] = "CRITICAL"
fd, tp = tempfile.mkstemp(suffix=".json")
with os.fdopen(fd, "w") as f: json.dump(cj, f)
try:
    eng_nc = build_engine(nc, cfg_path=tp)
    for _ in range(10): eng_nc._tick()
    check("No orders -> planner never called", not nc.called)
finally: os.unlink(tp)

# ================================================================
print()
print("=" * 60)
print("TEST 4: Cross-component compatibility")
print("=" * 60)

bt2 = BatchTracker()
eng_h = build_engine(bt2, ta_name="HungarianTaskAssigner")
for _ in range(100): eng_h._tick()
check("Hungarian + batch: tasks", len(eng_h.world.task_state.tasks) > 0)
check("Hungarian + batch: batch called", bt2.batch_calls > 0)

bt3 = BatchTracker()
eng_g = build_engine(bt3, ta_name="GreedyTaskAssigner")
for _ in range(100): eng_g._tick()
check("Greedy + batch: tasks", len(eng_g.world.task_state.tasks) > 0)
check("Greedy + batch: batch called", bt3.batch_calls > 0)

s = eng_h.metrics.summarize()
check("Conflict detection active", "total_congestion_events" in s)
check("Orders completed",
      eng_g.world.order_state.total_completed > 0 or eng_h.world.order_state.total_completed > 0)

# ================================================================
print()
print("=" * 60)
print("TEST 5: Long-run stability (500 ticks)")
print("=" * 60)

bt_l = BatchTracker()
eng_l = build_engine(bt_l)
crashed = False
try:
    for _ in range(500): eng_l._tick()
except Exception as e: crashed = True; print(f"    CRASH at tick {eng_l.world.tick}: {e}")
check("500 ticks no crash", not crashed)
check("Tick = 500", eng_l.world.tick == 500)
s = eng_l.metrics.summarize()
check("Metrics complete", s["total_ticks"] == 500)
check("batch called many times", bt_l.batch_calls > 10, str(bt_l.batch_calls))
from collections import Counter
ap = [t.pod_id for t in eng_l.world.task_state.tasks.values()
      if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS) and t.task_type == TaskType.PICK]
dbl = {p: c for p, c in Counter(ap).items() if c > 1}
check("No pod double-assigned", len(dbl) == 0, str(dbl))
print(f"  completed={s['final_throughput']}, batch_calls={bt_l.batch_calls}, "
      f"plan_ms={s['avg_plan_ms']:.3f}")

# ================================================================
print()
print("=" * 60)
print("TEST 6: Batch wrapper produces valid results (not strict equality)")
print("=" * 60)
# NOTE: AStarPathPlanner with avoid_agents=True is order-sensitive.
# Batch mode plans all agents simultaneously (same blocked set),
# while individual mode plans sequentially (each agent sees prior
# agents' assigned positions). Both are valid; results may differ.

Task._next_id = 0; Order._next_id = 0
pp_ind = get_policy("path_planner", "AStarPathPlanner")()
eng_ind = build_engine(pp_ind)
for _ in range(100): eng_ind._tick()

Task._next_id = 0; Order._next_id = 0
bt_eq = BatchTracker()
eng_bt = build_engine(bt_eq)
for _ in range(100): eng_bt._tick()

ind_completed = eng_ind.world.order_state.total_completed
bt_completed = eng_bt.world.order_state.total_completed
check("Both complete orders",
      ind_completed > 0 and bt_completed > 0,
      f"ind={ind_completed}, batch={bt_completed}")
check("Throughput within 50% of each other",
      max(ind_completed, bt_completed) <= 2 * max(min(ind_completed, bt_completed), 1),
      f"ind={ind_completed}, batch={bt_completed}")
check("Both create tasks",
      len(eng_ind.world.task_state.tasks) > 0 and len(eng_bt.world.task_state.tasks) > 0,
      f"ind={len(eng_ind.world.task_state.tasks)}, batch={len(eng_bt.world.task_state.tasks)}")
check("Same number of agents",
      len(eng_ind.world.agents) == len(eng_bt.world.agents))

# ================================================================
print()
print("=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 60)
sys.exit(1 if failed > 0 else 0)
