# GT 子图构建模块 — 从头复现使用说明

本文档说明如何在本地从零跑通 `gt_subgraph/` 全流程：**环境准备 → 数据校验 → 渲染冒烟测试 → 阈值统计 → GT 构建 → 可视化验证**。

---

## 0. 模块做什么

把 3RScan 的一个 scan（mesh + 实例分割 + 相机轨迹 + 3DSSG 关系标注）转换成 **按连续时间步 `t` 单调增长的 GT 场景图序列** `{G*_{≤t}}`。

- **输入**：静态最终 GT + 逐帧位姿
- **输出**：每个 scan 一个紧凑 JSON（节点 `commit_time`、边 `activation_time`）
- **消费方式**：`output.materialize(out, t)` 按需重建任意时刻的子图
- **节点范围**：默认 `NODE_SCOPE=rel_endpoints`：关系端点静态白名单在 **每个 t** 的 `materialize(t)` 中生效（非仅最后时刻）；可用 `--node-scope all_renderable` 恢复「所有可见 commit 节点」

推荐复现顺序（与 handoff 文档第 4 节一致）：

```
smoke_test.py  →  stats.py  →  run.py  →  visualize.py  →  visualize_compare.py
  渲染对齐         定阈值        正式构建      子图可视化        与原始标注对比
```

---

## 1. 环境准备

### 1.1 Python 版本

建议 **Python 3.9+**（与 Open3D 兼容即可）。

### 1.2 依赖包

| 包 | 用途 |
|---|---|
| `numpy` | 数组、几何计算 |
| `open3d` | RaycastingScene 渲染 face-ID buffer |
| `matplotlib` | stats 直方图、visualize 平面图/GIF |
| `Pillow` | smoke_test RGB 叠加 |
| `tqdm` | 逐帧进度条 |

安装示例（任选其一）：

```bash
# pip
pip install numpy open3d matplotlib pillow tqdm

# conda（推荐，Open3D 预编译包较稳）
conda create -n agentsgg python=3.10 -y
conda activate agentsgg
conda install -c conda-forge open3d numpy matplotlib pillow tqdm -y
```

验证：

```bash
python -c "import open3d, numpy, matplotlib, PIL, tqdm; print('deps ok')"
```

### 1.3 工作目录

后续命令均在 **`gt_subgraph/`** 目录下执行：

```bash
cd /path/to/AgentSGG/gt_subgraph
```

---

## 2. 数据准备

### 2.1 数据集根目录结构

默认代码中 `ROOT` 指向：

```
/home/data16t1/fengchangqun/AgentSGG/3RScan
```

若你的数据在其他位置，需修改以下文件顶部的 `ROOT` 常量（见第 3 节）：

- `run.py`
- `stats.py`
- `visualize.py`
- `smoke_test.py`

期望的目录布局：

```
<ROOT>/
├── 3DSSG_subset/
│   ├── relationships.json
│   ├── relationships_train.json
│   ├── relationships_validation.json
│   ├── relationships_test.json
│   ├── classes.txt
│   ├── relationships.txt
│   ├── train_scans.txt
│   └── validation_scans.txt
├── 3RScan.json                    # 本模块不读对齐变换，可忽略
└── <scan_id>/                     # 每个 scan 一个文件夹
    ├── mesh.refined.v2.obj
    ├── mesh.refined.0.010000.segs.v2.json
    ├── semseg.v2.json
    ├── labels.instances.annotated.v2.ply
    └── sequence/  或  sequence.zip   # 二者有其一即可
        ├── _info.txt
        ├── frame-000000.pose.txt
        ├── frame-000000.color.jpg   # smoke_test 叠加用；主流程不依赖 depth
        └── ...
```

### 2.2 仓库自带的三个样本 scan

| scan_id | split | 特点 |
|---|---|---|
| `7272e16c-a01b-20f6-8961-a0927b4a7629` | validation | 浴室，117 帧，11 关系，适合入门 |
| `7272e161-a01b-20f6-8b5a-0b97efeb6545` | train | 客厅，151 帧，251 关系，含 annotation-only 端点 |
| `f62fd5fd-9a3f-2f44-883a-1e5cf819608e` | train | 471 帧，较大场景 |

### 2.3 数据自检清单

进入某个 scan 目录，确认以下文件存在：

```bash
SCAN=7272e16c-a01b-20f6-8961-a0927b4a7629
ls $ROOT/$SCAN/{mesh.refined.v2.obj,semseg.v2.json,mesh.refined.0.010000.segs.v2.json,labels.instances.annotated.v2.ply}
ls $ROOT/$SCAN/sequence 2>/dev/null || ls $ROOT/$SCAN/sequence.zip
```

**注意**：

- `sequence/` 与 `sequence.zip` 二选一；代码会自动回退读取 zip 内的 pose / 内参 / RGB
- 不要对 rescan 施加 `3RScan.json` 里的 reference 对齐变换
- 本模块 **不读传感器 depth**，可见性完全来自 mesh 渲染

---

## 3. 路径配置（换机器必做）

所有路径集中在 **`paths.py`**，各脚本统一 import，无需逐个改文件。

```
gt_subgraph/
├── paths.py              ← 只改这里
├── outputs/gt/           ← run.py 全量主产物
└── demo/
    ├── overlays/         ← smoke_test.py
    ├── stats/            ← stats.py
    └── viz/              ← visualize.py（读 outputs/gt/ 里的 JSON）
```

| 变量 | 默认值 | 含义 |
|---|---|---|
| `ROOT` | `<项目根>/3RScan` | 3RScan 数据集根目录 |
| `GT_OUT` | `gt_subgraph/outputs/gt` | 全量 GT JSON（`run.py`） |
| `DEMO_OVERLAYS` | `gt_subgraph/demo/overlays` | RGB 叠加图 |
| `DEMO_STATS` | `gt_subgraph/demo/stats` | 阈值分布 |
| `DEMO_VIZ` | `gt_subgraph/demo/viz` | 子图可视化 |

首次复现前，若数据不在项目下的 `3RScan/`，只需修改 `paths.py` 里的 `ROOT`（或改 `PROJECT_ROOT` 的推导方式）。

目录会自动创建；`.gitignore` 已忽略 `outputs/` 与 `demo/`。

---

## 4. 逐步复现

以下以 **7272e16c** 为例；其他 scan 替换 scan_id 即可。

### Step 1 — 渲染冒烟测试（`smoke_test.py`）

**目的**：验证 mesh/seg 对齐、坐标系、遮挡渲染是否正确（handoff §4.1）。

```bash
cd gt_subgraph
python smoke_test.py
```

默认 scan 为 `paths.DEFAULT_SCANS[0]`（7272e16c）；换 scan 可改 `smoke_test.py` 里的 `SCAN` 或后续统一从 `paths.py` 引用。

**控制台应看到**：

- `verts == segIdx` 数量一致
- `PLY-crosscheck disagree` 比例极低（通常 <1%）
- 5 个采样帧的 `visible_instances` 和 top 物体列表

**产物**（写入 `demo/overlays/`）：

| 文件 | 说明 |
|---|---|
| `overlay_frame-000000.png` | RGB 与 instance 着色 50% 混合 |
| `overlay_frame-000029.png` | 同上（1/4 处） |
| `overlay_frame-000058.png` | 同上（1/2 处） |
| `overlay_frame-000087.png` | 同上（3/4 处） |
| `overlay_frame-000116.png` | 同上（末帧） |

**肉眼检查**：

- 着色轮廓与 RGB 中物体对齐 → 坐标/内参正确
- 被遮挡物体不应「漏出」到前景 → z-buffer 正常
- 若 overlay 整体错位 → 检查是否误用了 rescan 对齐变换

---

### Step 2 — 阈值分布统计（`stats.py`）

**目的**：收集面积比例、像素数、遮挡比分布，用于定 `TAU_*`（handoff §4.2）。  
此步 **不会** 用最终阈值 commit，仅收集证据样本。

```bash
python stats.py 7272e16c-a01b-20f6-8961-a0927b4a7629
```

不传参数时默认只跑 7272e16c；可传多个 scan_id。

**产物**：

| 文件 | 说明 |
|---|---|
| `stats_7272e16c.json` | 分位数报告（p50/p75/p90/p95/p99） |
| `stats_7272e16c.png` | 四张直方图 |

**如何读 `stats_*.json` 定阈值**（参考 handoff §5）：

| 参数 | 建议读法 |
|---|---|
| `TAU_INST_PIX_MIN` | 看 `inst_pixels_per_frame.pct["1"]`（1% 分位），取 ~10–20 |
| `TAU_INST_VIS_RATIO` | 看 `vis_ratio.pct`，取 0.1–0.2 挡极端 sliver |
| `TAU_FACE_PIX` | 固定小值 1–2 |
| `TAU_STRONG` | 看 `single_area_ratio.pct["90"]`/`["95"]`，**不要默认 0.6** |
| `TAU_COMMIT` | 看 `cumulative_area_ratio` 分布，常见 ~0.3–0.4 |
| `ENABLE_PERSIST` / `K` / `TAU_PERSIST` | 先用默认；若小物体漏 commit 再开 persist 分支 |

7272e16c 参考值（仅供对照）：

- `single_area_ratio` p90 ≈ 0.68
- `cumulative_area_ratio` p50 ≈ 0.75
- `inst_pixels_per_frame` p1 ≈ 39

---

### Step 3 — 正式 GT 构建（`run.py`）

**目的**：Phase 0 → Phase A → Phase B，输出主产物 `gt_*.json`。

```bash
# 跑默认三个样本（默认 NODE_SCOPE=rel_endpoints）
python run.py

# 只跑一个 scan
python run.py 7272e16c-a01b-20f6-8961-a0927b4a7629

# 保留所有可见 commit 节点（旧行为，含关系文件外的 toilet/towel 等）
python run.py --node-scope all_renderable 7272e16c-a01b-20f6-8961-a0927b4a7629
```

**默认阈值与节点范围**（在 `run.py` 的 `main()` 中，可按 stats 结果修改）：

```python
cfg = gb.Config(
    TAU_INST_PIX_MIN=20,
    TAU_INST_VIS_RATIO=0.10,
    TAU_FACE_PIX=2,
    TAU_STRONG=0.6,
    TAU_COMMIT=0.4,
    ENABLE_PERSIST=True,
    K=3,
    TAU_PERSIST=0.10,
    NODE_SCOPE="rel_endpoints",  # 或 "all_renderable"
)
```

| `NODE_SCOPE` | 下游 JSON / `materialize(t)` 中的节点 |
|---|---|
| `rel_endpoints`（**默认**） | 仅当前关系文件里作为 subject/object 出现、且已 commit 的 instance |
| `all_renderable` | 所有有几何且已 commit 的 instance（可能多于标注子图，如 toilet/mirror） |

**`NODE_SCOPE` 的时序语义（重要）**

- 构建时在关系文件中算出 **`rel_endpoint_ids` 静态白名单**，之后不随 `t` 变化。
- 过滤作用于 **每一次** `materialize(out, t)`，**不是**只在最后一个时刻才删节点。
- `rel_endpoints` 下，节点需同时满足：`commit_time <= t` **且** `id ∈ rel_endpoint_ids`；边要求两端都在白名单内且 `activation_time <= t`。
- 因此：不在关系端点集合里的 instance（如 toilet），即便在 Phase A 很早就被 commit，也 **不会出现在任意时刻** 的 `G*_{≤t}` 中；其 commit 记录仅保留在 `debug.nodes_excluded_by_scope`。
- Phase A 仍对全部可渲染 instance 做可见性 commit；`NODE_SCOPE` 只影响 **写入 JSON 与 `materialize` 的节点/边集合**。

7272e16c（validation，9 个关系端点）在 `rel_endpoints` 下的节点数随 `t` 变化示例：

| t | 说明 | 节点数 |
|---|------|--------|
| 0 | toilet 等已 commit，但不在 9 端点内 → 被滤掉 | 0 |
| 2 | wall(1)、floor(5) commit 且在端点内 | 2 |
| 116 | 9 个端点全部 commit | 9 |

被排除的节点可在 `debug.nodes_excluded_by_scope` 中查看。

修改阈值或 `NODE_SCOPE`：编辑 `run.py` 中 `Config(...)` / 命令行 `--node-scope` 后重跑。

**控制台应看到**（每个 scan）：

```
=== <scan_id>  (split=...) ===
  [invariant] obj_v=... seg=... ply=... faces=...
  [phase0] semseg_inst=... has_geometry=... annotation_only=...
  [id_check] relations=... missing_relation_rows=0
  [phaseA] committed X/Y renderable nodes, reasons={...}
  [phaseB] active edges A/B (missing endpoints=...)
  [output] NODE_SCOPE=rel_endpoints: 9 nodes in JSON, 11 committed but excluded ...
  [timeline] first commits: [...]
    materialize(t=...): N nodes, M edges, edge-before-endpoint violations=0
  saved .../outputs/gt/gt_<前8位>.json
```

**关键通过标准**：

- `missing_relation_rows=0`（或 <5% 否则抛错）
- `edge-before-endpoint violations=0`
- 7272e16c 预期（`NODE_SCOPE=rel_endpoints`）：Phase A `20/20` commit；JSON / 最终 `materialize` **9 节点、11 边**（与 validation 标注一致）
- 若 `--node-scope all_renderable`：最终 **20 节点、11 边**（多 11 个无边节点）

**主产物** `gt_<scan前8位>.json` 结构摘要：

| 字段 | 含义 |
|---|---|
| `num_processed_frames` | 有效连续 t 的帧数 |
| `t_to_frame` | `t` → 原始 `frame-XXXXXX` |
| `config` | 本次使用的全部阈值 + `NODE_SCOPE` |
| `rel_endpoint_ids` | 关系文件中的 subject/object 端点 id 列表 |
| `nodes` | 已 commit 且通过 `NODE_SCOPE` 过滤的节点 |
| `edges` | 有向关系 + `activation_time`（null 表示未激活） |
| `id_check` | ID 对齐报告 |
| `debug.uncommitted_renderable` | 有几何但未 commit 的实例诊断 |
| `debug.nodes_excluded_by_scope` | 已 commit 但因 `NODE_SCOPE=rel_endpoints` 未写入 `nodes` 的实例 |

**程序化读取示例**：

```python
import json, sys
sys.path.insert(0, "gt_subgraph")
import output as op

with open("gt_subgraph/outputs/gt/gt_7272e16c.json") as f:
    out = json.load(f)

# 使用 JSON 内 config.NODE_SCOPE（默认 rel_endpoints）
nodes, edges = op.materialize(out, t=10)

# 临时覆盖：查看所有 commit 节点
nodes_all, edges_all = op.materialize(out, t=10, node_scope="all_renderable")
print(len(nodes), len(edges), len(nodes_all))
```

---

### Step 4 — 子图可视化（`visualize.py`）

**目的**：验证 `materialize(t)` 随时间增长的合理性（handoff §4.4）。  
**前置条件**：对应 scan 的 `gt_*.json` 已由 Step 3 生成。

```bash
python visualize.py 7272e16c-a01b-20f6-8961-a0927b4a7629
```

不传参数时默认 7272e16c。

**产物**（写入 `demo/viz/`，每个 scan 三个文件）：

| 文件 | 内容 |
|---|---|
| `viz_<前8位>_panel.png` | 2×3 俯视图，在 t=0, T/4, T/2, 3T/4, T-1 五个时刻的子图对比 |
| `viz_<前8位>_floorplan.gif` | 俯视图动画，子图随 t 增长（最多 ~60 帧采样） |
| `viz_<前8位>_timeline.png` | 上：节点/边数量随 t 的阶梯曲线；下：各节点 commit 事件 |

**图例说明**：

- 布局：每个 instance 的面积加权 3D 质心，投影到 room 水平面（自动检测 floor 的 up 轴）
- 节点填充色：`commit_time`（viridis，早=蓝，晚=黄）
- 节点边框色：`commit_reason`（红=strong_single，蓝=cumulative，绿=persistent）
- 边：3DSSG 谓词颜色；边少时带箭头

**检查要点**：

- timeline 中节点/边数量应单调不减
- panel 中早时刻节点少、晚时刻多
- 边不应出现在其端点 commit 之前（与 run.py 的 materialize 校验一致）

---

### Step 5 — 与原始标注对比（`visualize_compare.py`）

**目的**：并排对比 **3DSSG 原始关系标注** 与 **最终时序子图** `materialize(out, T-1)`，人眼检查结构是否一致。  
**前置条件**：Step 3 已生成 `gt_*.json`。

```bash
# 左：split 关系文件标注；右：最终时序 GT（默认 NODE_SCOPE 下应与左一致）
python visualize_compare.py 7272e16c-a01b-20f6-8961-a0927b4a7629 --rel split

# 三列：split 标注 | relationships.json 全量 | 最终时序 GT
python visualize_compare.py 7272e16c-a01b-20f6-8961-a0927b4a7629 --rel both
```

**产物**（`demo/viz/`）：

| 文件 | 内容 |
|---|---|
| `compare_<前8位>_split.png` | 2 列俯视图对比 |
| `compare_<前8位>_split.json` | 节点/边 diff 报告 |
| `compare_<前8位>_both.png` | 3 列对比（含 full 标注） |

在默认 `NODE_SCOPE=rel_endpoints` 下，7272e16c 的 split 对比应为 **9 节点、11 边** 两侧完全一致。

---

## 5. 一键复现命令（单 scan）

以 7272e16c 为例，完整流水线：

```bash
cd /path/to/AgentSGG/gt_subgraph
conda activate agentsgg   # 或你的虚拟环境

# 1. 渲染对齐
python smoke_test.py

# 2. 阈值统计（可选但推荐）
python stats.py 7272e16c-a01b-20f6-8961-a0927b4a7629

# 3. （可选）根据 stats_*.json 修改 run.py 中的 Config，然后：
python run.py 7272e16c-a01b-20f6-8961-a0927b4a7629

# 4. 子图可视化
python visualize.py 7272e16c-a01b-20f6-8961-a0927b4a7629

# 5. 与原始标注对比
python visualize_compare.py 7272e16c-a01b-20f6-8961-a0927b4a7629 --rel split
```

跑全部三个默认样本：

```bash
python stats.py   # 需分别传三个 id，或改 stats.py 默认列表
python run.py     # 默认跑三个
python visualize.py 7272e16c-a01b-20f6-8961-a0927b4a7629 \
                    7272e161-a01b-20f6-8b5a-0b97efeb6545 \
                    f62fd5fd-9a3f-2f44-883a-1e5cf819608e
```

---

## 6. 产物总览

执行完 demo 全流程后，目录结构如下：

```
gt_subgraph/
├── outputs/gt/
│   └── gt_<scan8>.json          # run.py：全量主交付物
└── demo/
    ├── overlays/
    │   └── overlay_frame-*.png  # smoke_test：RGB 叠加
    ├── stats/
    │   └── stats_<scan8>.json/.png  # stats：阈值分布
    └── viz/
        ├── viz_<scan8>_panel.png
        ├── viz_<scan8>_floorplan.gif
        └── viz_<scan8>_timeline.png
```

**全量跑批**时通常只需 `outputs/gt/`；`demo/` 用于小样本验证与调参。

---

## 7. 代码模块速查

| 文件 | 职责 |
|---|---|
| `paths.py` | 统一 `ROOT` / `GT_OUT` / `demo/*` 路径 |
| `data_loader.py` | 读 mesh / semseg / segs / PLY / pose / 内参 / 关系 |
| `mesh_instance.py` | Phase 0：face→instance、面积、诊断、PLY 交叉校验 |
| `renderer.py` | Open3D 整场景 + 单 instance 渲染 |
| `gt_builder.py` | Phase A 证据累计 + commit；Phase B 边激活 |
| `output.py` | JSON 组装 + `materialize(t)` + `NODE_SCOPE` 过滤 |
| `run.py` | 主入口：Phase 0→A→B + 内置校验；`--node-scope` |
| `stats.py` | 阈值分布收集 |
| `smoke_test.py` | 渲染冒烟 + RGB 叠加 |
| `visualize.py` | 子图俯视图 + timeline + GIF |
| `visualize_compare.py` | 原始标注 vs 时序 GT 并排对比 |

---

## 8. 常见问题

### Q1: `ModuleNotFoundError: open3d`

在正确的 conda/venv 中安装 Open3D，并用该环境的 `python` 运行脚本。

### Q2: `vertex count mismatch: obj=... seg=... ply=...`

OBJ 顶点顺序与 segIndices/PLY 不一致。不要用 trimesh 等会合并顶点的 loader 替换 `load_mesh`；检查是否用了错误的 mesh 文件。

### Q3: `too many relationship rows have missing semseg endpoints`

关系文件中的 instance id 在 `semseg.v2.json` 里找不到。检查 scan_id 是否匹配、是否读错了 split 的关系文件。

### Q4: overlay 与 RGB 完全错位

常见原因：对 rescan 误用了 `3RScan.json` 的对齐变换。本模块应在 scan 原生坐标系内渲染。

### Q5: 大量 `activation_time: null` 的边

先查 `id_check.relation_endpoint_ids_without_renderable_geometry`：若端点是 annotation-only（semseg 有、mesh 无面），边永远无法激活，属预期行为。  
若端点有几何但未 commit，查 `debug.uncommitted_renderable` 并考虑调低 `TAU_COMMIT` 或开启 persist。

### Q6: `visualize.py` 报找不到 `gt_*.json`

先运行 `python run.py <scan_id>` 生成 `outputs/gt/gt_<前8位>.json`。

### Q7: 为什么 Phase A commit 了 20 个节点，JSON 里只有 9 个？

默认 `NODE_SCOPE=rel_endpoints`：toilet、towel 等虽被相机看到并在 Phase A commit，但不在当前关系文件的端点集合里，**在任意 t 的 `materialize` 中都不会出现**（不是只在最后时刻才删）。完整 commit 信息见 `debug.nodes_excluded_by_scope`；若需要全部节点，使用 `python run.py --node-scope all_renderable` 或在 `materialize(out, t, node_scope="all_renderable")` 中临时覆盖。

### Q8: 渲染很慢

Open3D raycast 对每帧、每个未 commit 实例可能做第二次渲染（算 vis_ratio）。大 scan（如 f62fd5fd 471 帧）需数分钟到十几分钟，属正常。

---

## 9. 与 handoff 文档的对应关系

| handoff 章节 | 本仓库脚本 |
|---|---|
| Phase 0 预处理 | `run.py` + `mesh_instance.py` |
| Phase A 逐帧 commit | `gt_builder.build_scan` |
| Phase B 边激活 | `gt_builder.activate_edges` |
| §4.1 渲染叠加 | `smoke_test.py` |
| §4.2 分布统计 | `stats.py` |
| §4.3 commit 时间线 | `run.py` 控制台 + `visualize timeline` |
| §4.4 子图物化 | `output.materialize` + `visualize.py` |
| 输出契约 | `output.py` |

---

## 10. 下一步（本模块之外）

本模块只产出 `{G*_{≤t}}`。下游阶段（未实现）将消费 `materialize(t)`：

- 奖励函数 Φ(belief, G*_{≤t})
- Oracle / SFT 标签（相邻帧 GT 差分）
- 策略 SFT + RL 训练
