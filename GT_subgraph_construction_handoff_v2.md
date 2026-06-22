# 任务移交文档:逐帧/累积 GT 场景图子图构建

> **本文档的范围**:仅实现**GT 子图构建**这一个模块。这是整个项目的第一块地基,后续还有奖励、oracle、策略训练等阶段(见文末"后续阶段")。本阶段是一个**确定性的标注/数据生成任务**,**不涉及任何策略、动作、奖励逻辑**。请严格守住这个边界。

---

## 0. 项目背景(让你理解这块代码服务于什么)

我们在做 **online / incremental 3D scene graph generation(在线增量式 3D 场景图生成)**。

整体思路:把该任务建模为 POMDP,用一个**学习得到的策略**(SFT + RL 训练)来替代以往方法(SceneGraphFusion / MonoSSG / FROSS / OGScene3D)中**手工设计的融合规则**(加权滑动平均、置信度累积、Hellinger 距离 + 阈值、多数投票等)。策略每帧对正在维护的场景图 belief 做图编辑(增删改节点/边,或 skip)。

**为什么需要本模块**:训练这个策略需要监督信号——奖励函数 Φ 和 oracle 动作标签。这两者都需要一个**"到第 t 帧为止,GT 场景图应该长什么样"**的参照,即一个**按时间索引的 GT 子图序列** `{G*_{≤t}}`。本模块就是从数据集已有的"最终 GT + 相机位姿"**确定性地构造出这个序列**。

数据集只提供一个**最终的、完整的** GT 场景图。我们利用每帧的相机位姿,把"到第 t 帧为止相机实际观测到了哪些 GT 实体/关系"渲染出来,从而把一个静态 GT 切成随时间生长的子图序列。

**再次强调本阶段的边界**:你只需要产出确定性的 `{G*_{≤t}}`。不要实现奖励、不要实现动作集合、不要实现策略模型、不要碰 SFT/RL。这些是消费 `{G*_{≤t}}` 的下游阶段。我们的目标不是构建复杂在线系统,而是构建一个**可信、单调、可解释的 GT 子图监督序列**。

---

## 1. 数据与输入

### 1.1 本阶段先跑通,不用全量数据

我会先提供 **3RScan 中的几个场景样本**(2~3 个 scan)。目标是先在小样本上**跑通 + 肉眼验证正确性 + 输出统计直方图**,不要一上来就处理全量数据集。代码要写成能平滑扩展到全量的形式(按 scan 循环、路径可配置),但验证和调参先在样本上做。

### 1.2 数据集目录结构(两层:数据集根目录 / 每个 scan 文件夹)

数据分两层。**每个 scan 文件夹**内是该次扫描的几何与 RGB-D 流;**数据集根目录**放全局元数据、场景图标注、划分文件、词表等。下面是一份真实目录里确认到的布局(以实际为准):

```
<dataset_root>/
├── 3RScan.json                     ← 全局元数据(reference↔rescan 对齐变换、train/val/test 类型)
├── 3DSSG_subset/                   ← 3DSSG 场景图标注目录(关系/物体/词表都在这里)
│   ├── relationships.json          ← 关系标注(全量,含多个 scan)
│   ├── relationships_train.json    ← 关系标注(train split,每 scan 含 objects + relationships + split)
│   ├── relationships_validation.json
│   ├── relationships_test.json
│   ├── classes.txt                 ← 物体类别词表(159 类)
│   ├── relationships.txt           ← 谓词词表(26 种)
│   └── train_scans.txt / validation_scans.txt   ← split 划分
├── obj_boxes_{train,val}_refined.json  ← 每 scan 每物体的 3D 框(本模块不需要)
├── <scan_id>/                      ← 单个 scan 文件夹
│   ├── sequence.zip                ← RGB-D 流(逐帧 color/depth/pose + 内参,约 117 帧)
│   ├── mesh.refined.v2.obj         ← 重建三角网格(几何)
│   ├── mesh.refined.mtl / mesh.refined_0.png  ← 材质/纹理
│   ├── mesh.refined.0.010000.segs.v2.json     ← mesh 过分割(顶点→segment id)
│   ├── semseg.v2.json              ← 实例分割(若干 segment 聚成一个物体,含 objectId/label/OBB)
│   └── labels.instances.annotated.v2.ply      ← 实例分割可视化(逐顶点上色)
└── ...
```

> 实际确认到的字段:关系记录为 `[subjectId, objectId, predId, predName]`(例 `[2, 1, 15, "standing on"]`);谓词 26 种、物体类别 159 种;深度 `depthShift=1000`(毫米)。这些与本文档其余部分一致,**但仍建议 agent 解压一个样本亲自核对一遍**,以防版本差异。

**scan 文件夹内、本模块要用的文件:**

| 文件 | 内容 | 本模块用途 |
|---|---|---|
| `mesh.refined.v2.obj` | 重建的三角网格(几何) | 渲染用的几何体 |
| `mesh.refined.0.010000.segs.v2.json` | mesh 过分割(顶点 → segment id) | 建立 face/顶点 → instance 映射的中间件 |
| `semseg.v2.json` | 实例分割(segment 聚成物体,含 objectId / label / OBB) | **核心**:得到每个 instance 由哪些 segment/顶点构成;也是 instance label 的来源 |
| `labels.instances.annotated.v2.ply` | 实例分割可视化(逐顶点实例上色) | **交叉校验**你建的 face→instance 映射 |
| `sequence.zip` | 标定好的 RGB-D 流:逐帧 color、depth、camera pose、内参 | **位姿 + 内参**(渲染用);depth 见下方说明 |

> **本模块不读传感器 depth 做几何**:可见性来自 **GT mesh 渲染**(z-buffer 自带渲染深度),不靠 `depth.pgm` 反投影。depth 帧仅在你想做"渲染深度 vs 传感器深度"交叉校验时才用得上,主流程不依赖它。不要套用 SGF 那种"读 depth 反投影建点云"的做法。

**关系标注在 `3DSSG_subset/`**(不在 scan 文件夹内),见 1.3 节。

**`sequence.zip` 内部结构**(约 117 帧/scan,以实际解压为准):
- `frame-XXXXXX.color.jpg` — 彩色帧(960×540)
- `frame-XXXXXX.depth.pgm` — 深度帧(224×172,16 位,`depthShift=1000` 即毫米;**本模块主流程不用**)
- `frame-XXXXXX.pose.txt` — 4×4 相机位姿(**相机 → 世界**)
- `_info.txt` — 相机内参 K、分辨率等

### 1.3 3DSSG 提供的场景图标注(关系来源)

3DSSG 是建在 3RScan 之上的场景图标注。**关系是数据集直接给的现成标注,不需要你从几何/位姿去推断关系内容。** 它在磁盘上的位置和组织方式与 scan 内文件不同,确认到的情况如下(仍建议亲自核对一遍):

- 关系标注文件在 **`3DSSG_subset/` 目录**下(不在每个 scan 文件夹内):`relationships.json`(全量,含多个 scan)以及按划分切开的 `relationships_{train,validation,test}.json`。
- 这类文件**一个文件包含多个 scan**,外层按 scan 组织(每个 scan 条目含 `objects`(id→label)、`relationships`、`split`)。读取某个 scan 时**按 `scan_id` 检索对应条目**,不要去 scan 目录找 per-scan 关系文件。
- 词表在 `3DSSG_subset/`:`classes.txt`(159 类物体)、`relationships.txt`(26 种谓词)。谓词若以数字 id 出现,用 `relationships.txt` 映射成名称。
- 存在 **train / validation / test 划分**。本阶段只构建 GT、不训练,但请记录每个 scan 属于哪个 split,并从对应 split 的关系文件取数据(或确认全量 `relationships.json` 已覆盖你的样本)。

**关系记录形态**:每条关系是一个**四元组 `[subjectId, objectId, predId, predName]`**,例 `[2, 1, 15, "standing on"]`——即前两位是两个物体的 instance ID,后两位是谓词 id 与谓词名。确认后,"这两个 instance 之间是什么关系"直接查表即可;Phase B 只决定它**何时**进入 GT,不负责判断关系内容。(字段顺序已核对一致,但解析前仍建议打印两条原始记录确认。)

**instance label 来源**:以 scan 内 `semseg.v2.json` 的 segGroup(`objectId/label`)为准;split 关系文件里每个 scan 也带 `objects`(id→label)可作交叉参考。无需依赖独立的 `objects.json`。

⚠️ **ID 链路必须自己核(关键)**:关系文件里写的是 instance ID(如 2、1),你要靠这套 ID 把关系和 mesh 几何串起来:

```
3DSSG_subset/关系文件(按 scan_id 检索)──(subject/object 的 instance id)──┐
                                                                       ├─► 需为同一套 instance id
scan/semseg.v2.json          ──(objectId → segments)───────────────────┘
                                  └──(segments → 顶点/面)──► mesh 上的几何(即 Phase 0 的 face_to_instance)
```

一般情况下关系文件与 `semseg.v2.json` 用**同一套 instance 编号**,直接能对上;但**不保证**——版本间字段命名/编码可能不同,或某些 instance 在关系文件里出现却在实例分割里找不到。因此**不能默认对齐,必须显式校验**(见 Phase 0 第 6 步的前置自检)。

### 1.4 单位与坐标的坑(务必先处理对,否则渲染全错)

- **位姿与 mesh 必须同坐标系、同单位。** mesh 顶点与 `pose.txt` 应统一到同一坐标系下渲染;若涉及单位换算(深度毫米 vs 位姿/网格的单位),三者口径要一致,`TAU_DEPTH` 等阈值跟随所选单位。
- `pose.txt` 给的是 **相机 → 世界** 的变换 `P_t`。渲染时需要世界 → 相机,用 `P_t⁻¹`。
- **每个 scan 都用它自己目录下的 mesh + 自己的逐帧 pose,在该 scan 自身的原生坐标系内渲染。** 样本里可能混有 **rescan**(重访扫描,如属于 validation 的那个)。**rescan 可以正常使用——把它当成一条独立扫描即可**;但**绝不要对它施加 `3RScan.json` 里的 rescan→reference 对齐变换**。那个对齐是给"跨扫描融合"用的;本任务在单序列内构建 GT,一旦把 rescan 位姿对齐到 reference 坐标系、而 mesh 仍是 rescan 自身坐标系,**位姿与 mesh 就会系统性错位,渲染全错且不报错**。简言之:**本模块完全不读 `3RScan.json` 的对齐变换。**
- **3RScan 的帧是稀疏采样的**(相邻帧间隔较大)。不要假设可见性在相邻帧间平滑变化;一帧内可能同时新增多个可见实体。

---

## 2. 算法:三步构建 `{G*_{≤t}}`

整体分三步:**Phase 0(预处理)→ Phase A(逐帧证据更新 + commit)→ Phase B(关系边激活)**。

### Phase 0 — 预处理(每个 scan 跑一次,与帧无关)

1. 加载 `mesh.refined.v2.obj`,得到顶点和面(faces)。
2. 用 `semseg.v2.json` + `mesh.refined.0.010000.segs.v2.json` 建立 **`face_to_instance[]`**:每个三角面属于哪个 instance ID。(semseg 给 instance→segments,segs 给 segment→顶点;由顶点的 instance 推面的 instance,若一个面三个顶点 instance 不一致,取多数;记录这类不一致面的数量。)
3. **交叉校验**:用 `labels.instances.annotated.v2.ply` 的逐顶点实例标注核对 `face_to_instance`,打印不一致比例。
4. **预计算面积(重要)**:
   - `face_area[f]` — 每个三角面的面积。
   - `total_area(i)` — instance i 所有面的面积和(后续算可见比例的分母)。
   - 同时仍保留 `total_faces(i)` 仅供调试参考,但**主逻辑一律用面积,不用面数**。
5. **加载该 scan 的关系**:关系文件在 **`3DSSG_subset/`**(见 1.2 / 1.3),按 split 划分、一个文件含多个 scan。先确认用哪个文件(全量 `relationships.json`,或该 scan 所属 split 的 `relationships_<split>.json`),再**按 `scan_id` 检索**出该 scan 的关系列表。存成**有向三元组列表** `[(subjectId, predicate, objectId), ...]`,**保留方向、保留同一对实例间的多条边**(见 Phase B)。谓词若是数字 id,用 `3DSSG_subset/relationships.txt` 映射成名称。
6. **ID 链路前置自检(必须在写主逻辑前先跑通)**。这一步是整条管线能否对齐的前提,**请实现成一个独立的、最先运行的检查,通过后再继续**。下面是检查要点,**具体字段名/文件名以你实际数据为准,不要照搬假设**:
   - **(a) 字段与来源探查**:打印关系文件外层结构和前几条原始记录,确认外层如何按 scan 组织。字段顺序已知为 `[subjectId, objectId, predId, predName]`,但**仍打印两条核对**以防版本差异。确认 instance label 来源(`semseg.v2.json` 的 `objectId/label`;split 文件的 `objects` 可交叉参考)。
   - **(b) 全量映射检查**:取该 scan 所有关系的 subject/object endpoint ID,逐个到 `semseg.v2.json` 的 instance 集合(即 `face_to_instance` 覆盖的 ID 集合)里查找。统计 `relations_total` 与 `relations_with_missing_endpoint`,打印缺失比例。**找不到的 endpoint 记日志**(scan_id、关系记录、缺失的 ID),不要静默丢弃。
   - **(c) 人工可读抽查**:随机挑 5~10 条关系,把两个 endpoint ID 翻成 label(来源用你在 (a) 确认的权威源),打印成可读句子(如 `"cup (id=14) supported by table (id=6)"`),人眼确认 label 与谓词讲得通、ID 对应到的是合理物体。**这是确认"关系里的 ID ↔ mesh 上的 instance"真正对齐的最直接证据。**
   - 若缺失比例异常高(例如 >5%),先停下排查 ID 编码/字段/split 选择问题,不要带着错位的映射往下跑。

   通过后,关系里的 ID 即可与 `face_to_instance` 互通:关系 `(14, 6, "supported by")` 就对应"面集合属于 14 的物体被属于 6 的物体支撑"。

### Phase A — 逐帧证据更新 + commit

**时间索引 t 的定义(整条管线的时间语义基础,务必先按此处理):**
- 帧的时间先后**以 `sequence/` 内的帧编号(`frame-000000`、`frame-000001`…)升序为准**——该编号即采集的时间顺序。
- **跳过无效帧**:位姿缺失/无效、或无法渲染的帧不参与 GT 构建,也不占用 t。
- 对**实际处理的有效帧重新编连续索引** `t = 0, 1, 2, …`(不要直接用原始帧编号当 t)。原因:帧编号可能不连续(缺帧或抽帧),而下游(奖励、oracle)需要的是连续无洞的步序;用编号数字会让"相邻帧"语义出错。
- 排序依据是帧编号(保证时序正确),但 t 是排序+过滤后的连续序号。所有 `commit_time` / `activation_time` 都记这个**连续 t**。
- 额外维护一个 **`t → 原始 frame_id`** 的映射并写进输出,方便事后把某个 t 对应回具体哪一帧 / 哪张 RGB 做可视化核对。

按上述 t 的顺序遍历每一帧。维护跨帧状态:
- `seen_faces[i]`:instance i **累计被看到过的 face 集合**(set / bitset,**并集去重**)。
- `frame_count[i]`:i 通过 instance 级过滤、被算作可见的**帧数**(仅供持久性分支与调试)。
- `commit_time[i]` / `commit_meta[i]`:i 被 commit 的 t 及其调试元信息(初始未 commit)。

每帧执行:

```
# 1. 渲染(带 z-buffer 做遮挡),输出 face-ID buffer
face_id_buffer = render(mesh, faces, P_t⁻¹, K)        # 每像素最近面的 face ID(如 PyTorch3D pix_to_face)
# instance 层派生:instance_id = face_to_instance[face_id]

# 2. 统计像素
pixel_count_per_face      = count_pixels_per_face(face_id_buffer)
pixel_count_per_instance  = aggregate_to_instance(pixel_count_per_face, face_to_instance)

# 3. 逐 instance 更新证据
for i in instances_touched_this_frame:

    # ---- 碎片过滤:三层分工,替代单一绝对像素门槛 ----
    # tier1: instance 级绝对地板(极低,只挡 1~几十像素的采样噪声/反走样)
    if pixel_count_per_instance[i] < TAU_INST_PIX_MIN:
        continue
    # tier2: instance 级相对遮挡比(尺度无关)——可见像素 / 该 instance 单独渲染的完整投影像素
    full_proj_pixels = render_instance_alone(i, P_t⁻¹, K).count_hit_pixels()
    if full_proj_pixels == 0 or pixel_count_per_instance[i] / full_proj_pixels < TAU_INST_VIS_RATIO:
        continue                                       # 只露一点边角(被前景大面积遮挡),不更新证据

    # tier3: face 级小地板(防"一个大面只露 1 像素却按整面面积累计")
    visible_faces_i = { f in faces_of(i) : pixel_count_per_face[f] >= TAU_FACE_PIX }   # TAU_FACE_PIX 取小(如 1~2)

    # ---- 用面积算可见比例 ----
    single_area_ratio = sum(face_area[f] for f in visible_faces_i) / total_area(i)     # 本帧强证据判据

    seen_faces[i] |= visible_faces_i                   # 并集更新(自动去重)
    cumulative_area_ratio = sum(face_area[f] for f in seen_faces[i]) / total_area(i)    # 累计证据

    if |visible_faces_i| > 0:
        frame_count[i] += 1

    # 4. commit 判定(单调,永不撤销)
    if commit_time[i] is None:
        commit = False
        if single_area_ratio >= TAU_STRONG:                          # (a) 强证据:一帧看清,立即加
            commit, reason = True, "strong_single"
        elif cumulative_area_ratio >= TAU_COMMIT:                    # (b) 累积证据够
            commit, reason = True, "cumulative"
        elif ENABLE_PERSIST and frame_count[i] >= K \
             and cumulative_area_ratio >= TAU_PERSIST:               # (c) 持久 AND 弱证据地板
            commit, reason = True, "persistent"
        if commit:
            commit_time[i] = t
            commit_meta[i] = {
                "reason": reason,
                "single_ratio_at_commit": single_area_ratio,
                "cumulative_ratio_at_commit": cumulative_area_ratio,
                "pixel_count_at_commit": pixel_count_per_instance[i],
                "frame_count_at_commit": frame_count[i],
            }
```

**关键实现要点(容易写错):**

- **可见度量一律用面积比例,不用面数比例。** mesh 三角面大小长尾分布,面数会被三角化疏密支配。分母用 `total_area(i)`,分子用可见 face 的面积和。
- **累计可见比例必须用 face 集合并集 `seen_faces[i]`**,不能把每帧的 `single_area_ratio` 相加——否则同一块表面被多帧重复观测会被重复计数,`cumulative_area_ratio` 会虚高甚至 > 1。并集天然去重,也正好对应"换角度看到新部分才算新证据"。
- **碎片过滤分三层独立阈值(原单一 `TAU_INST_PIX` 绝对门槛会系统性漏掉真实小物体,已废弃):**
  - `TAU_INST_PIX_MIN`(instance 级,极低):整个 instance 当帧总像素少到不可靠(1~几十像素的采样噪声)→ 整体跳过。**只挡噪声,不要像旧的 400 那样挡真实小物体**(如小/远 lamp 每帧仅上百像素但确属可见)。
  - `TAU_INST_VIS_RATIO`(instance 级,**遮挡归一化的 2D 可见比例**,尺度无关):`pix_vis / pix_full`,其中 `pix_vis` = 整场景渲染中该 instance 的可见像素,`pix_full` = **同一相机(同 K / 外参 / 分辨率 / near-far / face rasterization-or-raycasting 规则)下、只渲染该 instance 的 faces** 得到的完整投影像素。**`pix_full` 不要用 bbox 或 2D 面积近似**——必须走与主渲染完全一致的 face 渲染路径(只是三角形子集不同),否则 vis_ratio 会引入另一个不一致来源;实现上在 Phase 0 预建每实例一个 RaycastingScene,每帧只对被触发的实例做第二次 raycast。语义上:**它不是投影面积阈值,而是遮挡归一化后的可见比例**——不惩罚"完整但投影很小"的物体(大平面被掠射角侧看、且未被遮挡时 `pix_vis≈pix_full`、ratio≈1,不应被本层过滤;"投影本身极小是否可靠"交给 `TAU_INST_PIX_MIN`);它惩罚的是"`pix_full` 本该不小、但 `pix_vis` 只占极小部分的 sliver observation"(被前景大面积遮挡、只露边角)。这正是唯一能区分"小物体全可见"与"大物体露边角"的判据——降 `TAU_INST_PIX_MIN` 不足以替代,因为两者在像素维度不可区分。
  - `TAU_FACE_PIX`(face 级,取小):**不要用 `pixel_count[f] > 0`**。`seen_faces` 按"整面面积"累加,若一个大面只露一个角也算整面可见,`cumulative_area_ratio` 会系统性虚高、commit 偏早。给一个小地板(1~2 像素)挡"露一角算整面"的单面泄漏。
  - 三层职责正交:tier1 防极少像素噪声 / tier2 防边角泄漏(累积层)/ tier3 防单面面积虚高。tier2 阈值取低不取高(如 0.1~0.2),只挡极端 sliver,中等遮挡交还给 `single_area_ratio`/commit,避免 over-filter 拖慢大物体累积。
- **`frame_count >= K` 不是独立触发。** 它必须与"弱证据地板 `TAU_PERSIST`"合取:只有**既反复出现、又确实积累了非碎片量级的可见表面**,才走持久性 commit。这样挡住"每帧只露一丁点、刷帧数"的碎片,同时救回"中等物体总被部分看到、`TAU_STRONG` 够不着、`TAU_COMMIT` 爬得慢"的合理实体(否则会漏报,伤下游 recall)。`ENABLE_PERSIST` 默认可设为 false 先跑纯证据版,看统计后再决定是否开启。
- **单调性**:`commit_time[i]` 一旦赋值,永不更改、永不移除。物体离开视野不代表从图里删掉。
- 渲染请直接输出 **face-ID buffer**,instance 层查 `face_to_instance` 派生即可。

### Phase B — 边激活(从节点 commit 时刻派生)

关系本身不会出现在渲染 buffer 里(渲染不出"椅子 attached to 地板"),所以边的出现时机只能从两端节点**派生**:

```
for (subject_i, predicate, object_j) in relationships:
    if commit_time[i] is not None and commit_time[j] is not None:
        edge_activation_time = max(commit_time[i], commit_time[j])
    else:
        edge_activation_time = None      # 两端未都 commit → 该边始终不进入 GT
```

即:**边的激活时刻 = 两端点 commit 时刻里较晚的那个**。要先认识 i 也认识 j,才谈得上它们之间是什么关系。

**Phase B 注意事项:**
- **ID 一致性(承接 Phase 0 第 6 步)**:激活前确认 subject/object 都能映射到有效 instance。找不到的关系记日志、计入统计,不静默丢。
- **保留方向**:3DSSG 关系有向(`i supported by j` ≠ `j supported by i`)。存有向三元组,**不要无向化去重**。
- **保留多重边**:同一对 `(i, j)` 间可有多个谓词,作为**多条独立边**进图,共享同一激活时刻,不合并。
- **单调**:边一旦激活,永不撤销。
- **仅在单条 sequence 内构建**。3RScan 同一房间有 reference + 多个 rescan,跨扫描场景会变;但**单条 scan 内部场景静态**,单调性成立。每个 scan(含 rescan)各自独立构建,scan 之间不建立任何关系。**不要做跨 rescan / 动态关系,不要使用 `3RScan.json` 的对齐变换**(见 1.4)。

---

## 3. 输出契约(下游会消费)

每个 scan 输出一个**紧凑 JSON**(不为每帧存完整图),并提供物化函数。节点上带调试字段:

```json
{
  "scan_id": "...",
  "num_processed_frames": 1000,
  "t_to_frame": { "0": "frame-000000", "1": "frame-000002", "2": "frame-000005" },
  "config": {
    "TAU_INST_PIX_MIN": 0, "TAU_INST_VIS_RATIO": 0.0, "TAU_FACE_PIX": 0,
    "TAU_STRONG": 0.0, "TAU_COMMIT": 0.0,
    "ENABLE_PERSIST": false, "K": 0, "TAU_PERSIST": 0.0
  },
  "nodes": {
    "<instance_id>": {
      "commit_time": 57,
      "label": "chair",
      "commit_reason": "cumulative",
      "single_ratio_at_commit": 0.22,
      "cumulative_ratio_at_commit": 0.31,
      "pixel_count_at_commit": 1840,
      "frame_count_at_commit": 4
    }
  },
  "edges": [
    { "subject": "<id_i>", "predicate": "supported by", "object": "<id_j>", "activation_time": 88 }
  ],
  "id_check": { "relations_total": 0, "relations_with_missing_endpoint": 0 },
  "debug": {
    "uncommitted_renderable": {
      "<instance_id>": {
        "label": "lamp",
        "max_pix_vis": 119, "max_pix_full": 1210, "max_vis_ratio": 0.098,
        "max_single_area_ratio": 0.088, "max_cumulative_area_ratio": 0.133,
        "valid_observation_frame_count": 0,
        "filtered_by_pix_min_count": 0,
        "filtered_by_vis_ratio_count": 47
      }
    }
  }
}
```

`debug.uncommitted_renderable` 记录每个**有几何但最终未 commit** 的 instance 的逐帧证据极值与被过滤次数,供调参时直接定位原因:`max_pix_full≈0` → 根本没投影到画面;`max_vis_ratio` 低且 `filtered_by_vis_ratio_count` 高 → 只露边角;`max_pix_vis` 低且 `filtered_by_pix_min_count` 高 → 像素太少;三者都不低但 `max_cumulative_area_ratio < TAU_COMMIT` → 面积证据确实不足。

`commit_time` / `activation_time` 均为**连续 t**(见 Phase A 的 t 定义);`t_to_frame` 把每个 t 映射回原始帧编号,供可视化核对。

提供工具函数 **`materialize(t) -> G*_{≤t}`**:给定 t,返回节点集(`commit_time <= t`)和有向边集(`activation_time <= t`)。下游(奖励、oracle)会反复调用它。

可选:额外 dump 每帧可见实例集 `V_t` 与 commit 事件时间线,**仅供调试**。

---

## 4. 验证与统计(扩到更多数据前必须做)

在 2~3 个样本 scan 上**先肉眼确认正确性 + 输出统计**,再谈规模化:

1. **渲染叠加检查**:把 instance-ID buffer(按实例上色)叠到对应 RGB 帧,输出 5~10 帧可视化。确认:遮挡正确(被挡物体不"漏出来")、单位/坐标对齐(渲染轮廓与 RGB 物体对齐)、无 mesh 空洞导致的可见性泄漏。
2. **分布统计(定阈值用)**:dump 直方图——
   - `single_area_ratio` 的分布(决定 `TAU_STRONG`;注意单视角受自遮挡限制,通常远小于 0.6,阈值应按分布高分位来定,**不要默认 0.6**);
   - `cumulative_area_ratio` 的分布(决定 `TAU_COMMIT`);
   - 每帧 instance 总像素分布(决定 `TAU_INST_PIX_MIN`,看 1% 分位)及 `visible_pixels / full_projection_pixels` 比(决定 `TAU_INST_VIS_RATIO`);
   - 各类别的 `commit_time` 分布、各 `commit_reason` 占比。
3. **commit 时间线检查**:打印若干 instance 的 `commit_time` + `commit_reason` + 比例,抽查是否符合直觉(大而近的物体应早 commit;别有大量物体只靠 persistent 勉强 commit)。
4. **子图物化检查**:在几个 t 调 `materialize(t)`,画出节点 + **有向边**,核对边不早于两端节点。
5. **ID 校验报告**:打印 `relations_with_missing_endpoint / relations_total` 比例,确认 3DSSG↔3RScan ID 对齐没有大面积缺失。

把可视化与统计产物保存,便于 review。

---

## 5. 可配置项(旋钮)

全部做成 config。**第一版不要把任何数值当定论——按第 4 节统计结果定值。** 下面只是占位起步值:

| 参数 | 含义 | 起步建议 |
|---|---|---|
| `TAU_INST_PIX_MIN` | instance 当帧总像素下限(极低,只挡 1~几十像素噪声) | ~10–20,看 inst_pix 1% 分位 |
| `TAU_INST_VIS_RATIO` | 可见像素 / 该 instance 单独渲染的完整投影像素(相对遮挡比,尺度无关) | 0.1~0.2,只挡极端 sliver |
| `TAU_FACE_PIX` | face 当帧像素下限(防"露一角算整面"的单面泄漏) | 小值,1~2 |
| `TAU_STRONG` | 单帧可见**面积**比例阈值,超过即立即 commit | **先统计,勿默认 0.6**(单视角受自遮挡,常 <0.5) |
| `TAU_COMMIT` | 累计可见**面积**比例阈值 | ~0.3(后续要做 ablation) |
| `ENABLE_PERSIST` | 是否启用"持久 AND 弱证据"分支 | 默认 false,统计后决定 |
| `K` | 持久性所需出现帧数(仅在 persist 分支) | 2~3 |
| `TAU_PERSIST` | persist 分支的弱证据地板(< `TAU_COMMIT`) | ~0.1 |

---

## 6. 建议的代码结构(便于接入后续阶段)

```
gt_subgraph/
├── data_loader.py      # 读一个 scan:scan 内 mesh/segs/semseg/sequence(pose,K) + 3DSSG_subset 关系文件(按 scan_id 检索)
├── mesh_instance.py    # Phase 0:face_to_instance + 交叉校验 + face_area/total_area + ID 一致性校验
├── renderer.py         # 渲染 face-ID buffer(建议 PyTorch3D pix_to_face;Open3D/pyrender 亦可)
├── gt_builder.py       # Phase A + Phase B:逐帧证据更新、commit、边激活
├── output.py           # 写 JSON + materialize(t)
├── stats.py            # 第 4 节的分布直方图与统计报告
├── visualize.py        # 第 4 节的渲染叠加 / 子图可视化
└── run.py              # 按 scan 循环的入口,路径/参数可配置
```

把 `gt_builder` 与渲染、IO、统计解耦,后续阶段(奖励 Φ、oracle)可直接 import `materialize(t)`,无需重跑渲染。

---

## 7. 明确不要做的事(scope 边界)

- ❌ 不要实现奖励函数 Φ、势函数、对齐打分。
- ❌ 不要实现 oracle 动作集合、SFT 标签生成。
- ❌ 不要实现策略模型、SFT、RL、任何训练逻辑。
- ❌ 不要做节点/边的删除或撤销(本阶段一切单调)。
- ❌ 不要做跨 rescan / 动态关系 / 多序列对齐。
- ❌ 不要用 face 数量比例(用面积比例)。
- ❌ 不要把 face 可见判定写成 `pixel_count > 0`(留 `TAU_FACE_PIX` 地板)。
- ❌ 不要用单一绝对像素门槛做 instance 级过滤(会系统性漏掉真实小物体,如小/远 lamp);用 `TAU_INST_PIX_MIN`+`TAU_INST_VIS_RATIO` 两层替代。
- ❌ 不要把任何阈值当成已确定的正确值(先统计后定)。
- ❌ 不要一上来跑全量数据;先在样本上跑通并通过第 4 节验证。

---

## 8. 后续阶段(给你背景,本次不实现)

本模块产出的 `{G*_{≤t}}` 之后会被用于:
1. **奖励**:势函数 `Φ(belief_G_t, G*_{≤t})` 衡量当前预测图与该帧 GT 子图的对齐度,做 potential-based reward shaping。
2. **oracle / SFT 标签**:相邻帧 GT 子图的差 `G*_{≤t} \ G*_{≤t-1}` 派生"该帧的正确编辑集合";skip 的正确性由 `G*_{≤t} == G*_{≤t-1}` 判定。
3. **策略训练**:SFT(warm-start 动作原语)+ RL(在不确定下学在线最优策略)。

了解这些只是为了让你把接口留干净(尤其 `materialize(t)`)。**本次只交付第 0~4 节的 GT 子图构建。**
