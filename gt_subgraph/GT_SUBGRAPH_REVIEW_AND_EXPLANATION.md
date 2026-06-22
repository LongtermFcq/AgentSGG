# GT 子图构建模块审查与解读

## 1. 一句话概括

这个模块把 3RScan 的几何、实例分割、相机轨迹和 3DSSG 的最终关系标注，转换成一个随有效帧时间单调增长的 GT 场景图子图序列。

## 2. 审查结论

当前实现已经覆盖 handoff 文档要求的主流程：Phase 0 预处理、Phase A 逐帧可见证据累计与节点 commit、Phase B 根据两端节点 commit 时间激活有向关系边、紧凑 JSON 输出以及按时间物化子图。

本次审查发现并修复了几处偏差：

| 问题 | 影响 | 修复位置 | 修复结果 |
|---|---|---|---|
| 混合 instance 的面统计不完整 | 只统计了三个顶点全不同的面，漏掉两个顶点一致、一个顶点不同的边界面 | gt_subgraph/mesh_instance.py | 现在只要三角面三个顶点的 instance 不完全一致，都会计入 mixed-instance face 统计 |
| ID 自检把无几何物体误报成关系 ID 缺失 | annotation-only 物体存在于 semseg，但没有 mesh 面，原逻辑会把相关关系算成 missing endpoint | gt_subgraph/run.py、gt_subgraph/gt_builder.py、gt_subgraph/output.py | 现在 ID 缺失按 semseg objectId 判断；无几何端点单独记录，不再污染 ID 对齐报告 |
| Phase B 无法区分“ID 真缺失”和“ID 存在但未 commit” | 会导致 id_check 统计失真，也会误导后续排查 | gt_subgraph/gt_builder.py | 现在边激活使用 semseg instance 集合作为已知 ID 集合；已知但无几何或未 commit 的端点只会让边不激活 |
| 只支持已解压 sequence 目录 | 如果数据只保留 sequence.zip，帧位姿读取会失败 | gt_subgraph/data_loader.py | 现在内参和 pose 都支持从 sequence.zip 回退读取 |
| 渲染失败帧的连续 t 语义不够稳健 | 如果某帧无法渲染，原实现不方便保持“只对成功处理帧编号”的语义 | gt_subgraph/gt_builder.py | 现在只有成功渲染的帧才占用连续 t；失败帧会被跳过并记录警告 |
| 下游按包导入 builder 时可能失败 | 绝对导入依赖运行目录，不利于后续模块复用 | gt_subgraph/gt_builder.py | 现在同时支持包导入和脚本式运行 |
| 可视化 smoke test 依赖已解压 RGB | 只有 sequence.zip 时无法输出叠加图 | gt_subgraph/smoke_test.py | 现在 RGB 也支持从 sequence.zip 读取 |

## 3. 输入数据长什么样

| 输入来源 | 结构和类型 | 尺寸或规模 | 例子 |
|---|---|---|---|
| mesh.refined.v2.obj | 三维顶点表和三角面表 | 顶点数量通常是几万级，面数量通常也是几万级 | 样本 7272e16c 有 18496 个顶点、26517 个三角面 |
| mesh.refined.0.010000.segs.v2.json | 每个顶点对应一个 segment id | 长度必须等于 OBJ 顶点数 | 第 100 个顶点属于某个 segment |
| semseg.v2.json | 每个 objectId 对应一个 label 和若干 segments | 一个 scan 通常几十个 instance | objectId 为 5 的实例 label 是 floor |
| labels.instances.annotated.v2.ply | 每个顶点带 objectId 的实例标注 | 顶点数量必须等于 OBJ 顶点数 | 用来交叉验证 semseg 和 segIndices 的链路 |
| sequence 或 sequence.zip | 每帧的 RGB、pose、相机内参 | frame 编号升序表示时间顺序；color 常见为 960×540 | frame-000000.pose.txt 是一个 4×4 相机到世界位姿矩阵 |
| 3DSSG_subset/relationships_*.json | 每个 scan 的 objects 和 relationships | 关系是有向四元组列表 | wall(id=1) attached to floor(id=5) |
| 3DSSG_subset/train_scans.txt、validation_scans.txt | scan 到 split 的映射 | 每行一个 scan id | 用来优先选择对应 split 的关系文件 |

## 4. 输出数据长什么样

每个 scan 输出一个紧凑 JSON，不按帧保存完整图，而是保存节点 commit 时间和边 activation 时间。

| 输出字段 | 类型和结构 | 含义 |
|---|---|---|
| scan_id | 字符串 | 当前 scan 的唯一 ID |
| num_processed_frames | 整数 | 实际成功处理并参与连续 t 编号的帧数 |
| t_to_frame | 字典 | 连续 t 到原始 frame id 的映射 |
| config | 字典 | 本次构建使用的阈值和开关 |
| nodes | 字典 | key 是 instance id，value 是节点 label、commit_time、commit 原因和证据统计 |
| edges | 列表 | 每条关系保留 subject、predicate、object、activation_time |
| id_check | 字典 | 关系端点 ID 对齐检查和无几何端点记录 |
| debug.uncommitted_renderable | 字典 | 有几何但最终未 commit 的实例诊断信息 |

样本 7272e16c 的一次输出结果如下：117 个有效处理帧，20 个可渲染节点全部 commit，11 条关系边全部激活，ID 缺失关系数为 0。

## 5. 一个具体样例如何流转

以 scan 7272e16c-a01b-20f6-8961-a0927b4a7629 为例：

| 阶段 | 输入片段 | 处理动作 | 产生的中间结果 |
|---|---|---|---|
| Phase 0：几何对齐 | OBJ 顶点和三角面、segIndices、semseg | 把“顶点属于哪个 segment”接到“segment 属于哪个 objectId” | 每个三角面都有一个 instance id |
| Phase 0：面积统计 | 每个三角面和它的 instance id | 计算每个三角面的面积，再按 instance 汇总 | 每个物体有自己的总表面积 |
| Phase 0：关系读取 | relationships_validation.json | 找到当前 scan 的关系列表 | 得到 11 条有向关系 |
| Phase 0：ID 自检 | 关系端点 id 和 semseg objectId | 检查每个关系端点是否能在 semseg 中找到 | 该样本 11 条关系中没有缺失端点 |
| Phase A：逐帧渲染 | frame-000000 的 pose、相机内参、mesh | 用相机视角渲染出每个像素看到的最近三角面 | 得到 face-ID buffer |
| Phase A：像素聚合 | face-ID buffer 和 face 到 instance 的映射 | 把像素命中从 face 汇总到 instance | 得到每个当帧可见物体的像素数 |
| Phase A：碎片过滤 | 当帧像素数、完整投影像素数、face 像素数 | 过滤极小噪声、严重遮挡边角和单面泄漏 | 留下可信的可见 face 集合 |
| Phase A：证据累计 | 当前可见 face 集合和历史 seen face 集合 | 用并集累计，不重复计算同一表面 | 得到单帧面积比例和累计面积比例 |
| Phase A：节点 commit | 面积比例、出现帧数、阈值配置 | 达到强证据、累计证据或持久弱证据条件 | 记录节点 commit_time 和 commit_reason |
| Phase B：边激活 | 3DSSG 关系和两端节点 commit_time | 只有两个端点都 commit 后，关系才进入图 | 边的 activation_time 是两端 commit_time 中较晚的一个 |
| 输出 | 节点时间、边时间、调试统计 | 写成紧凑 JSON | 下游通过 materialize 得到任意 t 的 GT 子图 |

直观比喻：这套流程像给房间里的每个物体盖“已看见”印章。相机每经过一帧，就像手电筒照亮一部分表面；同一块表面反复照到只算一次，换角度看到新的表面才增加证据。物体证据足够后盖章，两个物体都盖章后，它们之间已有的 3DSSG 关系才被放进当前 GT 图。

## 6. 数据流转路径

源数据 → gt_subgraph/run.py.run_scan [按 scan 组织一次完整构建]

run_scan → gt_subgraph/data_loader.py.load_mesh [读取 OBJ 顶点和三角面]

load_mesh → gt_subgraph/data_loader.py.load_segs [读取每个顶点的 segment id]

load_segs → gt_subgraph/data_loader.py.load_semseg [读取 objectId、label 和 instance 到 segments 的关系]

load_semseg → gt_subgraph/data_loader.py.load_ply_objectid [读取 PLY 中的逐顶点 objectId，用于交叉验证]

load_ply_objectid → gt_subgraph/data_loader.py.validate_invariants [确认 OBJ、segIndices、PLY 的顶点数量和面索引一致]

validate_invariants → gt_subgraph/mesh_instance.py.build_face_to_instance [把每个三角面映射到一个 instance id]

build_face_to_instance → gt_subgraph/mesh_instance.py.compute_face_areas [计算每个三角面的真实面积]

compute_face_areas → gt_subgraph/mesh_instance.py.total_area_per_instance [汇总每个 instance 的总面积]

total_area_per_instance → gt_subgraph/mesh_instance.py.crosscheck_with_ply [检查 semseg 链路和 PLY instance 标注是否一致]

crosscheck_with_ply → gt_subgraph/mesh_instance.py.instance_diagnostics [区分有几何、annotation-only 和空实例]

instance_diagnostics → gt_subgraph/data_loader.py.load_relationships [从 3DSSG_subset 中读取当前 scan 的有向关系列表]

load_relationships → gt_subgraph/run.py.id_check [确认关系端点能在 semseg objectId 集合中找到，并输出人工可读抽查]

id_check → gt_subgraph/data_loader.py.load_intrinsics [读取颜色相机内参和分辨率]

load_intrinsics → gt_subgraph/data_loader.py.iter_frames [按原始 frame 编号升序读取有效 pose]

iter_frames → gt_subgraph/gt_builder.py.build_scan [逐帧渲染、过滤碎片、累计面积证据、commit 节点]

build_scan → gt_subgraph/renderer.py.FaceRenderer.render_face_ids [渲染整场景 face-ID buffer，负责遮挡关系]

render_face_ids → gt_subgraph/renderer.py.InstanceRenderer.pix_full [只渲染单个 instance，得到完整投影像素数]

pix_full → gt_subgraph/gt_builder.py.activate_edges [根据两端节点 commit_time 激活有向多重边]

activate_edges → gt_subgraph/output.py.build_output [组装紧凑 JSON 输出]

build_output → gt_subgraph/output.py.save [保存每个 scan 的 GT 子图时间序列]

保存后的 JSON → gt_subgraph/output.py.materialize [按给定 t 返回截至该时刻的节点集和有向边集]

## 7. 已执行验证

| 验证项 | 结果 |
|---|---|
| Python 语法编译 | gt_subgraph 下所有 Python 文件编译通过 |
| 包导入验证 | gt_subgraph.gt_builder 和 gt_subgraph.output 可作为包导入 |
| ID 缺失统计复核 | 7272e161 样本的 semseg 关系端点缺失数为 0，避免了旧逻辑把无几何端点误报为缺失 |
| 完整样本运行 | 7272e16c 样本构建成功，117 个有效帧、20 个可渲染节点、11 条关系边 |
| 误报样本复跑 | 7272e161 样本构建成功，151 个有效帧、28 个可渲染节点、251 条关系中真实缺失端点为 0 |
| materialize 约束 | 在两个样本的多个 t 上检查，没有出现边早于端点节点的情况 |

## 8. 仍需注意的事项

阈值配置仍然是数据驱动调参项，当前默认值只是可运行配置，不应视为最终实验结论。扩展到更多 scan 前，仍应结合 stats.py 输出的面积比例、像素数和遮挡比例分布重新定阈值。

本模块仍严格保持 handoff 的边界：只构建确定性的 GT 子图序列，不实现奖励、oracle、动作集合、策略模型或训练逻辑。
