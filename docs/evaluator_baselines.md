# Floor-plan Evaluator 最新运行基准

更新时间：2026-09-10 14:36（Asia/Shanghai）。依据用户再次明确的规则：**每个 floor-plan mode 最新实际使用的 evaluator 就是基准，包括仍在运行的版本**。登记应反映实际使用，不另设滞后的“正式基准”；历史结果仍保留原身份。

唯一登记源：[floorplan_evaluator_baselines_v1.json](../configs/runners/floorplan_evaluator_baselines_v1.json)。每个 mode 在 `current` 中恰好有一个基准 ID。latest 以可核实的实际使用为准，不按版本号大小、分支时间或工作目录 HEAD 排序。发现新的实际使用后应更新登记；尚未实现启动器自动更新。

## 当前三个基准

| Mode | 唯一当前基准 | 来源运行 | Collision / Support / OOB / L3 / Prompt |
|---|---|---|---|
| `single_room`（Open-space / 独立场景） | `single_room_sceneweaver_20260909_v1` | 9 月 9 日 SceneWeaver floor-frame，S100–S108；整批 15:08 开始，17:16 结束 | v4 / v12 / v2 / v8 / v31 |
| `multi_room`（矩形多房间） | `multi_room_hy4_wall_20260902_v1` | HY4 带墙多房间 r2；最新 case 于 9 月 2 日 10:56 开始、11:11 结束 | v4 / v12 / v2 / v8 / v31 |
| `non_rectangular_multi_room`（复杂 / 非矩形 FloorPlan） | `nonrect_3186983_execution_v4_20260910_v1` | 当前 combined142 hardened v4；commit `31869837105d7ef10c3b3e382cb84eeb1f1f1efc`，执行结果契约 v4 | v3 / v11 / polygon-v2 / v7 / v30 |

FrozenAssets 的不同 generation harness 不各自拥有另一个 single-room latest。今后讨论这些 harness 的“最新评估器基准”时，统一引用本表的 `single_room`。但以前用其他 evaluator 跑出的报告保留原版本，不追溯改标签、不自动重算。

## 固定了什么

- **Single-room**：核验现存冻结源码树 860 个文件，tree SHA256 为 `893b3a5c38751a553f348ac2e12a0142121cf489858abb28a66152f33ed069dd`；登记 curation lock、experiment plan、case/control manifest 的文件哈希，以及实际 scoring/renderer 配置。
- **Multi-room**：固定最近实际运行的 selection manifest、最新 case 的运行清单、L1/L3 报告、camera control 和历史 launcher 文件哈希；登记实际指标及 scoring 配置。**未找到可独立核实的完整历史源码快照，因此 `replay_ready=false`。** 这是有确定来源的参考运行基准，不冒充已经恢复出的可重放源码 release；不得用当前工作目录或 single-room 源码树补成“相同代码”。
- **Nonrect**：当前源码为干净的 `Support/worktrees/collision-final-bundle-v1`，commit `3186983`。固定当前 execution plan SHA256 `7f31eee74c297b0cccdac8a815a72dd028bf8a95d71ff465d95caf6ac8397507`、执行层文件哈希和三个 lane 的 runtime config 哈希；本次核验全部执行层文件与封存计划一致。旧 `7fb8f91` 条目和来源证据保留为历史记录。执行层仍依赖本地冻结 release，不能仅推送核心 commit 就声称可移植重放。

登记文件中的相对 artifact 路径均相对仓库根目录；外部冻结 checkout 使用绝对路径。此文件属于本地 operator/runtime 登记，不是可搬运的 wheel 配置，未复制到 `_resources`。

## 登记不改变实验

本次只新增基准登记和使用约定：**未修改启动器、evaluator 实现、正在运行的参数/并发/输入，没有暂停、重启、重跑、发布或合并结果**。运行器尚未增加自动读取此登记的启动校验，不能把文档约定说成运行时已强制执行。

历史 `fc0b656` 以及其他指标组合仍然有效地描述其历史结果，但不再作为这些模式的 current reference。旧 official binding 文件不改写，以免破坏原运行身份；它们的 `ready` 不是本登记下的 latest 指针。

用户已授权按最新实际使用更新 current；每次仍须核实来源、建立明确身份并保留旧条目。新旧结果只有满足相同代码/配置/证据契约等比较条件后才能做同版汇总；相同指标版本号本身不是代码相同证明。

## 仍保留的问题

1. `3186983` 包含 Collision 最终证据合并加载优化，但当前运行未完成，不能据此宣称性能已验收。
2. 旧 `7fb8f91` 的 mesh 接入限制保留在历史条目；本次登记未改变当前任何算法或取证行为。现有 Placement/取证失败不会因登记而变成有效结果。
3. Single-room 来源运行仍有明确的建筑兼容性限制：native clear height 比 public height 低约 0.254881 m。登记不批准跨 harness 建筑条件可比。
4. Multi-room 历史源码恢复、可移植 release、以及启动器级别的 registry 校验，均未在本次执行。
5. 当前全局 4 个房间槽位不是公平队列，排队上限为 3600 秒；Sol `scene_012173/room_003` 已出现 AdmissionTimeout，仍 pending。GLM `scene_011687/room_000` 已出现与相机/渲染有关的 failed_nonretryable，1 次评估、0 次房间重试；不是有效零分，也尚未确认为可继续的 hard skip。

## 当前运行与集成状态（2026-09-10 14:36）

combined142 仍在运行，完整报告 17/142（15 复用 + 2 本次新完成）；后续 plus20 的 113 个任务尚未启动。上述两项异常仍存在，未停机、未改活动代码、未启动额外评估。这是有时间点的快照，不是最终计数。

尚未推送或合并本次变更。主工作区包含大量不相关未提交文件；当前 Nonrect 源码处于独立 Git 仓库，其 `origin` 指向本地前序源码仓库，不是 GitHub。安全集成必须从目标远端最新 main 建立独立干净集成分支，审阅必要依赖后迁入 evaluator、配套测试与登记；不能对当前工作区执行整体 add/commit，不能覆盖运行中的冻结 checkout。执行层应单独审查本地路径与冻结 release 依赖，不携带凭据、原始 API 交互、资产或生成结果。

合并前应在隔离环境完成相关回归、默认测试和打包验证，审查行为差异与失败分类，检查目标 main 是否前进，再通过 PR/受保护分支流程合并；不强推 main。Single-room 冻结树与 Nonrect 并非相同实现，Multi-room 缺少历史源码，不得用一个分支冒充三种可重放版本。

之前的诊断和版本差异证据见 [2026-09-09 核查报告](/Users/han_mohan/Desktop/Layout_DDD/docs/evaluator_audit_20260909.md)。本登记覆盖其中“尚未建立 current 映射”的状态结论，但不覆盖其性能、评分和来源限制。
