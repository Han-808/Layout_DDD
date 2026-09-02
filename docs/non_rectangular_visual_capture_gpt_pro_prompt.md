# GPT-Pro Prompt: Non-Rectangular Visual Evidence Acquisition Review

请启用你可用的最深层推理模式（Deep Thinking / extended reasoning）。先完整理解现状，再提出建议；不要直接写代码，也不要默认现有相机算法必须推翻。

## 你的角色

你是 3D scene benchmark、computational geometry、camera planning、visual evidence acquisition 和 VLM evaluation 的资深研究工程师。你的任务是审查一个已经完成基础增量兼容层的 benchmark，并回答：面对 non-rectangular multi-room layouts，现有视觉抓取算法的方法本身与调用顺序，哪些可以原样复用，哪些必须演进？

代码分支：

`https://github.com/Han-808/Layout_DDD/tree/codex/nonrect-evaluator-compat`

优先阅读：

- `non_rectangular_evaluation_plan.md`
- `src/benchmark/visual_judge/render_views.py`
- `src/benchmark/visual_judge/orchestration/camera_acquisition.py`
- `src/benchmark/visual_judge/evidence_sufficiency.py`
- `src/benchmark/visual_judge/active_fallback.py`
- `src/benchmark/visual_judge/active_policy.py`
- `src/benchmark/visual_judge/camera_targets.py`
- `src/benchmark/visual_judge/camera_ranking.py`
- `src/benchmark/visual_judge/camera_repair.py`
- `src/benchmark/visual_judge/visual_config.py`
- `src/benchmark/visual_judge/p0b.py`
- `src/benchmark/evaluator/generic_validity/{collision,oob,support}.py`
- `src/benchmark/camera_cal_scene_level/adapters.py`
- `docs/camera_cal_scene_level_runtime.md`

如果无法访问仓库，请明确列出需要我粘贴或上传的文件，不要臆测实现。

## 已冻结的 benchmark scope

这是增量功能。现有 rectangular evaluator workflow 必须完全保持原样，新行为只能由显式 `evaluation_mode="non_rectangular_multi_room"` 路由进入。

输入层级是：

```text
Scene → Rooms → Objects
```

每个完整 multi-room generation 会被拆成 authoritative per-room evaluation units，再逐 room 复用现有 evaluator。当前不评估显式 cross-room semantic relations、doors、windows、ceilings、point cloud、BBox annotations 或 navigation。所有对象与 room polygons 使用同一个 global XY、Z-up coordinate frame，不 recenter。

几何仅包含：

- concave/convex floor polygons；
- room-local wall segments；
- floor `z=0`；
- generated object poses/sizes/meshes。

Walls 属于 architecture，不作为 Collision objects。当前 selected cohort 有 10 scenes、50 rooms、283 wall segments；scene adjacency 连通，但没有 exact-coincident shared walls，near/partial shared-wall pairing 尚未实现。

Metrics：

- L1：Collision、OOB、Support；
- L3：Scale、Style、Object Pairing、Functional、Placement；
- Functional 必须先于 Placement；
- 所有 metric 当前均以 room 为 evaluation unit；
- 不新增 cross-room judge。

## 当前视觉抓取范式（请用代码核实）

当前大体设计是：

1. deterministic detector / metric logic 先产生 event、target IDs、geometry hints；
2. deterministic camera candidate bank；
3. technical feasibility 与 candidate pruning；
4. preview render；
5. target visibility / collision mask / focus measurements；
6. metric-specific deterministic ranking/selection；
7. final evidence render 与固定角色组装；
8. deterministic evidence-sufficiency gate；
9. 仅当不足是 measured、camera-repairable 时，进入 bounded active fallback；
10. VLM 只能在确定性 proposal bank 中选择，不能自由生成 pose，也不能输出 metric verdict；
11. 执行 dolly/orbit/elevate/lower 等 bounded repair，重新 preview、测量并评估 sufficiency；
12. 最终 evidence packet 交给 metric judge。

现有候选/选择包含 `visibility_ranked`、`support_contact_plane`、`query_cov`、BBox track、collision candidate ranking、global-context candidates 等路径。P0b 的 local RGB/highlight/contour/global-context roles 与 evidence budgets 已经版本化。Active fallback 可能处于 shadow mode，且 `unknown` evidence 不应自动触发 camera search。

请先根据实际代码修正这段描述，并给出准确 call graph。

## 当前已接受的 metric workflow 原则

### Collision

- deterministic broad phase 能确定 no collision 时直接 valid；
- 其余 ambiguous event 交给 VLM；
- walls 不作为 collision objects；
- 核心 verdict workflow 暂不改变。

问题仅在于：non-rect room 中的 candidate generation、room feasibility、遮挡处理、pair-focus 视角和调用顺序是否仍可靠。

### OOB

- 总体 deterministic → ambiguous VLM workflow 可保留；
- polygon containment / polygon difference、非轴对齐 wall plane、penetration 快筛必须 polygon-aware；
- 视觉证据可能需要 interior normal、wall-tangent、boundary-focus 与 room-context views。

### Support

- 总体处理方式与 OOB 类似；
- 只考虑 floor、wall、object support，不考虑 ceiling；
- polygon boundary distance、wall attachment/contact 与 zero-view fallback 需要审查。

### Functional / Placement

- Functional 先于 Placement；
- 两者按 room planned-object count 加权；
- room-purpose mapping 已由 generation artifact 给出；
- 需要审查 concave room 的 global context、object-group coverage、局部 placement evidence 与 camera budget，而不是重新设计 scoring。

Scale、Style、Object Pairing 也逐 room 运行；请判断它们是否只需继承改进后的 context/local view bank，还是需要独立的 polygon-aware acquisition policy。

## 需要你回答的核心问题

请逐 metric 分析 Collision、OOB、Support、Functional、Placement、Scale、Style、Object Pairing，并对每个 metric 明确给出：

1. 当前 detector/event 是什么，何时需要视觉证据？
2. 当前 camera targets、candidate families、ranking signals、evidence roles、sufficiency conditions 是什么？
3. 当前调用顺序是什么？顺序是否会产生错误、信息泄漏、重复 render 或不必要成本？
4. concave polygon、狭长 room arm、非轴对齐 wall、局部遮挡、room-wide framing、global coordinates 会破坏哪些假设？
5. 分类选择：
   - `unchanged`；
   - `routing/config only`；
   - `new polygon-aware method required`；
   - `call-order change required`。
6. 如果要变，最小安全改动是什么？应放在 evaluator method、camera candidate builder、ranking/sufficiency、workflow routing，还是独立 preprocessing script？
7. 哪些建议是 v1 必需，哪些应延期到 future work？

特别检查这些设计选择：

- room-wide/global view 是否应从 rectangular center/corners 改为 polygon centroid、interior point、medial axis、triangulation 或 visibility decomposition；
- camera location 的 room-interior feasibility 是否必须使用 polygon containment、wall clearance 与 line-of-sight；
- concave polygon 是否需要按 visibility cell / reflex vertex 生成多个 context views；
- OOB 是否应先由 violated edge/plane 的 inward normal 生成 views，再补 tangent/oblique views；
- Support 是否应按 support contact plane / wall attachment plane 生成 views；
- Collision pair views 是否真正与 room shape 无关，还是会被 concave-wall occlusion 破坏；
- global-context view 应在 local views 之前固定，还是在 local sufficiency 后补；
- preview visibility gate、deterministic ranking、VLM proposal selection、final render、final sufficiency 的顺序是否最优；
- 是否存在可以先用 analytic visibility / ray casting / polygon clipping 排除、从而避免 Blender preview 的步骤；
- 是否需要按 metric 共用一个 polygon-aware candidate bank，还是各 metric 保持独立 bank；
- 当 camera insufficiency 实际来自 geometry grounding、presentation 或 missing architecture 时，是否应禁止 active camera repair。

## 强约束

- 不改变现有 rectangular workflow、camera behavior 或 score semantics。
- 不改变 metrics、weights、thresholds、aggregation 或 benchmark hierarchy。
- 不引入 cross-room semantic metric。
- 不把 VLM 变成 free-form camera generator。
- 不让 camera selector 输出 verdict 或 score。
- 不用 camera movement 掩盖 geometry/evidence/rendering failure。
- 不默认增加大量 renders；优先 bounded、measurable、cacheable 的方法。
- 所有新行为必须 additive、explicitly routed、fail-closed、hash/provenance 可审计。
- 本轮只做设计审查，不写代码。

## 输出格式

请输出以下内容：

1. **As-is call graph**：准确到关键函数/模块，并指出哪些是 metric-specific。
2. **Assumption audit**：列出当前算法中的 rectangular/axis-aligned/convex/room-center 假设，并标注证据位置。
3. **逐 metric decision table**：当前方法、问题、变化类别、最小方案、成本、风险、v1/future。
4. **推荐的统一调用顺序**：给出清晰伪代码；同时说明是否需要 metric-specific deviations。
5. **最小 v1 proposal**：只包含 benchmark 可运行所必需的演进。
6. **Deferred proposal**：shared-wall pairing、cross-room context 或高级 visibility decomposition 等可延期项。
7. **Ablation/test matrix**：rectangular parity、convex polygon、concave L-shape、狭长/多凹角 room、wall attachment、OOB、support、collision occlusion；包括 correctness、evidence sufficiency、render count、latency 和 VLM-call cost。
8. **Open decisions for human approval**：只列真正会改变科学含义、成本上限或 failure policy 的选择。

对每条结论标注：`code-confirmed`、`design inference` 或 `needs experiment`。如果现有实现已经足够，不要为了新颖而建议修改。
