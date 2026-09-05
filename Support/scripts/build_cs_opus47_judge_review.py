#!/usr/bin/env python3
"""Build a local annotation-style review of Opus 4.7 VLM judge inputs.

The review is reconstructed from the completed Counter-Strike evaluation
report.  Blue-bordered images are only the images that were actually sent to
the VLM.  Scene-level L4 cards also include the neutral occupancy/spawn diagram
that accompanied the two original-runtime global views.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "Support/artifacts/outputs/cs_benchmark_v1_gpt56_20260728_r1"
)
DEFAULT_REPORT = (
    DEFAULT_RUN_ROOT
    / "cases/claude_opus_4_7/counter_strike_evaluation_report.json"
)
DEFAULT_OUT = DEFAULT_RUN_ROOT / "opus47_judge_review"


SCENE_METRIC_QUESTIONS = {
    "style_consistency": (
        "Do the supplied global runtime views show any significant visible "
        "style incompatibility in this static Counter-Strike-like arena?"
    ),
    "zone_clarity": (
        "Are at least four of the five required spatial roles structurally "
        "clear: team A spawn, team B spawn, preparation/transition space, "
        "main engagement region, and flank region?"
    ),
    "landmark_legibility": (
        "Are at least three spatially distinct static visual cues easy to name "
        "and reuse as location or engagement callouts?"
    ),
    "cover_diversity": (
        "Are at least four meaningfully different cover configurations visible, "
        "covering at least two height profiles, two width profiles, and two "
        "arrangement types?"
    ),
}

SCENE_METRIC_RUBRICS = {
    "style_consistency": (
        "Judge only significant visible style incompatibility among scene "
        "objects and against supplied frozen visual-style directives. Minor "
        "variation and subjective preference remain valid."
    ),
    "zone_clarity": (
        "A role is clear only when specific geometry delimits it, such as "
        "enclosure, a threshold or doorway, backing walls, a height change, "
        "or visibly convergent approaches. A merely plausible inferred role "
        "is weak. Valid requires at least 4/5 clear roles."
    ),
    "landmark_legibility": (
        "A landmark must be visible in the original-runtime views, have a "
        "concise appearance-based name, occupy a distinct spatial region, and "
        "be distinguishable from repeated generic boxes or walls."
    ),
    "cover_diversity": (
        "Differences must be visibly supportable in height/profile, effective "
        "width/profile, shape, or spatial arrangement. Repeated generic boxes "
        "and mesh fragments of one structure do not create extra forms."
    ),
}

REQUIRED_FACTS = {
    "style_consistency": [
        "the overall arena appearance in both global perspectives",
        "material, color, geometric abstraction, and visible outliers",
        "only significant incompatibility, not subjective preference",
    ],
    "zone_clarity": [
        "both declared spawn regions",
        "preparation/transition space leaving each spawn",
        "main engagement region",
        "one or more flank regions",
        "visible delimiting geometry for every role marked clear",
    ],
    "landmark_legibility": [
        "at least three visible and spatially distinct cues",
        "each cue has a concise appearance-based name",
        "cues are not repeated generic boxes, walls, HUD, or diagram-only marks",
    ],
    "cover_diversity": [
        "at least four distinct visible cover configurations",
        "at least two height profiles",
        "at least two width profiles",
        "at least two arrangement types",
        "repeated instances are not counted as new forms",
    ],
    "collision": [
        "both target objects and their local relation",
        "visible surface penetration rather than ordinary contact or assembly",
        "detector/mesh measurements are context, not an invalidity prior",
    ],
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _relative_image(path_value: str, out_root: Path) -> str:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return Path(os.path.relpath(path, out_root)).as_posix()


def _image_card(
    path_value: str,
    out_root: Path,
    *,
    index: int,
    role: str,
) -> dict[str, Any]:
    path = Path(path_value)
    return {
        "src": _relative_image(path_value, out_root),
        "name": path.stem,
        "role": role,
        "order": index,
        "scope": "exact_model_input",
    }


def _safe_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _request_summary(request: dict[str, Any]) -> list[str]:
    detector = request.get("detector_evidence")
    if not isinstance(detector, dict):
        return []
    facts: list[str] = []
    mesh = detector.get("mesh")
    if isinstance(mesh, dict):
        facts.append(f"mesh state: {mesh.get('mesh_state', 'unknown')}")
        distance = _safe_number(mesh.get("minimum_surface_distance_m"))
        if distance is not None:
            facts.append(f"minimum surface distance: {distance:.6f} m")
        intersection = mesh.get("surface_intersection")
        if isinstance(intersection, bool):
            facts.append(f"mesh surface intersection: {str(intersection).lower()}")
    diagnostics = detector.get("diagnostics")
    if isinstance(diagnostics, dict):
        xy_overlap = _safe_number(diagnostics.get("xy_overlap_area"))
        z_overlap = _safe_number(diagnostics.get("z_overlap"))
        if xy_overlap is not None:
            facts.append(f"XY overlap area: {xy_overlap:.6f} m²")
        if z_overlap is not None:
            facts.append(f"Z overlap: {z_overlap:.6f} m")
    return facts


def _collision_cards(report: dict[str, Any], out_root: Path) -> list[dict[str, Any]]:
    metric = (
        report["layer_reports"]["l1_physical_plausibility"]["metrics"]["collision"]
    )
    cards: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(metric.get("pairs") or []):
        judge = pair.get("judge_result")
        if not isinstance(judge, dict):
            continue
        request = judge.get("request") or {}
        object_a = str(pair.get("object_a"))
        object_b = str(pair.get("object_b"))
        evidence_paths = request.get("render_evidence") or []
        images = [
            _image_card(
                str(path),
                out_root,
                index=index,
                role=(
                    "same-pose contour / geometry annotation"
                    if "overlay" in Path(str(path)).stem
                    else "original runtime RGB"
                ),
            )
            for index, path in enumerate(evidence_paths, start=1)
        ]
        objects = request.get("objects") or []
        object_lookup = {
            str(item.get("id")): item for item in objects if isinstance(item, dict)
        }
        targets = []
        for object_id in (object_a, object_b):
            item = object_lookup.get(object_id, {})
            size = item.get("size")
            size_text = (
                " × ".join(f"{float(value):.3g}" for value in size)
                if isinstance(size, list) and len(size) == 3
                else "size unavailable"
            )
            targets.append(
                {
                    "id": object_id,
                    "description": f"{item.get('category') or 'mesh'} · {size_text} m",
                }
            )
        judgement = judge.get("judgement") or judge
        cards.append(
            {
                "card_id": f"collision_pair_{pair_index:03d}_{object_a}_{object_b}",
                "metric": "collision",
                "level": "L1",
                "title": f"{object_a} ↔ {object_b}",
                "question": (
                    f"Do {object_a} and {object_b} have actual unintended "
                    "physical surface interpenetration?"
                ),
                "rubric": str(request.get("metric_rubric") or ""),
                "targets": targets,
                "required_visible_facts": REQUIRED_FACTS["collision"],
                "text_context": _request_summary(request),
                "images": images,
                "judgements": [
                    {
                        "repeat": 1,
                        "evidence_status": judgement.get(
                            "evidence_status", "not_recorded"
                        ),
                        "verdict": judgement.get("verdict") or judge.get("verdict"),
                        "score": judgement.get("score", judge.get("score")),
                        "confidence": judgement.get(
                            "confidence", judge.get("confidence")
                        ),
                        "reason": judgement.get("reason") or judge.get("reason"),
                        "findings": [],
                    }
                ],
                "final_status": judge.get("status"),
                "final_verdict": judge.get("verdict"),
                "final_score": judge.get("score"),
                "evidence_phase": "pair-local plus global context",
                "image_count": len(images),
            }
        )
    # Put actual invalid pairs first; they are the fastest collision sanity check.
    cards.sort(
        key=lambda item: (
            item["final_verdict"] != "invalid",
            item["card_id"],
        )
    )
    return cards


def _finding_lines(repeat: dict[str, Any]) -> list[str]:
    for key in ("role_findings", "landmarks", "cover_findings"):
        rows = repeat.get(key)
        if not isinstance(rows, list):
            continue
        output = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = (
                row.get("role")
                or row.get("name")
                or row.get("form_name")
                or "finding"
            )
            state = (
                row.get("status")
                or row.get("height_profile")
                or row.get("arrangement")
                or ""
            )
            detail = (
                row.get("evidence")
                or row.get("visible_cue")
                or row.get("spatial_region")
                or ""
            )
            output.append(f"{name} · {state}: {detail}")
        return output
    return []


def _scene_metric_cards(
    report: dict[str, Any],
    out_root: Path,
    *,
    case_root: Path,
) -> list[dict[str, Any]]:
    metric_vector = report["metric_vector"]
    global_paths = [
        case_root / "renders/global_global_oblique_00.png",
        case_root / "renders/global_global_oblique_01.png",
    ]
    neutral_diagram = case_root / "counter_strike_l4/judge_observation_diagram.png"
    cards: list[dict[str, Any]] = []
    for metric in (
        "style_consistency",
        "zone_clarity",
        "landmark_legibility",
        "cover_diversity",
    ):
        value = metric_vector[metric]
        if metric == "style_consistency":
            judgement = value.get("judgement") or {}
            evidence_paths = judgement.get("images_used") or value.get("evidence_paths") or []
            repeats = [judgement]
            final_score = value.get("score")
            final_verdict = judgement.get("verdict")
            final_status = value.get("status")
            phase = str(value.get("route") or "global")
        else:
            perceptual = (
                value.get("perceptual_component")
                if isinstance(value.get("perceptual_component"), dict)
                else value
            )
            repeats = perceptual.get("repeats") or []
            evidence_paths = [str(path) for path in global_paths] + [
                str(neutral_diagram)
            ]
            final_score = perceptual.get("score")
            final_verdict = perceptual.get("verdict")
            final_status = perceptual.get("status")
            phase = str(perceptual.get("evidence_phase") or "global")
        images = [
            _image_card(
                str(path),
                out_root,
                index=index,
                role=(
                    "neutral occupancy + declared spawn aid"
                    if Path(str(path)).name == neutral_diagram.name
                    else "original runtime RGB"
                ),
            )
            for index, path in enumerate(evidence_paths, start=1)
        ]
        normalized_repeats = []
        for repeat_index, repeat in enumerate(repeats, start=1):
            normalized_repeats.append(
                {
                    "repeat": repeat_index,
                    "evidence_status": repeat.get("evidence_status"),
                    "verdict": repeat.get("verdict"),
                    "score": repeat.get("score", final_score),
                    "confidence": repeat.get("confidence"),
                    "reason": repeat.get("reason"),
                    "findings": _finding_lines(repeat),
                }
            )
        text_context = []
        if metric in {"zone_clarity", "landmark_legibility", "cover_diversity"}:
            text_context.extend(
                [
                    "neutral occupancy grid and declared A/B spawn coordinates",
                    "room boundary and walkable/topology summary",
                    "no deterministic score, verdict, inferred role, or route label",
                ]
            )
        elif metric == "style_consistency":
            text_context.extend(
                [
                    "static Counter-Strike-like arena scope",
                    "frozen visual-style specification",
                    "no prompt-authorized style deviations",
                ]
            )
        cards.append(
            {
                "card_id": f"scene_metric_{metric}",
                "metric": metric,
                "level": "L3" if metric == "style_consistency" else "L4",
                "title": metric.replace("_", " ").title(),
                "question": SCENE_METRIC_QUESTIONS[metric],
                "rubric": SCENE_METRIC_RUBRICS[metric],
                "targets": [
                    {
                        "id": "whole_static_arena",
                        "description": "Claude Opus 4.7 generated CS scene",
                    }
                ],
                "required_visible_facts": REQUIRED_FACTS[metric],
                "text_context": text_context,
                "images": images,
                "judgements": normalized_repeats,
                "final_status": final_status,
                "final_verdict": final_verdict,
                "final_score": final_score,
                "evidence_phase": phase,
                "image_count": len(images),
            }
        )
    return cards


def _html(cards: list[dict[str, Any]], report: dict[str, Any]) -> str:
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    metrics = [
        "style_consistency",
        "zone_clarity",
        "landmark_legibility",
        "cover_diversity",
        "collision",
    ]
    options = "".join(
        f'<option value="{html.escape(metric)}">'
        f"{html.escape(metric.replace('_', ' '))}</option>"
        for metric in metrics
    )
    summary = {
        "benchmark_score": report.get("benchmark_score"),
        "evaluation_status": report.get("evaluation_status"),
        "card_count": len(cards),
        "collision_card_count": sum(card["metric"] == "collision" for card in cards),
    }
    summary_json = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opus 4.7 · VLM judge evidence review</title>
<style>
:root{{--bg:#111418;--panel:#1b2026;--muted:#9ba7b4;--line:#38424d;--accent:#62a9ff;--ok:#51c878;--bad:#ff6b6b;--amb:#e8bb55;--model:#ae8cff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:#eef2f6;font:15px/1.45 system-ui,-apple-system,sans-serif}}
button,select,input,textarea{{font:inherit}} .top{{position:sticky;top:0;z-index:10;background:#15191ef2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}}
button,select{{background:#242b33;color:#eef2f6;border:1px solid #4b5865;border-radius:7px;padding:7px 10px}} button:hover{{border-color:var(--accent)}} .grow{{flex:1}} .progress{{color:var(--muted);min-width:210px;text-align:right}}
main{{max-width:1700px;margin:auto;padding:18px}} .identity{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}} h1{{font-size:25px;margin:0}} .pill{{background:#26384c;color:#b9dcff;border:1px solid #41668d;padding:3px 8px;border-radius:999px}} .pill.valid{{background:#174a2c;color:#9beab3;border-color:#397b51}} .pill.invalid{{background:#551f25;color:#ffabb3;border-color:#8b454d}} .pill.ambiguous,.pill.unresolved{{background:#514019;color:#ffe1a1;border-color:#8e7130}}
.notice{{color:#d7c58c;background:#2c281b;border:1px solid #62562c;border-radius:8px;padding:10px 12px;margin:12px 0}} .summary{{color:#aeb8c3;margin-left:auto}}
.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(390px,.95fr);gap:16px}} .gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;align-content:start}}
.view{{background:var(--panel);border:2px solid #507eb0;border-radius:9px;overflow:hidden}} .view img{{display:block;width:100%;aspect-ratio:1.25;object-fit:contain;background:#252a30;cursor:zoom-in}} .caption{{padding:8px 10px;color:#c7d0da}} .scope{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#8dc4ff}} .order{{float:right;color:#83909e}}
.sidebar{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px;align-self:start;position:sticky;top:72px;max-height:calc(100vh - 90px);overflow:auto}} h2{{font-size:16px;margin:15px 0 5px;color:#a9cfff}} p{{margin:5px 0}} ul{{margin:5px 0 12px;padding-left:20px}} code{{color:#b6d8ff}} details{{margin-top:12px;border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:#161b21}} summary{{cursor:pointer;color:#a9cfff}}
.targets{{display:flex;gap:6px;flex-wrap:wrap}} .target{{background:#252f39;border:1px solid #465769;border-radius:6px;padding:4px 6px}}
.judges{{display:grid;grid-template-columns:1fr;gap:8px;margin:8px 0}} .judge{{border:1px solid #514471;border-radius:8px;padding:9px;background:#171c22}} .judge-head{{display:flex;justify-content:space-between;gap:8px;align-items:center}} .judge-meta{{display:flex;gap:8px;flex-wrap:wrap;color:var(--muted);font-size:12px}} .verdict{{font-weight:700;color:var(--model)}} .reason{{margin-top:7px;color:#d7dde4}} .findings{{color:#c4cbd3;font-size:13px}}
.labels{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:10px 0}} .labels button.active[data-value=valid]{{background:#174a2c;border-color:var(--ok)}} .labels button.active[data-value=invalid]{{background:#551f25;border-color:var(--bad)}} .labels button.active[data-value=ambiguous]{{background:#514019;border-color:var(--amb)}}
.checks label{{display:block;margin:8px 0}} .checks input{{margin-right:7px}} textarea{{width:100%;min-height:90px;background:#11161b;color:white;border:1px solid #4a5662;border-radius:7px;padding:9px;resize:vertical}} .saved{{color:var(--ok);font-size:13px;min-height:20px}}
.lightbox{{display:none;position:fixed;inset:0;background:#000e;z-index:30;align-items:center;justify-content:center;padding:24px}} .lightbox.open{{display:flex}} .lightbox img{{max-width:97vw;max-height:95vh;object-fit:contain}} .kbd{{color:var(--muted);font-size:12px}}
@media(max-width:1080px){{.layout{{grid-template-columns:1fr}}.sidebar{{position:static;max-height:none}}}} @media(max-width:700px){{.gallery{{grid-template-columns:1fr}}.progress{{text-align:left}}}}
</style>
</head>
<body>
<div class="top">
  <button id="prev">← Previous</button><button id="next">Next →</button>
  <select id="metric"><option value="all">all VLM questions</option>{options}</select>
  <select id="verdict"><option value="all">all verdicts</option><option value="valid">valid</option><option value="invalid">invalid</option><option value="ambiguous">ambiguous / unresolved</option></select>
  <select id="caseSelect" class="grow"></select>
  <button id="export">Export TSV</button>
  <span class="progress" id="progress"></span>
</div>
<main id="app"></main>
<div id="lightbox" class="lightbox"><img alt="Expanded exact model input"></div>
<script>
const CARDS={data};
const RUN={summary_json};
const STORE='cs_opus47_vlm_judge_review_v1';
let answers=JSON.parse(localStorage.getItem(STORE)||'{{}}');
let filtered=CARDS.slice(),index=0;
const app=document.getElementById('app'),sel=document.getElementById('caseSelect');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fmtScore=v=>v===null||v===undefined?'—':Number(v).toFixed(3);
function currentAnswer(c){{return answers[c.card_id]||{{human_label:'',evidence_sufficient:false,reviewed:false,notes:''}}}}
function save(c,patch){{answers[c.card_id]={{...currentAnswer(c),...patch}};localStorage.setItem(STORE,JSON.stringify(answers));render(false)}}
function applyFilter(){{const m=document.getElementById('metric').value,v=document.getElementById('verdict').value,old=filtered[index]?.card_id;filtered=CARDS.filter(c=>(m==='all'||c.metric===m)&&(v==='all'||(v==='ambiguous'?['ambiguous','unresolved'].includes(c.final_verdict)||c.final_status==='unresolved':c.final_verdict===v)));index=Math.max(0,filtered.findIndex(c=>c.card_id===old));if(index<0)index=0;rebuildSelect();render()}}
function rebuildSelect(){{sel.innerHTML=filtered.map((c,i)=>`<option value="${{i}}">${{esc(c.level)}} · ${{esc(c.metric)}} · ${{esc(c.title)}} · ${{esc(c.final_verdict)}}</option>`).join('');sel.value=String(index)}}
function judgeBlock(r){{const meta=[`evidence: ${{esc(r.evidence_status||'not recorded')}}`,`score: ${{fmtScore(r.score)}}`,r.confidence===null||r.confidence===undefined?'':`confidence: ${{Number(r.confidence).toFixed(2)}}`].filter(Boolean).map(x=>`<span>${{x}}</span>`).join('');const findings=(r.findings||[]).length?`<ul class="findings">${{r.findings.map(x=>`<li>${{esc(x)}}</li>`).join('')}}</ul>`:'';return `<article class="judge"><div class="judge-head"><strong>Judge call / repeat ${{r.repeat}}</strong><span class="verdict">${{esc(r.verdict)}}</span></div><div class="judge-meta">${{meta}}</div><p class="reason">${{esc(r.reason||'No reason recorded')}}</p>${{findings}}</article>`}}
function render(scroll=true){{
 const c=filtered[index];if(!c){{app.innerHTML='<p>No questions in this filter.</p>';document.getElementById('progress').textContent='0';return}}
 const a=currentAnswer(c);sel.value=String(index);const reviewed=filtered.filter(x=>currentAnswer(x).reviewed).length;
 document.getElementById('progress').textContent=`${{index+1}} / ${{filtered.length}} · reviewed ${{reviewed}}`;
 const images=c.images.map(v=>`<figure class="view"><img src="${{esc(v.src)}}" data-full="${{esc(v.src)}}" alt="${{esc(v.name)}}"><figcaption class="caption"><span class="scope">exact VLM image input</span><span class="order">#${{v.order}} / ${{c.image_count}}</span><br>${{esc(v.role)}} · ${{esc(v.name)}}</figcaption></figure>`).join('');
 const facts=c.required_visible_facts.map(x=>`<li>${{esc(x)}}</li>`).join('');
 const context=c.text_context.map(x=>`<li>${{esc(x)}}</li>`).join('');
 const targets=c.targets.map(x=>`<span class="target"><code>${{esc(x.id)}}</code> · ${{esc(x.description)}}</span>`).join('');
 app.innerHTML=`<div class="identity"><h1>${{esc(c.title)}}</h1><span class="pill">${{esc(c.level)}} · ${{esc(c.metric)}}</span><span class="pill ${{esc(c.final_verdict||c.final_status)}}">${{esc(c.final_status)}} · ${{esc(c.final_verdict)}} · ${{fmtScore(c.final_score)}}</span><span class="summary">Opus 4.7 scene · benchmark ${{fmtScore(RUN.benchmark_score)}}</span></div>
 <div class="notice"><strong>Input-faithful review.</strong> Every blue-bordered image was included in the actual VLM packet, in the displayed order. No contact sheet or later audit render is mixed in. The occupancy diagram is a neutral aid, not GT.</div>
 <div class="layout"><section class="gallery">${{images}}</section>
 <aside class="sidebar"><h2>Metric question</h2><p>${{esc(c.question)}}</p><h2>Targets</h2><div class="targets">${{targets}}</div><h2>Required visible facts</h2><ul>${{facts}}</ul>
 <h2>Other textual context received</h2><ul>${{context||'<li>metric rubric, object metadata, and detector measurements</li>'}}</ul>
 <details><summary>Frozen rubric</summary><p>${{esc(c.rubric)}}</p></details>
 <h2>VLM judgement</h2><div class="judges">${{c.judgements.map(judgeBlock).join('')}}</div>
 <h2>Your audit label</h2><div class="labels">${{['valid','invalid','ambiguous'].map(v=>`<button data-label="${{v}}" data-value="${{v}}" class="${{a.human_label===v?'active':''}}">${{v}}</button>`).join('')}}</div>
 <div class="checks"><label><input id="sufficient" type="checkbox" ${{a.evidence_sufficient?'checked':''}}>Visual evidence sufficient</label><label><input id="reviewed" type="checkbox" ${{a.reviewed?'checked':''}}>Reviewed</label></div>
 <h2>Notes</h2><textarea id="notes" placeholder="Evidence gap, rubric-boundary issue, or judgement note...">${{esc(a.notes)}}</textarea><p class="saved">${{a.reviewed?'Reviewed and saved locally':'Changes save locally'}}</p><p class="kbd">Keys: J/→ next · K/← previous · 1 valid · 2 invalid · 3 ambiguous · R reviewed</p></aside></div>`;
 document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>save(c,{{human_label:b.dataset.label}}));
 document.getElementById('sufficient').onchange=e=>save(c,{{evidence_sufficient:e.target.checked}});
 document.getElementById('reviewed').onchange=e=>save(c,{{reviewed:e.target.checked}});
 document.getElementById('notes').oninput=e=>{{answers[c.card_id]={{...currentAnswer(c),notes:e.target.value}};localStorage.setItem(STORE,JSON.stringify(answers))}};
 document.querySelectorAll('.view img').forEach(img=>img.onclick=()=>{{document.querySelector('#lightbox img').src=img.dataset.full;document.getElementById('lightbox').classList.add('open')}});
 if(scroll)window.scrollTo(0,0);
}}
function move(delta){{if(!filtered.length)return;index=(index+delta+filtered.length)%filtered.length;render()}}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
document.getElementById('metric').onchange=applyFilter;document.getElementById('verdict').onchange=applyFilter;sel.onchange=e=>{{index=Number(e.target.value);render()}};
document.getElementById('lightbox').onclick=e=>e.currentTarget.classList.remove('open');
document.addEventListener('keydown',e=>{{if(e.target.matches('textarea,input,select'))return;if(e.key==='j'||e.key==='ArrowRight')move(1);if(e.key==='k'||e.key==='ArrowLeft')move(-1);if(['1','2','3'].includes(e.key))save(filtered[index],{{human_label:{{'1':'valid','2':'invalid','3':'ambiguous'}}[e.key]}});if(e.key.toLowerCase()==='r')save(filtered[index],{{reviewed:!currentAnswer(filtered[index]).reviewed}})}});
document.getElementById('export').onclick=()=>{{const fields=['card_id','level','metric','title','machine_status','machine_verdict','machine_score','human_label','visual_evidence_sufficient','reviewed','notes'];const q=s=>'"'+String(s??'').replaceAll('"','""')+'"';const rows=[fields.join('\\t'),...CARDS.map(c=>{{const a=currentAnswer(c),values={{card_id:c.card_id,level:c.level,metric:c.metric,title:c.title,machine_status:c.final_status,machine_verdict:c.final_verdict,machine_score:c.final_score,human_label:a.human_label,visual_evidence_sufficient:a.evidence_sufficient,reviewed:a.reviewed,notes:a.notes}};return fields.map(f=>q(values[f])).join('\\t')}})];const blob=new Blob([rows.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='opus47_vlm_judge_review.tsv';a.click();URL.revokeObjectURL(a.href)}};
rebuildSelect();render();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    report_path = args.report.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    report = _read_json(report_path)
    case_root = report_path.parent
    out_root.mkdir(parents=True, exist_ok=True)

    cards = _scene_metric_cards(report, out_root, case_root=case_root)
    cards.extend(_collision_cards(report, out_root))

    payload = {
        "schema_version": "cs_opus47_vlm_judge_review_v1",
        "source_report": str(report_path),
        "card_count": len(cards),
        "scene_metric_card_count": sum(card["metric"] != "collision" for card in cards),
        "collision_pair_card_count": sum(card["metric"] == "collision" for card in cards),
        "cards": cards,
    }
    (out_root / "review_cases.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_root / "index.html").write_text(
        _html(cards, report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "index": str(out_root / "index.html"),
                "card_count": len(cards),
                "scene_metric_card_count": payload["scene_metric_card_count"],
                "collision_pair_card_count": payload["collision_pair_card_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
