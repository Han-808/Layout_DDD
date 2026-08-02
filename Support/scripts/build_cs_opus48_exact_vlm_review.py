#!/usr/bin/env python3
"""Build an input-faithful review UI for the latest Opus 4.8 CS run.

The generated review contains one recorded judgement per card.  For visual
metrics that were sampled twice, only repeat_index=0 is shown.  Prompt text is
reconstructed through the same production prompt builders and the script
fails if its text-character count differs from the count recorded at request
time.  Review images are the metadata-free, alpha-flattened RGB PNG bytes
produced by the same outbound image normalization code used for the VLM call.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Callable

from benchmark.evaluator.scene_quality.interfaces import _judge_request
from benchmark.game_scene.counter_strike.evaluator import _neutral_visual_context
from benchmark.game_scene.counter_strike.judge import (
    JUDGE_SYSTEM_PROMPT,
    _judge_user_prompt,
    _metric_rubric,
    _multimodal_messages,
    _normalized_rgb_png_data_url as _cs_image_data_url,
)
from benchmark.game_scene.counter_strike.loader import (
    load_counter_strike_benchmark_config,
)
from benchmark.models.openai_compatible_model import _message_text_chars
from benchmark.visual_judge.openai_compatible import (
    CANONICAL_METRIC_SYSTEM_PROMPT,
    CATEGORY_RUBRICS,
    P0B_SYSTEM_PROMPT,
    _budgeted_context_json,
    _generic_view_names,
    _image_data_url,
    _sanitize_outbound_view_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "Support/artifacts/outputs/cs_benchmark_v1_gpt56_20260728_v3_r1"
)
DEFAULT_CASE_ROOT = DEFAULT_RUN_ROOT / "cases/claude_opus_4_8"
DEFAULT_REPORT = DEFAULT_CASE_ROOT / "counter_strike_evaluation_report.json"
DEFAULT_OUT = DEFAULT_RUN_ROOT / "opus48_exact_vlm_review"
DEFAULT_CS_CONFIG = REPO_ROOT / "configs/game/counter_strike/benchmark_v1.yaml"
SCENE_METRICS = (
    "style_consistency",
    "zone_clarity",
    "landmark_legibility",
    "cover_diversity",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_data_url(value: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        raise ValueError("outbound image normalizer did not return a PNG data URL")
    return base64.b64decode(value[len(prefix) :], validate=True)


def _write_exact_image(
    *,
    source: Path,
    out_root: Path,
    card_id: str,
    index: int,
    alias: str,
    normalizer: Callable[[Path], str],
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    outbound = _decoded_data_url(normalizer(source))
    destination = (
        out_root
        / "exact_model_images"
        / card_id
        / f"{index:02d}_{alias}.png"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(outbound)
    return {
        "order": index + 1,
        "alias": alias,
        "src": Path(os.path.relpath(destination, out_root)).as_posix(),
        "source_path": str(source),
        "source_sha256": _sha256_file(source),
        "outbound_png_sha256": _sha256_bytes(outbound),
        "normalization": "EXIF transpose → RGBA → white-flattened RGB → PNG",
    }


def _prompt_payload(
    *,
    system_message: str,
    user_message: str,
    expected_prompt_chars: int,
    image_labels: list[str] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": user_message}]
    for label in image_labels or []:
        content.extend(
            (
                {"type": "text", "text": f"Image ID: {label}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            )
        )
    if image_labels is None:
        content.extend([])
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content},
    ]
    actual = _message_text_chars(messages)
    if actual != expected_prompt_chars:
        raise RuntimeError(
            "prompt reconstruction mismatch: "
            f"expected {expected_prompt_chars} chars, reconstructed {actual}"
        )
    return {
        "system_message": system_message,
        "user_message": user_message,
        "expected_prompt_chars": expected_prompt_chars,
        "reconstructed_prompt_chars": actual,
        "prompt_chars_match": True,
        "system_sha256": _sha256_bytes(system_message.encode("utf-8")),
        "user_sha256": _sha256_bytes(user_message.encode("utf-8")),
    }


def _canonical_style_card(
    report: dict[str, Any],
    *,
    case_root: Path,
    out_root: Path,
) -> dict[str, Any]:
    record = report["metric_vector"]["style_consistency"]
    judgement = record["judgement"]
    scene = _read_json(case_root / "generated_scene.json")
    request_record = _read_json(case_root / "case_bundle/scene_request.json")
    visual_style_spec = _read_json(
        case_root / "case_bundle/visual_style_spec.json"
    )
    request = _judge_request(
        metric_name="style_consistency",
        scene=scene,
        prompt=str(request_record["instruction"]),
        render_evidence=list(record["evidence_paths"]),
        selected_object_ids=[],
        selected_group_ids=[],
        groups=[],
        authorized_deviations=list(record["authorized_deviations"]),
        visual_style_spec=visual_style_spec,
        evidence_phase="global_screen",
        decision_mode="screen",
    )
    source_paths = [Path(value) for value in request["render_evidence"]]
    context = {
        "family": "scene_quality",
        "metric": "style_consistency",
        "rubric": CATEGORY_RUBRICS["style_consistency"],
        "metric_rubric": request.get("metric_rubric"),
        "natural_language_request": request.get("prompt"),
        "authorized_deviations": request.get("authorized_deviations"),
        "metric_scope": request.get("judgment_scope"),
        "claims": request.get("claims"),
        "components": request.get("components"),
        "object_groups": request.get("object_groups"),
        "visual_style_spec": request.get("visual_style_spec"),
        "canonical_scene": request.get("scene_summary"),
        "deterministic_evidence": request.get("deterministic_evidence"),
        "evidence_phase": request.get("evidence_phase"),
        "decision_mode": request.get("decision_mode"),
        "phase_instruction": (
            "This is the global screening pass. Return valid when the global "
            "views are sufficient and show no significant in-scope style "
            "defect. Return invalid only with explicit target IDs for a "
            "significant visible defect. Return ambiguous/insufficient when "
            "closer local evidence is needed; do not guess."
        ),
        "view_names": _generic_view_names(source_paths),
    }
    context_text = _budgeted_context_json(
        context,
        30000,
        priority_keys=(
            "metric",
            "rubric",
            "metric_rubric",
            "natural_language_request",
            "authorized_deviations",
            "metric_scope",
            "claims",
            "components",
            "object_groups",
            "visual_style_spec",
            "deterministic_evidence",
            "evidence_phase",
            "decision_mode",
            "phase_instruction",
            "view_names",
        ),
    )
    user_message = (
        "Adjudicate this canonical metric only.\n" + context_text
    )
    prompt = _prompt_payload(
        system_message=CANONICAL_METRIC_SYSTEM_PROMPT,
        user_message=user_message,
        expected_prompt_chars=int(
            judgement["request_metadata"]["prompt_chars"]
        ),
    )
    card_id = "scene_style_consistency"
    images = [
        _write_exact_image(
            source=path,
            out_root=out_root,
            card_id=card_id,
            index=index,
            alias=f"image_{index:02d}",
            normalizer=_image_data_url,
        )
        for index, path in enumerate(source_paths)
    ]
    return {
        "card_id": card_id,
        "level": "L3",
        "metric": "style_consistency",
        "title": "Style Consistency",
        "prompt": prompt,
        "images": images,
        "judgement": deepcopy(judgement),
        "request_metadata": deepcopy(judgement["request_metadata"]),
        "evidence_phase": record.get("route"),
        "repeat_policy": "single production call",
    }


def _l4_record(report: dict[str, Any], metric: str) -> dict[str, Any]:
    value = report["metric_vector"][metric]
    if metric in {"zone_clarity", "cover_diversity"}:
        return value["perceptual_component"]
    return value


def _l4_card(
    report: dict[str, Any],
    *,
    metric: str,
    case_root: Path,
    out_root: Path,
    config: Any,
) -> dict[str, Any]:
    record = _l4_record(report, metric)
    judgement = next(
        item
        for item in record["repeats"]
        if int(item.get("repeat_index", -1)) == 0
    )
    view_ids = ["global_oblique_00", "global_oblique_01"] + list(
        record.get("selected_regional_ids") or []
    )
    source_paths: list[Path] = []
    for view_id in view_ids:
        prefix = "global" if view_id.startswith("global_") else "local"
        source_paths.append(case_root / "renders" / f"{prefix}_{view_id}.png")
    topology_path = (
        case_root / "counter_strike_l4/judge_observation_diagram.png"
    )
    user_message = _judge_user_prompt(
        metric,
        metric_config=config.raw["l4_metrics"][metric],
        topology_context=_neutral_visual_context(metric),
        evidence_phase=str(record["evidence_phase"]),
        view_ids=view_ids,
    )
    image_labels = [*view_ids, "topology_diagram"]
    prompt = _prompt_payload(
        system_message=JUDGE_SYSTEM_PROMPT,
        user_message=user_message,
        expected_prompt_chars=int(
            judgement["request_metadata"]["prompt_chars"]
        ),
        image_labels=image_labels,
    )
    card_id = f"scene_{metric}"
    images = [
        _write_exact_image(
            source=path,
            out_root=out_root,
            card_id=card_id,
            index=index,
            alias=alias,
            normalizer=_cs_image_data_url,
        )
        for index, (alias, path) in enumerate(
            zip(image_labels, [*source_paths, topology_path], strict=True)
        )
    ]
    return {
        "card_id": card_id,
        "level": "L4",
        "metric": metric,
        "title": metric.replace("_", " ").title(),
        "prompt": prompt,
        "images": images,
        "judgement": deepcopy(judgement),
        "request_metadata": deepcopy(judgement["request_metadata"]),
        "evidence_phase": record.get("evidence_phase"),
        "repeat_policy": (
            "repeat_index=0 only; repeat_index=1 intentionally omitted from "
            "this no-repeat review"
        ),
        "frozen_rubric": _metric_rubric(
            metric, config.raw["l4_metrics"][metric]
        ),
    }


def _collision_user_message(request: dict[str, Any]) -> str:
    paths = [Path(str(value)).expanduser() for value in request["render_evidence"]]
    context = {
        "metric": request.get("metric"),
        "metric_rubric": request.get("metric_rubric"),
        "candidate_selection_policy": request.get(
            "candidate_selection_policy"
        ),
        "collision_evidence_style_guide": request.get(
            "collision_evidence_style_guide"
        ),
        "visual_evidence_policy": request.get("visual_evidence_policy"),
        "event": request.get("event"),
        "detector_evidence": request.get("detector_evidence"),
        "objects": request.get("objects"),
        "architecture": request.get("architecture"),
        "natural_language_prompt": request.get("natural_language_prompt"),
        "extracted_relationships": request.get("extracted_relationships"),
        "view_names": _generic_view_names(paths),
        "view_evidence": _sanitize_outbound_view_evidence(
            request.get("local_render_evidence_metadata")
        ),
    }
    return "Adjudicate this P0b event.\n" + _budgeted_context_json(
        context,
        30000,
        priority_keys=(
            "metric",
            "detector_evidence",
            "event",
            "natural_language_prompt",
            "metric_rubric",
            "candidate_selection_policy",
            "collision_evidence_style_guide",
            "visual_evidence_policy",
            "view_names",
            "view_evidence",
        ),
    )


def _collision_cards(
    report: dict[str, Any],
    *,
    out_root: Path,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(
        report["metric_vector"]["collision"]["pairs"]
    ):
        judge_result = pair.get("judge_result")
        if not isinstance(judge_result, dict):
            continue
        request = judge_result["request"]
        judgement = judge_result["judgement"]
        object_a = str(pair["object_a"])
        object_b = str(pair["object_b"])
        card_id = f"collision_{pair_index:03d}_{object_a}_{object_b}"
        source_paths = [
            Path(value) for value in request["render_evidence"]
        ]
        user_message = _collision_user_message(request)
        prompt = _prompt_payload(
            system_message=P0B_SYSTEM_PROMPT,
            user_message=user_message,
            expected_prompt_chars=int(
                judgement["request_metadata"]["prompt_chars"]
            ),
        )
        images = [
            _write_exact_image(
                source=path,
                out_root=out_root,
                card_id=card_id,
                index=index,
                alias=f"image_{index:02d}",
                normalizer=_image_data_url,
            )
            for index, path in enumerate(source_paths)
        ]
        cards.append(
            {
                "card_id": card_id,
                "level": "L1",
                "metric": "collision",
                "title": f"{object_a} ↔ {object_b}",
                "prompt": prompt,
                "images": images,
                "judgement": deepcopy(judgement),
                "request_metadata": deepcopy(
                    judgement["request_metadata"]
                ),
                "evidence_phase": "collision local evidence + global context",
                "repeat_policy": "single production call",
                "targets": [object_a, object_b],
                "frozen_rubric": request.get("metric_rubric"),
            }
        )
    return cards


def _html(cards: list[dict[str, Any]], report: dict[str, Any]) -> str:
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    options = "".join(
        f'<option value="{html.escape(metric)}">'
        f"{html.escape(metric.replace('_', ' '))}</option>"
        for metric in (
            "style_consistency",
            "zone_clarity",
            "landmark_legibility",
            "cover_diversity",
            "collision",
        )
    )
    benchmark_score = report.get("benchmark_score")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Opus 4.8 · exact VLM review</title>
<style>
:root{{--bg:#111418;--panel:#1b2026;--muted:#96a2ae;--line:#37414c;--accent:#62a9ff;--ok:#51c878;--bad:#ff6b6b;--amb:#e8bb55}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:#edf2f7;font:15px/1.45 system-ui,-apple-system,sans-serif}}button,select{{font:inherit;background:#242b33;color:#eef2f6;border:1px solid #4b5865;border-radius:7px;padding:7px 10px}}button:hover{{border-color:var(--accent)}}.top{{position:sticky;top:0;z-index:20;background:#15191ef2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}}.grow{{flex:1}}main{{max-width:1780px;margin:auto;padding:18px}}h1{{font-size:25px;margin:0}}.identity{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:12px}}.pill{{padding:3px 8px;border:1px solid #41668d;background:#26384c;color:#b9dcff;border-radius:999px}}.pill.valid{{background:#174a2c;border-color:#397b51;color:#9beab3}}.pill.invalid{{background:#551f25;border-color:#8b454d;color:#ffabb3}}.pill.ambiguous{{background:#514019;border-color:#8e7130;color:#ffe1a1}}.summary{{margin-left:auto;color:var(--muted)}}.notice{{margin:0 0 14px;padding:10px 12px;border:1px solid #62562c;background:#2c281b;color:#eadb9e;border-radius:8px}}.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(440px,1fr);gap:16px}}.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;align-content:start}}figure{{margin:0;background:var(--panel);border:2px solid #4f8ac7;border-radius:9px;overflow:hidden}}figure img{{display:block;width:100%;aspect-ratio:1.25;object-fit:contain;background:#252a30;cursor:zoom-in}}figcaption{{padding:8px 10px;color:#cad2dc}}.scope{{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#8dc4ff}}.order{{float:right;color:var(--muted)}}.hash{{font:11px ui-monospace,monospace;color:#8e9ba8;overflow-wrap:anywhere}}aside{{position:sticky;top:72px;align-self:start;max-height:calc(100vh - 90px);overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px}}h2{{font-size:16px;color:#a9cfff;margin:15px 0 6px}}p{{margin:6px 0}}.judge{{border:1px solid #514471;border-radius:8px;padding:10px;background:#171c22}}.judge-head{{display:flex;justify-content:space-between;gap:8px}}.verdict{{font-weight:700;color:#bd9cff}}.meta{{font-size:12px;color:var(--muted)}}details{{margin-top:10px;border:1px solid var(--line);border-radius:8px;background:#151a20;padding:8px 10px}}summary{{cursor:pointer;color:#a9cfff}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0f1317;border:1px solid #303944;border-radius:6px;padding:10px;color:#d8e1ea;max-height:420px;overflow:auto}}.exact{{border-color:#3f7cb7}}.integrity{{color:var(--ok)}}.lightbox{{display:none;position:fixed;inset:0;background:#000e;z-index:30;align-items:center;justify-content:center;padding:24px}}.lightbox.open{{display:flex}}.lightbox img{{max-width:97vw;max-height:95vh;object-fit:contain}}@media(max-width:1100px){{.layout{{grid-template-columns:1fr}}aside{{position:static;max-height:none}}}}@media(max-width:720px){{.gallery{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="top">
<button id="prev">← Previous</button><button id="next">Next →</button>
<select id="metric"><option value="all">all VLM calls</option>{options}</select>
<select id="card" class="grow"></select><span id="progress"></span>
</div>
<main id="app"></main>
<div class="lightbox" id="lightbox"><img alt="Exact VLM image input"></div>
<script>
const CARDS={data};let filtered=CARDS.slice(),index=0;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const fmt=v=>v===null||v===undefined?'—':Number(v).toFixed(3);
const cardSelect=document.getElementById('card'),app=document.getElementById('app');
function rebuild(){{cardSelect.innerHTML=filtered.map((c,i)=>`<option value="${{i}}">${{esc(c.level)}} · ${{esc(c.metric)}} · ${{esc(c.title)}}</option>`).join('');cardSelect.value=String(index)}}
function findings(j){{for(const key of ['role_findings','landmarks','cover_findings']){{if(Array.isArray(j[key]))return `<details><summary>Structured findings</summary><pre>${{esc(JSON.stringify(j[key],null,2))}}</pre></details>`}}return ''}}
function render(){{
 const c=filtered[index];if(!c){{app.innerHTML='<p>No calls in this filter.</p>';return}}cardSelect.value=String(index);document.getElementById('progress').textContent=`${{index+1}} / ${{filtered.length}}`;
 const j=c.judgement,imgs=c.images.map(x=>`<figure><img src="${{esc(x.src)}}" data-full="${{esc(x.src)}}" alt="${{esc(x.alias)}}"><figcaption><span class="scope">exact outbound VLM pixels</span><span class="order">#${{x.order}}/${{c.images.length}}</span><br><strong>${{esc(x.alias)}}</strong><div class="hash">outbound sha256 ${{esc(x.outbound_png_sha256)}}</div><details><summary>Source + normalization</summary><div class="hash">${{esc(x.source_path)}}<br>source sha256 ${{esc(x.source_sha256)}}<br>${{esc(x.normalization)}}</div></details></figcaption></figure>`).join('');
 app.innerHTML=`<div class="identity"><h1>${{esc(c.title)}}</h1><span class="pill">${{esc(c.level)}} · ${{esc(c.metric)}}</span><span class="pill ${{esc(j.verdict)}}">${{esc(j.verdict)}} · ${{fmt(j.score)}}</span><span class="summary">Opus 4.8 latest v3_r1 · benchmark {fmtScore(benchmark_score)}</span></div>
 <div class="notice"><strong>Exact-input reconstruction.</strong> The PNGs below are the same outbound-normalized pixel bytes, in the same order, as the production VLM packet. The complete metric-specific user message is shown verbatim. Prompt integrity is hard-checked against the request log. No second repeat is displayed.</div>
 <div class="layout"><section class="gallery">${{imgs}}</section><aside>
 <h2>Exact metric-specific user message</h2><pre class="exact">${{esc(c.prompt.user_message)}}</pre>
 <p class="integrity">✓ prompt chars ${{c.prompt.reconstructed_prompt_chars}} / ${{c.prompt.expected_prompt_chars}} · exact match</p>
 <details><summary>Exact system message</summary><pre>${{esc(c.prompt.system_message)}}</pre></details>
 <details><summary>Prompt hashes</summary><div class="hash">system ${{esc(c.prompt.system_sha256)}}<br>user ${{esc(c.prompt.user_sha256)}}</div></details>
 <h2>Recorded VLM judgement</h2><article class="judge"><div class="judge-head"><strong>One displayed call</strong><span class="verdict">${{esc(j.verdict)}}</span></div><div class="meta">evidence ${{esc(j.evidence_status||'not recorded')}} · score ${{fmt(j.score)}} · confidence ${{fmt(j.confidence)}}</div><p>${{esc(j.reason||'No reason recorded')}}</p>${{findings(j)}}</article>
 <h2>Call identity</h2><p><code>${{esc(c.request_metadata.call_type)}}</code></p><p class="meta">${{esc(c.repeat_policy)}}<br>evidence phase: ${{esc(c.evidence_phase)}}<br>image count: ${{c.images.length}}</p>
 <details><summary>Recorded request metadata</summary><pre>${{esc(JSON.stringify(c.request_metadata,null,2))}}</pre></details>
 </aside></div>`;
 document.querySelectorAll('figure img').forEach(img=>img.onclick=()=>{{document.querySelector('#lightbox img').src=img.dataset.full;document.getElementById('lightbox').classList.add('open')}});window.scrollTo(0,0);
}}
function move(d){{if(!filtered.length)return;index=(index+d+filtered.length)%filtered.length;render()}}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);cardSelect.onchange=e=>{{index=Number(e.target.value);render()}};
document.getElementById('metric').onchange=e=>{{filtered=e.target.value==='all'?CARDS.slice():CARDS.filter(c=>c.metric===e.target.value);index=0;rebuild();render()}};
document.getElementById('lightbox').onclick=e=>e.currentTarget.classList.remove('open');
document.addEventListener('keydown',e=>{{if(e.target.matches('select'))return;if(e.key==='ArrowRight'||e.key==='j')move(1);if(e.key==='ArrowLeft'||e.key==='k')move(-1)}});
rebuild();render();
</script>
</body>
</html>"""


def fmtScore(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "—"
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cs-config", type=Path, default=DEFAULT_CS_CONFIG)
    args = parser.parse_args()

    report_path = args.report.expanduser().resolve()
    case_root = report_path.parent
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report = _read_json(report_path)
    if report.get("scene_id") != "claude_opus_4_8_scene":
        raise ValueError("review builder requires the Opus 4.8 scene report")
    config = load_counter_strike_benchmark_config(args.cs_config)

    cards = [
        _canonical_style_card(
            report, case_root=case_root, out_root=out_root
        )
    ]
    cards.extend(
        _l4_card(
            report,
            metric=metric,
            case_root=case_root,
            out_root=out_root,
            config=config,
        )
        for metric in (
            "zone_clarity",
            "landmark_legibility",
            "cover_diversity",
        )
    )
    cards.extend(_collision_cards(report, out_root=out_root))
    payload = {
        "schema_version": "cs_opus48_exact_vlm_review_v1",
        "source_report": str(report_path),
        "model_scene": "claude_opus_4_8",
        "run": report_path.parents[2].name,
        "repeat_policy": (
            "one judgement per card; first call only for repeated L4 metrics"
        ),
        "card_count": len(cards),
        "prompt_integrity_passed": all(
            card["prompt"]["prompt_chars_match"] for card in cards
        ),
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
                "manifest": str(out_root / "review_cases.json"),
                "card_count": len(cards),
                "scene_metric_calls": 4,
                "collision_calls": len(cards) - 4,
                "prompt_integrity_passed": payload[
                    "prompt_integrity_passed"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
