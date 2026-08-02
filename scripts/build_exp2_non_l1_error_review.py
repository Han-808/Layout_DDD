#!/usr/bin/env python3
"""Build a local review UI for production-default judgement errors in Exp 2.

The user's human labels are the only ground truth. A case is included when its
human label is binary and at least one of the two exact-input
``production_default`` calls is incorrect. Human-ambiguous cases are excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    REPO_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "exp2_non_l1_visual_evidence_gpt56"
)
DATASET_ROOT = (
    REPO_ROOT / "Support" / "datasets" / "cal_dataset2_non_l1_evidence"
)
GT_PATH = DATASET_ROOT / "human_review" / "human_gt_20260725.tsv"
REVIEW_ROOT = (
    REPO_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "cal_dataset2_non_l1_review_renders"
    / "review"
)
REVIEW_CASES_PATH = REVIEW_ROOT / "review_cases.json"
OUT_ROOT = RUN_ROOT / "error_review"


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.expanduser().resolve()
    gt_path = args.ground_truth.expanduser().resolve()
    review_cases_path = args.review_cases.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    human_rows = _read_tsv(gt_path)
    review_payload = _read_json(review_cases_path)
    review_cards = {
        str(card["case_id"]): card
        for card in review_payload.get("cases") or []
        if isinstance(card, dict)
    }
    if len(review_cards) != len(human_rows):
        raise ValueError(
            "human GT and review card counts differ: "
            f"{len(human_rows)} vs {len(review_cards)}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    wrong_call_count = 0
    for human_row in human_rows:
        human_label = str(human_row["human_semantic_label"])
        if human_label not in {"valid", "invalid"}:
            continue
        case_id = str(human_row["case_id"])
        repeat_results = [
            _read_json(
                run_root
                / f"repeat_{repeat}"
                / "events"
                / case_id
                / "production_default.json"
            )
            for repeat in (1, 2)
        ]
        _validate_repeat_pair(
            case_id=case_id,
            human_label=human_label,
            repeat_results=repeat_results,
        )
        wrong_flags = [result["binary_match"] is False for result in repeat_results]
        if not any(wrong_flags):
            continue
        wrong_call_count += sum(wrong_flags)
        cards.append(
            _error_card(
                output_root=out_root,
                human_row=human_row,
                review_card=review_cards[case_id],
                repeat_results=repeat_results,
            )
        )

    if not cards:
        raise ValueError("no production-default error cases found")
    status_counts = Counter(card["error_status"] for card in cards)
    metric_counts = Counter(card["metric"] for card in cards)
    if len(cards) != 38 or wrong_call_count != 71:
        raise ValueError(
            "unexpected error universe: "
            f"{len(cards)} cases / {wrong_call_count} calls"
        )

    index_path = out_root / "index.html"
    index_path.write_text(_review_html(cards), encoding="utf-8")
    _write_json(
        out_root / "error_cases.json",
        {
            "schema_version": "exp2_non_l1_error_review_cases_v1",
            "scope": {
                "arm": "production_default",
                "human_gt_policy": "human binary labels are authoritative",
                "case_inclusion": "at least one of two exact-input calls is wrong",
                "human_ambiguous_policy": "excluded",
            },
            "case_count": len(cards),
            "wrong_call_count": wrong_call_count,
            "status_counts": dict(sorted(status_counts.items())),
            "metric_counts": dict(sorted(metric_counts.items())),
            "cases": cards,
        },
    )
    source_files = (
        gt_path,
        review_cases_path,
        run_root / "repeat_1" / "per_event.tsv",
        run_root / "repeat_2" / "per_event.tsv",
    )
    _write_json(
        out_root / "manifest.json",
        {
            "schema_version": "exp2_non_l1_error_review_manifest_v1",
            "index": str(index_path),
            "case_count": len(cards),
            "wrong_call_count": wrong_call_count,
            "status_counts": dict(sorted(status_counts.items())),
            "metric_counts": dict(sorted(metric_counts.items())),
            "source_sha256": {
                str(path): _sha256(path) for path in source_files
            },
            "outputs_sha256": {
                "index.html": _sha256(index_path),
                "error_cases.json": _sha256(out_root / "error_cases.json"),
            },
        },
    )
    print(
        json.dumps(
            {
                "index": str(index_path),
                "case_count": len(cards),
                "wrong_call_count": wrong_call_count,
                "stable_wrong_case_count": status_counts["stable_wrong"],
                "one_repeat_wrong_case_count": status_counts["one_repeat_wrong"],
                "metric_counts": dict(sorted(metric_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _error_card(
    *,
    output_root: Path,
    human_row: dict[str, str],
    review_card: dict[str, Any],
    repeat_results: list[dict[str, Any]],
) -> dict[str, Any]:
    case_id = str(human_row["case_id"])
    production_paths = {
        Path(str(item["path"])).resolve()
        for item in repeat_results[0].get("evidence") or []
    }
    if production_paths != {
        Path(str(item["path"])).resolve()
        for item in repeat_results[1].get("evidence") or []
    }:
        raise ValueError(f"{case_id}: repeat evidence paths differ")

    gallery: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for item in repeat_results[0].get("evidence") or []:
        source = Path(str(item["path"])).resolve()
        gallery.append(
            {
                "scope": "model_input",
                "family": str(item.get("role") or "model_input"),
                "name": str(item.get("view_name") or source.stem),
                "src": _relative_existing(source, output_root),
            }
        )
        seen_paths.add(source)

    for item in review_card.get("rendered_views") or []:
        source = (REVIEW_ROOT / str(item["src"])).resolve()
        if source in seen_paths:
            continue
        gallery.append(
            {
                "scope": "additional_audit",
                "family": str(item.get("family") or "audit"),
                "name": str(item.get("name") or source.stem),
                "src": _relative_existing(source, output_root),
            }
        )
        seen_paths.add(source)

    construction_svg = (
        REVIEW_ROOT / str(review_card["construction_svg"])
    ).resolve()
    wrong_repeat_count = sum(
        result["binary_match"] is False for result in repeat_results
    )
    return {
        "case_id": case_id,
        "event_id": str(human_row["event_id"]),
        "metric": str(human_row["metric"]),
        "level": str(review_card["level"]),
        "prompt_granularity": str(review_card["prompt_granularity"]),
        "prompt": str(review_card["prompt"]),
        "review_question": str(review_card["review_question"]),
        "required_visible_facts": list(
            review_card.get("required_visible_facts") or []
        ),
        "target_objects": list(review_card.get("target_objects") or []),
        "human_gt": str(human_row["human_semantic_label"]),
        "human_notes": str(human_row.get("notes") or ""),
        "error_status": (
            "stable_wrong" if wrong_repeat_count == 2 else "one_repeat_wrong"
        ),
        "wrong_repeat_count": wrong_repeat_count,
        "machine_results": [
            {
                "repeat": int(index),
                "predicted_label": str(result["predicted_label"]),
                "is_correct": result["binary_match"] is True,
                "evidence_status": str(result["evidence_status"]),
                "confidence": float(result["confidence"]),
                "reason": str(result["reason"]),
                "missing_evidence": list(result.get("missing_evidence") or []),
            }
            for index, result in enumerate(repeat_results, start=1)
        ],
        "gallery": gallery,
        "construction_svg": _relative_existing(
            construction_svg,
            output_root,
        ),
    }


def _validate_repeat_pair(
    *,
    case_id: str,
    human_label: str,
    repeat_results: list[dict[str, Any]],
) -> None:
    if len(repeat_results) != 2:
        raise ValueError(f"{case_id}: expected two repeats")
    for repeat, result in enumerate(repeat_results, start=1):
        if result.get("error"):
            raise ValueError(f"{case_id}: repeat {repeat} has an error")
        if result.get("arm") != "production_default":
            raise ValueError(f"{case_id}: wrong arm in repeat {repeat}")
        if result.get("gt_label") != human_label:
            raise ValueError(f"{case_id}: GT mismatch in repeat {repeat}")
        if result.get("binary_scoreable") is not True:
            raise ValueError(f"{case_id}: repeat {repeat} is not binary scoreable")
    if (
        repeat_results[0].get("evidence_packet_sha256")
        != repeat_results[1].get("evidence_packet_sha256")
    ):
        raise ValueError(f"{case_id}: exact-input evidence packet hash drift")


def _review_html(cards: list[dict[str, Any]]) -> str:
    data = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    metrics = sorted({card["metric"] for card in cards})
    metric_options = "".join(
        f'<option value="{html.escape(metric)}">{html.escape(metric)}</option>'
        for metric in metrics
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exp 2 production error review</title>
<style>
:root{{--bg:#111418;--panel:#1b2026;--muted:#9ba7b4;--line:#38424d;--accent:#62a9ff;--ok:#51c878;--bad:#ff6b6b;--amb:#e8bb55;--model:#9f7aea}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:#eef2f6;font:15px/1.45 system-ui,sans-serif}}
button,select,input,textarea{{font:inherit}} .top{{position:sticky;top:0;z-index:10;background:#15191ef2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}}
button,select{{background:#242b33;color:#eef2f6;border:1px solid #4b5865;border-radius:7px;padding:7px 10px}} button:hover{{border-color:var(--accent)}} .grow{{flex:1}} .progress{{color:var(--muted);min-width:210px;text-align:right}}
main{{max-width:1660px;margin:auto;padding:18px}} .identity{{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}} h1{{font-size:25px;margin:0}} .pill{{background:#26384c;color:#b9dcff;border:1px solid #41668d;padding:3px 8px;border-radius:999px}} .pill.error{{background:#4b2529;color:#ffd1d5;border-color:#88474e}} .pill.unstable{{background:#4b3d20;color:#ffe4a6;border-color:#8c7339}}
.notice{{color:#d7c58c;background:#2c281b;border:1px solid #62562c;border-radius:8px;padding:10px 12px;margin:12px 0}}
.layout{{display:grid;grid-template-columns:minmax(0,2fr) minmax(360px,.9fr);gap:16px}} .gallery-block h2{{margin-top:0}} .gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;align-content:start}}
.view{{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}} .view.model-input{{border-color:#507eb0}} .view.audit-only{{border-color:#4a5159}} .view img{{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#252a30;cursor:zoom-in}} .caption{{padding:7px 9px;color:#c7d0da}} .scope{{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#8dc4ff}}
.sidebar{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:14px;align-self:start;position:sticky;top:72px;max-height:calc(100vh - 90px);overflow:auto}} h2{{font-size:16px;margin:14px 0 5px;color:#a9cfff}} h3{{font-size:15px;margin:0 0 5px}} p{{margin:5px 0}} ul{{margin:5px 0 12px;padding-left:20px}} code{{color:#b6d8ff}} .targets{{display:flex;gap:6px;flex-wrap:wrap}} .target{{background:#252f39;border:1px solid #465769;border-radius:6px;padding:4px 6px}}
.gt{{display:flex;align-items:center;justify-content:space-between;gap:12px;background:#15291e;border:1px solid #315d41;border-radius:8px;padding:9px 10px;margin:8px 0}} .gt strong{{color:#86e1a3}}
.judges{{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:8px 0}} .judge{{border:1px solid var(--line);border-radius:8px;padding:9px;background:#171c22}} .judge.wrong{{border-color:#7b3b43}} .judge.correct{{border-color:#315d41}} .judge-meta{{display:flex;gap:6px;flex-wrap:wrap;color:var(--muted);font-size:12px}} .verdict{{font-weight:700}} .judge.wrong .verdict{{color:#ff9da6}} .judge.correct .verdict{{color:#86e1a3}} .reason{{margin-top:7px;color:#d7dde4}}
.labels{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:10px 0}} .labels button.active[data-value=valid]{{background:#174a2c;border-color:var(--ok)}} .labels button.active[data-value=invalid]{{background:#551f25;border-color:var(--bad)}} .labels button.active[data-value=ambiguous]{{background:#514019;border-color:var(--amb)}}
.checks label{{display:block;margin:8px 0}} .checks input{{margin-right:7px}} textarea{{width:100%;min-height:100px;background:#11161b;color:white;border:1px solid #4a5662;border-radius:7px;padding:9px;resize:vertical}}
.saved{{color:var(--ok);font-size:13px;min-height:20px}} .lightbox{{display:none;position:fixed;inset:0;background:#000e;z-index:30;align-items:center;justify-content:center;padding:24px}} .lightbox.open{{display:flex}} .lightbox img{{max-width:96vw;max-height:94vh;object-fit:contain}} .kbd{{color:var(--muted);font-size:12px}}
@media(max-width:1050px){{.layout{{grid-template-columns:1fr}}.sidebar{{position:static;max-height:none}}}} @media(max-width:700px){{.gallery,.judges{{grid-template-columns:1fr}}.progress{{text-align:left}}}}
</style>
</head>
<body>
<div class="top">
  <button id="prev">← Previous</button><button id="next">Next →</button>
  <select id="metric"><option value="all">all metrics</option>{metric_options}</select>
  <select id="status"><option value="all">all error status</option><option value="stable_wrong">stable wrong</option><option value="one_repeat_wrong">one-repeat wrong</option></select>
  <select id="caseSelect" class="grow"></select>
  <button id="export">Export TSV</button>
  <span class="progress" id="progress"></span>
</div>
<main id="app"></main>
<div id="lightbox" class="lightbox"><img alt="Expanded review image"></div>
<script>
const CASES={data};
const STORE='exp2_non_l1_production_error_audit_v1';
let answers=JSON.parse(localStorage.getItem(STORE)||'{{}}');
let filtered=CASES.slice(),index=0;
const app=document.getElementById('app'),sel=document.getElementById('caseSelect');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function currentAnswer(c){{return answers[c.case_id]||{{audit_label:c.human_gt,prompt_compatible:false,target_mapping_correct:false,needs_render_check:false,reviewed:false,notes:c.human_notes||''}}}}
function save(c,patch){{answers[c.case_id]={{...currentAnswer(c),...patch}};localStorage.setItem(STORE,JSON.stringify(answers));render(false)}}
function applyFilter(){{const m=document.getElementById('metric').value,s=document.getElementById('status').value,old=filtered[index]?.case_id;filtered=CASES.filter(c=>(m==='all'||c.metric===m)&&(s==='all'||c.error_status===s));index=Math.max(0,filtered.findIndex(c=>c.case_id===old));if(index<0)index=0;rebuildSelect();render()}}
function rebuildSelect(){{sel.innerHTML=filtered.map((c,i)=>`<option value="${{i}}">${{esc(c.case_id)}} · ${{esc(c.metric)}} · ${{esc(c.error_status)}}</option>`).join('');sel.value=String(index)}}
function judgeBlock(r){{const missing=(r.missing_evidence||[]).length?`<p><strong>Missing:</strong> ${{esc(r.missing_evidence.join(' · '))}}</p>`:'';return `<article class="judge ${{r.is_correct?'correct':'wrong'}}"><h3>Repeat ${{r.repeat}} · <span class="verdict">${{esc(r.predicted_label)}}</span></h3><div class="judge-meta"><span>${{r.is_correct?'correct':'wrong vs GT'}}</span><span>evidence: ${{esc(r.evidence_status)}}</span><span>confidence: ${{Number(r.confidence).toFixed(2)}}</span></div><p class="reason">${{esc(r.reason)}}</p>${{missing}}</article>`}}
function render(scroll=true){{
 const c=filtered[index];if(!c){{app.innerHTML='<p>No cases in this filter.</p>';document.getElementById('progress').textContent='0 cases';return}}
 const a=currentAnswer(c);sel.value=String(index);
 const reviewed=filtered.filter(x=>currentAnswer(x).reviewed).length;
 document.getElementById('progress').textContent=`${{index+1}} / ${{filtered.length}} · reviewed ${{reviewed}}`;
 const views=c.gallery.map(v=>`<figure class="view ${{v.scope==='model_input'?'model-input':'audit-only'}}"><img src="${{esc(v.src)}}" data-full="${{esc(v.src)}}" alt="${{esc(v.name)}}"><figcaption class="caption"><span class="scope">${{v.scope==='model_input'?'exact model input':'additional audit view'}}</span><br>${{esc(v.family)}} · ${{esc(v.name)}}</figcaption></figure>`).join('');
 const facts=c.required_visible_facts.map(x=>`<li>${{esc(x)}}</li>`).join('');
 const targets=c.target_objects.map(x=>`<span class="target"><code>${{esc(x.id)}}</code> · ${{esc(x.description)}}</span>`).join('');
 const statusText=c.error_status==='stable_wrong'?'stable wrong · both repeats':'one-repeat wrong';
 app.innerHTML=`<div class="identity"><h1>${{esc(c.case_id)}}</h1><span class="pill">${{esc(c.metric)}}</span><span class="pill ${{c.error_status==='stable_wrong'?'error':'unstable'}}">${{statusText}}</span><span>${{esc(c.level)}} · ${{esc(c.prompt_granularity)}}</span></div>
 <div class="notice"><strong>Error audit.</strong> Your original human label is the scoring GT. Blue-bordered images are the exact production evidence packet; remaining views are audit-only.</div>
 <div class="layout"><section class="gallery-block"><div class="gallery">${{views}}<figure class="view audit-only"><img src="${{esc(c.construction_svg)}}" data-full="${{esc(c.construction_svg)}}" alt="Target geometry legend"><figcaption class="caption"><span class="scope">audit-only</span><br>target-ID geometry legend</figcaption></figure></div></section>
 <aside class="sidebar"><h2>Prompt</h2><p>${{esc(c.prompt)}}</p><h2>Metric question</h2><p>${{esc(c.review_question)}}</p>
 <h2>Targets</h2><div class="targets">${{targets}}</div><h2>Required visible facts</h2><ul>${{facts}}</ul>
 <h2>Your original GT</h2><div class="gt"><span>Human semantic label</span><strong>${{esc(c.human_gt)}}</strong></div>${{c.human_notes?`<p><strong>Original note:</strong> ${{esc(c.human_notes)}}</p>`:''}}
 <h2>Machine judgement</h2><div class="judges">${{c.machine_results.map(judgeBlock).join('')}}</div>
 <h2>Audit label</h2><div class="labels">${{['valid','invalid','ambiguous'].map(v=>`<button data-label="${{v}}" data-value="${{v}}" class="${{a.audit_label===v?'active':''}}">${{v}}</button>`).join('')}}</div>
 <div class="checks"><label><input id="reviewed" type="checkbox" ${{a.reviewed?'checked':''}}>Reviewed</label><label><input id="promptOK" type="checkbox" ${{a.prompt_compatible?'checked':''}}>Prompt compatible</label><label><input id="targetOK" type="checkbox" ${{a.target_mapping_correct?'checked':''}}>Target mapping correct</label><label><input id="renderNeeded" type="checkbox" ${{a.needs_render_check?'checked':''}}>Needs additional render check</label></div>
 <h2>Notes</h2><textarea id="notes" placeholder="Confirm GT, revise label, or record visual-evidence / judge-reasoning issue...">${{esc(a.notes)}}</textarea><p class="saved">${{a.reviewed?'Reviewed and saved locally':'Changes save locally'}}</p><p class="kbd">Keys: J/→ next · K/← previous · 1 valid · 2 invalid · 3 ambiguous · R reviewed</p></aside></div>`;
 document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>save(c,{{audit_label:b.dataset.label}}));
 document.getElementById('reviewed').onchange=e=>save(c,{{reviewed:e.target.checked}});
 document.getElementById('promptOK').onchange=e=>save(c,{{prompt_compatible:e.target.checked}});
 document.getElementById('targetOK').onchange=e=>save(c,{{target_mapping_correct:e.target.checked}});
 document.getElementById('renderNeeded').onchange=e=>save(c,{{needs_render_check:e.target.checked}});
 document.getElementById('notes').oninput=e=>{{answers[c.case_id]={{...currentAnswer(c),notes:e.target.value}};localStorage.setItem(STORE,JSON.stringify(answers))}};
 document.querySelectorAll('.view img').forEach(img=>img.onclick=()=>{{document.querySelector('#lightbox img').src=img.dataset.full;document.getElementById('lightbox').classList.add('open')}});
 if(scroll)window.scrollTo(0,0);
}}
function move(delta){{if(!filtered.length)return;index=(index+delta+filtered.length)%filtered.length;render()}}
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
document.getElementById('metric').onchange=applyFilter;document.getElementById('status').onchange=applyFilter;sel.onchange=e=>{{index=Number(e.target.value);render()}};
document.getElementById('lightbox').onclick=e=>e.currentTarget.classList.remove('open');
document.addEventListener('keydown',e=>{{if(e.target.matches('textarea,input,select'))return;if(e.key==='j'||e.key==='ArrowRight')move(1);if(e.key==='k'||e.key==='ArrowLeft')move(-1);if(['1','2','3'].includes(e.key))save(filtered[index],{{audit_label:{{'1':'valid','2':'invalid','3':'ambiguous'}}[e.key]}});if(e.key.toLowerCase()==='r')save(filtered[index],{{reviewed:!currentAnswer(filtered[index]).reviewed}})}});
document.getElementById('export').onclick=()=>{{const fields=['case_id','event_id','metric','error_status','original_human_gt','audit_label','label_changed','repeat_1_prediction','repeat_2_prediction','repeat_1_evidence_status','repeat_2_evidence_status','prompt_compatible','target_mapping_correct','needs_render_check','reviewed','notes','reason_repeat_1','reason_repeat_2'];const q=s=>'"'+String(s??'').replaceAll('"','""')+'"';const rows=[fields.join('\\t'),...CASES.map(c=>{{const a=currentAnswer(c),r1=c.machine_results[0],r2=c.machine_results[1],values={{case_id:c.case_id,event_id:c.event_id,metric:c.metric,error_status:c.error_status,original_human_gt:c.human_gt,audit_label:a.audit_label,label_changed:a.audit_label!==c.human_gt,repeat_1_prediction:r1.predicted_label,repeat_2_prediction:r2.predicted_label,repeat_1_evidence_status:r1.evidence_status,repeat_2_evidence_status:r2.evidence_status,prompt_compatible:a.prompt_compatible,target_mapping_correct:a.target_mapping_correct,needs_render_check:a.needs_render_check,reviewed:a.reviewed,notes:a.notes,reason_repeat_1:r1.reason,reason_repeat_2:r2.reason}};return fields.map(f=>q(values[f])).join('\\t')}})];const blob=new Blob([rows.join('\\n')+'\\n'],{{type:'text/tab-separated-values'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='exp2_non_l1_production_error_audit.tsv';a.click();URL.revokeObjectURL(a.href)}};
rebuildSelect();render();
</script>
</body>
</html>"""


def _relative_existing(source: Path, output_root: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    return Path(os.path.relpath(source, output_root)).as_posix()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--ground-truth", type=Path, default=GT_PATH)
    parser.add_argument("--review-cases", type=Path, default=REVIEW_CASES_PATH)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
