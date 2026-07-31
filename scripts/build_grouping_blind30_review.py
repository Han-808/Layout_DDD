#!/usr/bin/env python3
"""Build the anonymous, input-faithful grouping review interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.grouping_blind30_contracts import (
    BLIND_LABELS,
    REVIEW_DATA_SCHEMA_VERSION,
    ExperimentPaths,
    atomic_write_json,
    load_experiment_config,
    read_json,
)
from scripts.grouping_blind30_dataset import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "experiments"
        / "grouping_blind30_gpt56_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config, paths = load_experiment_config(
        args.config,
        repo_root=PROJECT_ROOT,
        output_override=args.output_root,
    )
    dataset = prepare_dataset(config, paths, resume=True)
    review = build_review(
        config=config,
        paths=paths,
        dataset=dataset,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        json.dumps(
            {
                "review_index": str(paths.review_root / "index.html"),
                "scene_count": len(review["cases"]),
                "complete": review["complete"],
            },
            indent=2,
        )
    )


def build_review(
    *,
    config: dict[str, Any],
    paths: ExperimentPaths,
    dataset: dict[str, Any],
    allow_incomplete: bool,
) -> dict[str, Any]:
    method_key = read_json(paths.method_key)
    review_root = paths.review_root
    assets_root = review_root / "assets"
    cases: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for case in dataset["cases"]:
        case_id = str(case["case_id"])
        case_root = paths.case_root(case_id)
        public_assets = assets_root / case_id
        public_assets.mkdir(parents=True, exist_ok=True)
        input_manifest = read_json(
            case_root / "input" / "input_manifest.json"
        )
        render_manifest = read_json(
            case_root / "render" / "render_manifest.json"
        )
        view_paths = {
            str(item.get("name")): Path(str(item.get("path")))
            for item in render_manifest.get("views", [])
            if isinstance(item, dict)
        }
        originals = []
        for name, title, role in (
            (
                "perspective",
                "Original rendered scene · perspective",
                "global perspective RGB",
            ),
            (
                "top",
                "Original rendered scene · top",
                "global top RGB",
            ),
        ):
            source = view_paths.get(name)
            if source is None or not source.is_file():
                raise FileNotFoundError(
                    f"{case_id} is missing the {name} render"
                )
            destination = public_assets / f"original_{name}.png"
            _copy_if_changed(source, destination)
            originals.append(
                {
                    "name": name,
                    "title": title,
                    "role": role,
                    "src": f"assets/{case_id}/{destination.name}",
                    "scope": "shared_input",
                }
            )
        identity_source = Path(input_manifest["identity_map_path"])
        identity_destination = public_assets / "identity_map.png"
        _copy_if_changed(identity_source, identity_destination)
        originals.append(
            {
                "name": "identity",
                "title": "Object identity map",
                "role": "neutral ID aid",
                "src": (
                    f"assets/{case_id}/{identity_destination.name}"
                ),
                "scope": "shared_input",
            }
        )

        variants: list[dict[str, Any]] = []
        mapping = method_key["cases"][case_id]
        for blind_label in BLIND_LABELS:
            backend = mapping[blind_label]
            result_path = (
                case_root / "grouping" / backend / "result.json"
            )
            overlay_source = (
                case_root / "grouping" / backend / "overlay.png"
            )
            if not result_path.is_file() or not overlay_source.is_file():
                missing.append(
                    {
                        "case_id": case_id,
                        "blind_result_id": blind_label,
                    }
                )
                variants.append(
                    {
                        "blind_result_id": blind_label,
                        "status": "unavailable",
                        "group_count": None,
                        "groups": [],
                        "overlay_src": None,
                    }
                )
                continue
            private_record = read_json(result_path)
            if private_record.get("status") != "complete":
                missing.append(
                    {
                        "case_id": case_id,
                        "blind_result_id": blind_label,
                    }
                )
                variants.append(
                    {
                        "blind_result_id": blind_label,
                        "status": "unavailable",
                        "group_count": None,
                        "groups": [],
                        "overlay_src": None,
                    }
                )
                continue
            preview = private_record.get("blind_preview")
            if not isinstance(preview, dict):
                raise ValueError(
                    f"{result_path} is missing blind_preview"
                )
            overlay_destination = (
                public_assets / f"result_{blind_label}.png"
            )
            _copy_if_changed(overlay_source, overlay_destination)
            variants.append(
                {
                    "blind_result_id": blind_label,
                    "status": "complete",
                    "group_count": int(preview["group_count"]),
                    "groups": preview["groups"],
                    "overlay_src": (
                        f"assets/{case_id}/{overlay_destination.name}"
                    ),
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "scene_type": case["scene_type"],
                "object_count": int(case["object_count"]),
                "stratum": case["stratum"],
                "original_evidence": originals,
                "object_legend": [
                    {
                        "object_alias": input_manifest[
                            "object_aliases"
                        ][item["object_id"]],
                        "description": item["description"],
                    }
                    for item in input_manifest["object_catalog"]
                ],
                "variants": variants,
            }
        )
    if missing and not allow_incomplete:
        raise RuntimeError(
            f"blind review requires all 90 grouping results; "
            f"{len(missing)} anonymous results are unavailable"
        )
    public = {
        "schema_version": REVIEW_DATA_SCHEMA_VERSION,
        "experiment_id": config["_experiment_id"],
        "blind": True,
        "method_identity_included": False,
        "scene_count": len(cases),
        "results_per_scene": len(BLIND_LABELS),
        "complete": not missing,
        "unavailable_result_count": len(missing),
        "review_contract": {
            "best_result_values": [
                "A",
                "B",
                "C",
                "tie",
                "unclear",
            ],
            "quality_values": [
                "correct",
                "partially_correct",
                "incorrect",
                "unclear",
            ],
            "storage": (
                "local review server when available; browser localStorage "
                "fallback; JSON/TSV export"
            ),
        },
        "cases": cases,
    }
    _assert_public_payload_is_blind(public)
    review_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(review_root / "review_data.json", public)
    (review_root / "index.html").write_text(
        _review_html(public),
        encoding="utf-8",
    )
    return public


def _assert_public_payload_is_blind(value: Any) -> None:
    forbidden_keys = {
        "backend",
        "grouping_backend",
        "policy_id",
        "grouping_policy_id",
        "model",
        "endpoint",
        "provenance",
        "reason",
        "anchor_object_id",
    }

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            exposed = forbidden_keys.intersection(item)
            if exposed:
                raise ValueError(
                    f"blind review payload exposes {sorted(exposed)} at "
                    f"{path}"
                )
            for key, nested in item.items():
                visit(nested, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                visit(nested, f"{path}[{index}]")

    visit(value, "$")


def _copy_if_changed(source: Path, destination: Path) -> None:
    if (
        destination.is_file()
        and source.stat().st_size == destination.stat().st_size
        and source.read_bytes() == destination.read_bytes()
    ):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _review_html(public: dict[str, Any]) -> str:
    encoded = json.dumps(
        public,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    encoded = (
        encoded.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    return _HTML_TEMPLATE.replace("__REVIEW_DATA__", encoded)


def _repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (
        path.resolve()
        if path.is_absolute()
        else (repo_root / path).resolve()
    )


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind grouping review · 30 scenes</title>
<style>
:root{--bg:#111418;--panel:#1b2026;--panel2:#151a20;--muted:#9ba7b4;--line:#38424d;--accent:#62a9ff;--ok:#51c878;--bad:#ff6b6b;--warn:#e8bb55;--a:#a68cff;--b:#47d4c7;--c:#f4ad58}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef2f6;font:15px/1.45 system-ui,-apple-system,sans-serif}button,select,input,textarea{font:inherit}
.top{position:sticky;top:0;z-index:20;background:#15191ef2;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:10px 16px;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
button,select{background:#242b33;color:#eef2f6;border:1px solid #4b5865;border-radius:7px;padding:7px 10px}button:hover{border-color:var(--accent)}.grow{flex:1}.progress{color:var(--muted);min-width:220px;text-align:right}
main{max-width:1900px;margin:auto;padding:18px}.identity{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}h1{font-size:27px;margin:0}.pill{background:#26384c;color:#b9dcff;border:1px solid #41668d;padding:3px 8px;border-radius:999px}.summary{color:#aeb8c3;margin-left:auto}
.notice{color:#d7c58c;background:#2c281b;border:1px solid #62562c;border-radius:8px;padding:10px 12px;margin:12px 0}.section-title{font-size:17px;color:#a9cfff;margin:18px 0 8px}
.evidence{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.view,.variant{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:0}.view{border:2px solid #507eb0}.view img,.variant img{display:block;width:100%;aspect-ratio:1.42;object-fit:contain;background:#252a30;cursor:zoom-in}.caption{padding:8px 10px;color:#c7d0da}.scope{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:#8dc4ff}.order{float:right;color:#83909e}
.results{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;align-items:start}.variant{border-top-width:5px}.variant[data-id=A]{border-top-color:var(--a)}.variant[data-id=B]{border-top-color:var(--b)}.variant[data-id=C]{border-top-color:var(--c)}
.variant-head{display:flex;align-items:center;gap:9px;padding:10px 12px}.result-id{font-size:20px;font-weight:800}.variant[data-id=A] .result-id{color:var(--a)}.variant[data-id=B] .result-id{color:var(--b)}.variant[data-id=C] .result-id{color:var(--c)}.count{color:var(--muted);margin-left:auto}
.quality{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;padding:10px}.quality button.active{background:#274b67;border-color:#70baff}.quality button[data-value=correct].active{background:#174a2c;border-color:var(--ok)}.quality button[data-value=incorrect].active{background:#551f25;border-color:var(--bad)}.quality button[data-value=partially_correct].active{background:#514019;border-color:var(--warn)}
.members{margin:0 10px 10px;border:1px solid var(--line);border-radius:8px;background:var(--panel2)}.members summary{cursor:pointer;padding:8px 10px;color:#b7cae0}.group-row{padding:8px 10px;border-top:1px solid #2d353e}.group-label{font-weight:700}.chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}.chip{border:1px solid #485463;border-radius:999px;padding:2px 6px;font-size:12px;color:#d6dee7}.variant textarea{width:calc(100% - 20px);margin:0 10px 10px;min-height:60px}
.review-panel{margin-top:15px;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;display:grid;grid-template-columns:minmax(420px,1fr) minmax(420px,1fr);gap:18px}.review-panel h2{font-size:16px;margin:0 0 8px;color:#a9cfff}.best{display:flex;gap:7px;flex-wrap:wrap}.best button.active{background:#274b67;border-color:#70baff}.checks label{display:block;margin:8px 0}.checks input{margin-right:7px}
textarea{background:#11161b;color:white;border:1px solid #4a5662;border-radius:7px;padding:9px;resize:vertical}.scene-notes{width:100%;min-height:95px}.saved{color:var(--ok);font-size:13px;min-height:20px}.kbd{color:var(--muted);font-size:12px}
.unavailable{padding:80px 18px;text-align:center;color:#ffb1b1;background:#321d22}.lightbox{display:none;position:fixed;inset:0;background:#000e;z-index:40;align-items:center;justify-content:center;padding:24px}.lightbox.open{display:flex}.lightbox img{max-width:97vw;max-height:95vh;object-fit:contain}
@media(max-width:1180px){.results,.evidence{grid-template-columns:1fr}.review-panel{grid-template-columns:1fr}}@media(max-width:700px){.progress{text-align:left}.review-panel{min-width:0}}
</style>
</head>
<body>
<div class="top">
  <button id="prev">← Previous</button><button id="next">Next →</button>
  <select id="stratum"><option value="all">all object-count strata</option></select>
  <select id="reviewFilter"><option value="all">all review states</option><option value="pending">pending</option><option value="reviewed">reviewed</option></select>
  <select id="caseSelect" class="grow"></select>
  <button id="exportJson">Export JSON</button><button id="exportTsv">Export TSV</button>
  <span id="progress" class="progress"></span>
</div>
<main id="app"></main>
<div id="lightbox" class="lightbox"><img alt="full-size review image"></div>
<script>
const DATA=__REVIEW_DATA__;
const STORE=`${DATA.experiment_id}_blind_grouping_review_v1`;
const labels=['A','B','C'];
let answers=JSON.parse(localStorage.getItem(STORE)||'{}'),filtered=DATA.cases.slice(),index=0,saveTimer=null;
const app=document.getElementById('app'),caseSelect=document.getElementById('caseSelect');
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const baseAnswer=()=>({reviewed:false,best_result:'',notes:'',variants:{A:{quality:'',notes:''},B:{quality:'',notes:''},C:{quality:'',notes:''}}});
function answer(c){return answers[c.case_id]||baseAnswer()}
function normalizeAnswers(v){return v&&typeof v==='object'?v:{}}
async function loadRemote(){try{const r=await fetch('/api/reviews',{cache:'no-store'});if(!r.ok)throw Error();const body=await r.json(),remote=normalizeAnswers(body.answers);answers={...answers,...remote};localStorage.setItem(STORE,JSON.stringify(answers));render(false);document.getElementById('saveState')?.replaceChildren(document.createTextNode(Object.keys(remote).length?'Loaded from review backend':'Review backend connected'))}catch(_){document.getElementById('saveState')?.replaceChildren(document.createTextNode('Local browser storage · start the review server for file persistence'))}}
function persist(){localStorage.setItem(STORE,JSON.stringify(answers));clearTimeout(saveTimer);saveTimer=setTimeout(async()=>{try{const r=await fetch('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({schema_version:'grouping_blind30_human_reviews_v1',experiment_id:DATA.experiment_id,answers})});if(!r.ok)throw Error();document.getElementById('saveState')?.replaceChildren(document.createTextNode('Saved to review backend'))}catch(_){document.getElementById('saveState')?.replaceChildren(document.createTextNode('Saved in this browser; review backend unavailable'))}},300)}
function patch(c,update){answers[c.case_id]={...answer(c),...update};persist();render(false)}
function patchVariant(c,id,update){const a=answer(c);answers[c.case_id]={...a,variants:{...a.variants,[id]:{...a.variants[id],...update}}};persist();render(false)}
function rebuildFilters(){const sel=document.getElementById('stratum');[...new Set(DATA.cases.map(c=>c.stratum))].forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x;sel.appendChild(o)})}
function applyFilter(){const s=document.getElementById('stratum').value,r=document.getElementById('reviewFilter').value,old=filtered[index]?.case_id;filtered=DATA.cases.filter(c=>(s==='all'||c.stratum===s)&&(r==='all'||(r==='reviewed')===Boolean(answer(c).reviewed)));index=Math.max(0,filtered.findIndex(c=>c.case_id===old));if(index<0)index=0;rebuildSelect();render()}
function rebuildSelect(){caseSelect.innerHTML=filtered.map((c,i)=>`<option value="${i}">${esc(c.case_id)} · ${esc(c.scene_type)} · ${c.object_count} objects</option>`).join('');caseSelect.value=String(index)}
function imageBlock(v,i){return `<figure class="view"><img src="${esc(v.src)}" data-full="${esc(v.src)}" alt="${esc(v.title)}"><figcaption class="caption"><span class="scope">${esc(v.scope)}</span><span class="order">#${i+1}/3</span><br><strong>${esc(v.title)}</strong><br>${esc(v.role)}</figcaption></figure>`}
function groupRows(v){return v.groups.map(g=>`<div class="group-row"><span class="group-label" style="color:${esc(g.color)}">${esc(g.display_group_id)}</span><div class="chips">${g.members.map(m=>`<span class="chip" title="${esc(m.description)}">${esc(m.object_alias)} · ${esc(m.description)}</span>`).join('')}</div></div>`).join('')}
function variantBlock(c,v){const a=answer(c).variants[v.blind_result_id]||{quality:'',notes:''};if(v.status!=='complete')return `<article class="variant" data-id="${v.blind_result_id}"><div class="variant-head"><span class="result-id">Result ${v.blind_result_id}</span></div><div class="unavailable">Result unavailable — do not infer a method.</div></article>`;const qs=[['correct','Correct'],['partially_correct','Partially correct'],['incorrect','Incorrect'],['unclear','Unclear']];return `<article class="variant" data-id="${v.blind_result_id}"><div class="variant-head"><span class="result-id">Result ${v.blind_result_id}</span><span class="count">${v.group_count} groups</span></div><img src="${esc(v.overlay_src)}" data-full="${esc(v.overlay_src)}" alt="Anonymous grouping result ${v.blind_result_id}"><div class="quality">${qs.map(([x,t])=>`<button data-quality="${v.blind_result_id}" data-value="${x}" class="${a.quality===x?'active':''}">${t}</button>`).join('')}</div><details class="members"><summary>Exact group membership</summary>${groupRows(v)}</details><textarea data-variant-notes="${v.blind_result_id}" placeholder="Result-specific note…">${esc(a.notes)}</textarea></article>`}
function render(scroll=true){const c=filtered[index];if(!c){app.innerHTML='<p>No scenes match this filter.</p>';document.getElementById('progress').textContent='0';return}const a=answer(c);caseSelect.value=String(index);const reviewed=filtered.filter(x=>answer(x).reviewed).length;document.getElementById('progress').textContent=`${index+1} / ${filtered.length} · reviewed ${reviewed}`;app.innerHTML=`<div class="identity"><h1>${esc(c.case_id)} · ${esc(c.scene_type)}</h1><span class="pill">${c.object_count} objects</span><span class="pill">${esc(c.stratum)}</span><span class="summary">30-scene blind grouping review</span></div><div class="notice"><strong>Blind comparison.</strong> A, B, and C are independently shuffled for every scene. The three results receive the exact same scene, object catalog, and visual evidence. Method names, rationales, anchors, models, and policies are hidden.</div><h2 class="section-title">Shared original scene evidence</h2><section class="evidence">${c.original_evidence.map(imageBlock).join('')}</section><h2 class="section-title">Anonymous grouping results</h2><section class="results">${c.variants.map(v=>variantBlock(c,v)).join('')}</section><section class="review-panel"><div><h2>Best partition for this scene</h2><div class="best">${[['A','A'],['B','B'],['C','C'],['tie','Tie'],['unclear','Unclear']].map(([x,t])=>`<button data-best="${x}" class="${a.best_result===x?'active':''}">${t}</button>`).join('')}</div><div class="checks"><label><input id="reviewed" type="checkbox" ${a.reviewed?'checked':''}>Reviewed</label></div><p id="saveState" class="saved">${a.reviewed?'Reviewed · saving automatically':'Saving automatically'}</p><p class="kbd">Keys: J/→ next · K/← previous · 1/2/3 best A/B/C · R reviewed</p></div><div><h2>Scene-level notes</h2><textarea id="sceneNotes" class="scene-notes" placeholder="Split/merge errors, ambiguous boundary, or other observations…">${esc(a.notes)}</textarea></div></section>`;document.querySelectorAll('[data-quality]').forEach(b=>b.onclick=()=>patchVariant(c,b.dataset.quality,{quality:b.dataset.value}));document.querySelectorAll('[data-variant-notes]').forEach(t=>t.oninput=e=>{const id=e.target.dataset.variantNotes,cur=answer(c),v={...cur.variants[id],notes:e.target.value};answers[c.case_id]={...cur,variants:{...cur.variants,[id]:v}};persist()});document.querySelectorAll('[data-best]').forEach(b=>b.onclick=()=>patch(c,{best_result:b.dataset.best}));document.getElementById('reviewed').onchange=e=>patch(c,{reviewed:e.target.checked});document.getElementById('sceneNotes').oninput=e=>{answers[c.case_id]={...answer(c),notes:e.target.value};persist()};document.querySelectorAll('img[data-full]').forEach(img=>img.onclick=()=>{document.querySelector('#lightbox img').src=img.dataset.full;document.getElementById('lightbox').classList.add('open')});if(scroll)window.scrollTo(0,0)}
function move(delta){if(!filtered.length)return;index=(index+delta+filtered.length)%filtered.length;render()}
function download(name,type,text){const blob=new Blob([text],{type}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();URL.revokeObjectURL(a.href)}
function exportPayload(){return {schema_version:'grouping_blind30_human_reviews_v1',experiment_id:DATA.experiment_id,blind:true,answers}}
document.getElementById('exportJson').onclick=()=>download('grouping_blind30_reviews.json','application/json',JSON.stringify(exportPayload(),null,2)+'\n');
document.getElementById('exportTsv').onclick=()=>{const q=x=>'"'+String(x??'').replaceAll('"','""')+'"',fields=['case_id','scene_type','object_count','stratum','reviewed','best_result','result_A_quality','result_B_quality','result_C_quality','result_A_notes','result_B_notes','result_C_notes','scene_notes'],rows=[fields.join('\t')];DATA.cases.forEach(c=>{const a=answer(c),v={case_id:c.case_id,scene_type:c.scene_type,object_count:c.object_count,stratum:c.stratum,reviewed:a.reviewed,best_result:a.best_result,result_A_quality:a.variants.A.quality,result_B_quality:a.variants.B.quality,result_C_quality:a.variants.C.quality,result_A_notes:a.variants.A.notes,result_B_notes:a.variants.B.notes,result_C_notes:a.variants.C.notes,scene_notes:a.notes};rows.push(fields.map(f=>q(v[f])).join('\t'))});download('grouping_blind30_reviews.tsv','text/tab-separated-values',rows.join('\n')+'\n')};
document.getElementById('prev').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);document.getElementById('stratum').onchange=applyFilter;document.getElementById('reviewFilter').onchange=applyFilter;caseSelect.onchange=e=>{index=Number(e.target.value);render()};document.getElementById('lightbox').onclick=e=>e.currentTarget.classList.remove('open');document.addEventListener('keydown',e=>{if(e.target.matches('textarea,input,select'))return;if(e.key==='j'||e.key==='ArrowRight')move(1);if(e.key==='k'||e.key==='ArrowLeft')move(-1);if(['1','2','3'].includes(e.key))patch(filtered[index],{best_result:{'1':'A','2':'B','3':'C'}[e.key]});if(e.key.toLowerCase()==='r')patch(filtered[index],{reviewed:!answer(filtered[index]).reviewed})});
rebuildFilters();rebuildSelect();render();loadRemote();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
