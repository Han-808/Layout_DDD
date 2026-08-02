#!/usr/bin/env python3
"""Build the blind human-review UI for the L3 mutation dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_l3_mutation_dataset import DEFAULT_CONFIG, load_config


REVIEW_SCHEMA_VERSION = "l3_mutation_review_data_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, output_override=args.output_root)
    result = build_review(
        config,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(result, indent=2))


def build_review(
    config: dict[str, Any],
    *,
    allow_incomplete: bool,
) -> dict[str, Any]:
    output_root = Path(config["_output_root"])
    dataset = _read_json(output_root / "dataset_manifest.json")
    source_by_id = {
        str(item["source_id"]): item for item in dataset["sources"]
    }
    cases: list[dict[str, Any]] = []
    incomplete: list[str] = []
    for variant in sorted(
        dataset["variants"],
        key=lambda item: str(item["review_id"]),
    ):
        source = source_by_id[str(variant["source_id"])]
        if (
            source.get("render_status") != "complete"
            or variant.get("render_status") != "complete"
        ):
            incomplete.append(str(variant["review_id"]))
            if not allow_incomplete:
                continue
        review_id = str(variant["review_id"])
        cases.append(
            {
                "review_id": review_id,
                "scene_type": str(source["scene_type"]),
                "object_count": int(source["object_count"]),
                "source_id": str(source["source_id"]),
                "source": _public_render_record(
                    source,
                    output_root=output_root,
                ),
                "variant": _public_render_record(
                    variant,
                    output_root=output_root,
                ),
                "permalink": f"/review/index.html?case={review_id}",
            }
        )
    if incomplete and not allow_incomplete:
        raise RuntimeError(
            f"{len(incomplete)} review cases are not fully rendered; "
            "rerun rendering or use --allow-incomplete"
        )
    review_root = output_root / "review"
    review_data = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "experiment_id": dataset["experiment_id"],
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "case_count": len(cases),
        "cases": cases,
        "annotation_contract": {
            "overall_labels": [
                "",
                "valid",
                "invalid",
                "ambiguous",
            ],
            "severity_labels": [
                "",
                "none",
                "minor",
                "moderate",
                "major",
            ],
            "issue_labels": [
                "orientation_or_function",
                "object_compatibility",
                "scale_consistency",
                "style_consistency",
                "other",
            ],
            "evidence_labels": [
                "",
                "sufficient",
                "insufficient",
                "uncertain",
            ],
            "render_labels": [
                "",
                "works",
                "broken",
                "uncertain",
            ],
        },
    }
    _write_json(review_root / "review_data.json", review_data)
    (review_root / "index.html").write_text(
        _review_html(),
        encoding="utf-8",
    )
    return {
        "review_index": str((review_root / "index.html").resolve()),
        "case_count": len(cases),
        "incomplete_count": len(incomplete),
        "server_command": (
            "PYTHONPATH=src .venv/bin/python "
            "scripts/serve_l3_mutation_review.py "
            f"--output-root {output_root}"
        ),
        "url": (
            f"http://{config['review']['host']}:"
            f"{config['review']['port']}/review/index.html"
        ),
    }


def _public_render_record(
    record: dict[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    if record.get("render_status") != "complete":
        return {
            "status": "unavailable",
            "top": None,
            "perspective": None,
            "identity": None,
            "blend": None,
        }
    views = record.get("view_paths") or {}
    return {
        "status": "complete",
        "top": _web_path(Path(str(views["top"])), output_root),
        "perspective": _web_path(
            Path(str(views["perspective"])), output_root
        ),
        "identity": _web_path(
            Path(str(views["identity_map"])), output_root
        ),
        "blend": _web_path(
            Path(str(record["blend_file"])), output_root
        ),
    }


def _web_path(path: Path, output_root: Path) -> str:
    relative = path.resolve().relative_to(output_root.resolve())
    return "/" + relative.as_posix()


def _review_html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>L3 Scene Mutation Review</title>
  <style>
    :root{color-scheme:dark;--bg:#0b0e13;--panel:#141922;--line:#2a3342;--text:#eef3fb;--muted:#93a0b4;--blue:#70a5ff;--good:#5ed4a8;--warn:#ffca6b}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
    header{position:sticky;top:0;z-index:10;background:#0b0e13ee;border-bottom:1px solid var(--line);padding:16px 22px;backdrop-filter:blur(14px)}
    h1{font-size:20px;margin:0 0 5px}.sub{color:var(--muted)}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:12px}
    button,select,input,textarea{font:inherit;color:var(--text);background:#111722;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
    button{cursor:pointer}button:hover{border-color:var(--blue)}button.primary{background:#1d5fd1;border-color:#2f79ed}
    .progress{margin-left:auto;color:var(--good)}main{max-width:1500px;margin:0 auto;padding:18px}.case{background:var(--panel);border:1px solid var(--line);border-radius:14px;margin:0 0 22px;overflow:hidden}
    .case-head{display:flex;gap:12px;align-items:center;padding:13px 16px;border-bottom:1px solid var(--line)}.case-head h2{font-size:17px;margin:0}.meta{color:var(--muted)}
    .images{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}.side{background:#0d1118;padding:12px}.side h3{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0 0 10px}
    .hero{width:100%;aspect-ratio:1/1;object-fit:contain;background:#090b10;border-radius:8px}.secondary{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}.secondary img{width:100%;aspect-ratio:1/1;object-fit:contain;background:#090b10;border-radius:7px}
    .actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}.actions a{color:var(--blue);text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:7px 9px}.actions a:hover{border-color:var(--blue)}
    .form{padding:15px 16px;display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}.field label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}.field select,.field textarea{width:100%}.issues{grid-column:span 2}.checks{display:flex;gap:12px;flex-wrap:wrap}.checks label{color:var(--text);font-size:13px}.checks input{margin-right:5px}.notes{grid-column:span 2}.notes textarea{min-height:70px;resize:vertical}
    .saved{color:var(--good);font-size:12px;margin-left:auto}.hidden{display:none!important}.lightbox{position:fixed;inset:0;background:#000e;z-index:30;display:flex;align-items:center;justify-content:center;padding:24px}.lightbox img{max-width:96vw;max-height:94vh}
    @media(max-width:900px){.images{grid-template-columns:1fr}.form{grid-template-columns:1fr 1fr}.issues,.notes{grid-column:span 2}}@media(max-width:560px){.form{grid-template-columns:1fr}.issues,.notes{grid-column:span 1}}
  </style>
</head>
<body>
<header>
  <h1>L3 scene mutation review</h1>
  <div class="sub">Compare the frozen source with the anonymous variation. Mutation family and intended severity are hidden.</div>
  <div class="toolbar">
    <button id="prev">Previous</button><button id="next">Next</button>
    <select id="filter"><option value="all">All cases</option><option value="unreviewed">Unreviewed</option><option value="reviewed">Reviewed</option><option value="broken">Render broken</option></select>
    <button class="primary" id="save">Save now</button>
    <span class="progress" id="progress"></span>
  </div>
</header>
<main id="root"></main>
<div id="lightbox" class="lightbox hidden"><img alt="expanded view"></div>
<script>
const STATE={data:null,answers:{},dirty:false};
const issueLabels=["orientation_or_function","object_compatibility","scale_consistency","style_consistency","other"];
const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
async function load(){
  STATE.data=await fetch("/review/review_data.json",{cache:"no-store"}).then(r=>r.json());
  try{const saved=await fetch("/api/reviews",{cache:"no-store"}).then(r=>r.json());STATE.answers=saved.answers||{}}catch(_){}
  render();
  const wanted=new URLSearchParams(location.search).get("case");if(wanted)setTimeout(()=>document.getElementById(wanted)?.scrollIntoView(),50);
}
function answer(id){return STATE.answers[id]||(STATE.answers[id]={overall_label:"",severity:"",issues:[],evidence_sufficiency:"",render_integrity:"",notes:"",reviewed:false})}
function render(){
  const mode=document.getElementById("filter").value;
  const cases=STATE.data.cases.filter(c=>{const a=STATE.answers[c.review_id];if(mode==="unreviewed")return !a?.reviewed;if(mode==="reviewed")return !!a?.reviewed;if(mode==="broken")return a?.render_integrity==="broken";return true});
  document.getElementById("root").innerHTML=cases.map(card).join("");
  bind();updateProgress();
}
function card(c){
  const a=answer(c.review_id);const checked=x=>a.issues.includes(x)?"checked":"";
  return `<section class="case" id="${esc(c.review_id)}"><div class="case-head"><h2>${esc(c.review_id)}</h2><span class="meta">${esc(c.scene_type)} · ${c.object_count} objects</span><a class="saved" href="${esc(c.permalink)}">permalink</a></div>
  <div class="images">${side(c,"source","Frozen source",c.source)}${side(c,"variant","Anonymous variation",c.variant)}</div>
  <div class="form" data-id="${esc(c.review_id)}">
    <div class="field"><label>Overall validity</label><select data-k="overall_label">${opts(["","valid","invalid","ambiguous"],a.overall_label)}</select></div>
    <div class="field"><label>Observed severity</label><select data-k="severity">${opts(["","none","minor","moderate","major"],a.severity)}</select></div>
    <div class="field"><label>Visual evidence</label><select data-k="evidence_sufficiency">${opts(["","sufficient","insufficient","uncertain"],a.evidence_sufficiency)}</select></div>
    <div class="field"><label>Render integrity</label><select data-k="render_integrity">${opts(["","works","broken","uncertain"],a.render_integrity)}</select></div>
    <div class="field issues"><label>Detected issue types</label><div class="checks">${issueLabels.map(x=>`<label><input type="checkbox" data-issue="${x}" ${checked(x)}>${x.replaceAll("_"," ")}</label>`).join("")}</div></div>
    <div class="field notes"><label>Notes</label><textarea data-k="notes">${esc(a.notes)}</textarea></div>
    <div class="field"><label>Complete</label><div class="checks"><label><input type="checkbox" data-k="reviewed" ${a.reviewed?"checked":""}>Reviewed</label></div></div>
  </div></section>`;
}
function side(c,which,title,v){
  if(v.status!=="complete")return `<div class="side"><h3>${title}</h3><p>Render unavailable.</p></div>`;
  return `<div class="side"><h3>${title}</h3><img class="hero zoom" src="${esc(v.top)}" alt="${title} top"><div class="secondary"><img class="zoom" src="${esc(v.perspective)}" alt="${title} perspective"><img class="zoom" src="${esc(v.identity)}" alt="${title} identity"></div><div class="actions"><button data-open="${which}" data-id="${esc(c.review_id)}">Open & move/render in Blender</button><a href="${esc(v.blend)}">Download .blend</a></div></div>`;
}
function opts(values,current){return values.map(x=>`<option value="${x}" ${x===current?"selected":""}>${x||"—"}</option>`).join("")}
function bind(){
  document.querySelectorAll(".form").forEach(f=>{const id=f.dataset.id;f.querySelectorAll("[data-k]").forEach(el=>el.addEventListener("change",()=>{const a=answer(id);a[el.dataset.k]=el.type==="checkbox"?el.checked:el.value;changed()}));f.querySelectorAll("[data-issue]").forEach(el=>el.addEventListener("change",()=>{const a=answer(id);const set=new Set(a.issues);el.checked?set.add(el.dataset.issue):set.delete(el.dataset.issue);a.issues=[...set];changed()}));});
  document.querySelectorAll("[data-open]").forEach(b=>b.addEventListener("click",async()=>{b.disabled=true;try{const r=await fetch("/api/open-blender",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({review_id:b.dataset.id,which:b.dataset.open})});const x=await r.json();if(!r.ok)throw Error(x.error||"open failed")}catch(e){alert(e.message)}finally{b.disabled=false}}));
  document.querySelectorAll(".zoom").forEach(img=>img.addEventListener("click",()=>{const l=document.getElementById("lightbox");l.querySelector("img").src=img.src;l.classList.remove("hidden")}));
}
function changed(){STATE.dirty=true;updateProgress();clearTimeout(changed.t);changed.t=setTimeout(save,700)}
async function save(){if(!STATE.data)return;const payload={schema_version:"l3_mutation_human_reviews_v1",experiment_id:STATE.data.experiment_id,dataset_fingerprint:STATE.data.dataset_fingerprint,answers:STATE.answers};const r=await fetch("/api/reviews",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(!r.ok)throw Error((await r.json()).error||"save failed");STATE.dirty=false;updateProgress()}
function updateProgress(){const total=STATE.data?.cases.length||0,done=Object.values(STATE.answers).filter(x=>x.reviewed).length;document.getElementById("progress").textContent=`${done}/${total} reviewed${STATE.dirty?" · unsaved":""}`}
function jump(delta){const cards=[...document.querySelectorAll(".case")],y=scrollY+100;let i=cards.findIndex(x=>x.offsetTop>=y);if(i<0)i=cards.length-1;i=Math.max(0,Math.min(cards.length-1,i+delta));cards[i]?.scrollIntoView({behavior:"smooth"})}
document.getElementById("save").onclick=()=>save().catch(e=>alert(e.message));document.getElementById("filter").onchange=render;document.getElementById("prev").onclick=()=>jump(-1);document.getElementById("next").onclick=()=>jump(1);document.getElementById("lightbox").onclick=e=>e.currentTarget.classList.add("hidden");window.addEventListener("beforeunload",e=>{if(STATE.dirty){e.preventDefault();e.returnValue=""}});load();
</script>
</body>
</html>"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
