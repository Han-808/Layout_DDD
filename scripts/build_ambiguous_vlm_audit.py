#!/usr/bin/env python3
"""Build a step-through audit of ambiguous evidence and repeated VLM verdicts."""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "Support"
    / "experiment_analysis"
    / "exp1_1_visual_config_20260723"
)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Support" / "datasets" / "cal_dataset1"
ARMS = ("fixed_global_highlight", "metric_local_highlight")
ARM_LABELS = {
    "fixed_global_highlight": "Fixed global + highlight",
    "metric_local_highlight": "Metric-local + highlight",
}
CONTACT_SHEET_TILE_SIZE = 320
CONTACT_SHEET_LABEL_HEIGHT = 28


def main() -> None:
    args = _parse_args()
    evidence_root = Path(args.evidence_root).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    fragment_path = Path(args.fragment).expanduser().resolve()

    rows = [
        row
        for row in _read_tsv(results_path)
        if row.get("stratum") == "ambiguous"
    ]
    if not rows:
        raise ValueError(f"no ambiguous rows in {results_path}")
    results = {
        (
            row["case_id"],
            row["metric"],
            row["event_id"],
            row["arm"],
            int(row["repeat"]),
        ): row
        for row in rows
    }

    manifests: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in sorted(
        evidence_root.glob("cases/*/events/*/comparison_manifest.json")
    ):
        payload = _read_json(path)
        if payload.get("semantic_label") != "ambiguous":
            continue
        key = (
            str(payload["case_id"]),
            str(payload["metric"]),
            str(payload["event_id"]),
        )
        manifests[key] = payload

    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in sorted(manifests):
        grouped[key[0]].append(key)
    if not grouped:
        raise ValueError(f"no ambiguous comparison manifests under {evidence_root}")

    object_maps: dict[str, dict[str, str]] = {}
    reports: dict[str, dict[str, Any]] = {}
    for case_id in grouped:
        scene = _read_json(
            dataset_root / "fixtures" / case_id / "generated_scene.json"
        )
        object_maps[case_id] = {
            str(item["id"]): str(item.get("category") or "unknown object")
            for item in scene.get("objects") or []
        }
        reports[case_id] = _read_json(
            dataset_root
            / "evaluation"
            / "mesh"
            / case_id
            / "generic_validity.json"
        )

    fragment = _build_fragment(
        grouped,
        manifests,
        results,
        object_maps,
        reports,
    )
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(fragment, encoding="utf-8")
    print(json.dumps({
        "fragment": str(fragment_path),
        "scene_count": len(grouped),
        "event_count": len(manifests),
        "judgement_count": len(results),
        "bytes": fragment_path.stat().st_size,
    }, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        default=str(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "outputs"
            / "exp1_1_fine_edge"
        ),
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_ANALYSIS_ROOT / "data" / "job1" / "per_event.tsv"),
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--fragment", required=True)
    return parser.parse_args()


def _build_fragment(
    grouped: dict[str, list[tuple[str, str, str]]],
    manifests: dict[tuple[str, str, str], dict[str, Any]],
    results: dict[tuple[str, str, str, str, int], dict[str, str]],
    object_maps: dict[str, dict[str, str]],
    reports: dict[str, dict[str, Any]],
) -> str:
    cases = sorted(grouped)
    panels: list[str] = []
    for scene_index, case_id in enumerate(cases):
        events: list[str] = []
        for _, metric, event_id in grouped[case_id]:
            manifest = manifests[(case_id, metric, event_id)]
            identity = _event_identity_context(
                manifest,
                object_maps[case_id],
                reports[case_id],
            )
            arm_blocks: list[str] = []
            for arm in ARMS:
                items = (manifest.get("arms") or {}).get(arm, {}).get("items") or []
                contact_sheet = _contact_sheet_data_url(items)
                judgements = []
                for repeat in (1, 2):
                    row = results[(case_id, metric, event_id, arm, repeat)]
                    verdict = html.escape(row["predicted_label"])
                    confidence = html.escape(row.get("confidence") or "")
                    reason = html.escape(row.get("reason") or "")
                    judgements.append(
                        '<div class="avj-judgement">'
                        f'<div class="viz-row"><strong>Repeat {repeat}</strong>'
                        f'<span class="viz-badge">{verdict}</span>'
                        f'<span class="text-small text-muted">confidence {confidence}</span></div>'
                        f'<p>{reason}</p></div>'
                    )
                arm_blocks.append(
                    '<section class="avj-arm">'
                    f'<h3>{html.escape(ARM_LABELS[arm])}</h3>'
                    f'<img src="{contact_sheet}" alt="{html.escape(ARM_LABELS[arm])} '
                    f'evidence for {html.escape(metric)} {html.escape(event_id)}">'
                    f'{"".join(judgements)}</section>'
                )
            events.append(
                '<article class="avj-event">'
                '<div class="avj-event-heading">'
                f'<h2>{html.escape(metric.upper())} · {html.escape(event_id)}</h2>'
                '</div>'
                '<div class="avj-audit-target">'
                f'<div><strong>需要判断:</strong> {html.escape(identity["pair"])}</div>'
                f'<div class="text-small">{html.escape(identity["question"])}</div>'
                f'<div class="text-small text-muted">{html.escape(identity["evidence"])}</div>'
                '</div>'
                f'<div class="avj-arms">{"".join(arm_blocks)}</div>'
                '</article>'
            )
        hidden = "" if scene_index == 0 else " hidden"
        panels.append(
            f'<section class="avj-scene" data-scene-index="{scene_index}"{hidden}>'
            f'<h2>{html.escape(case_id)}</h2>'
            f'{"".join(events)}</section>'
        )

    return f"""<div id="ambiguous-vlm-audit">
  <div class="viz-controls" aria-label="Scene navigation">
    <button type="button" class="btn" id="avj-prev" disabled>Previous scene</button>
    <span id="avj-position"><strong>1 / {len(cases)}</strong> · {html.escape(cases[0])}</span>
    <button type="button" class="btn btn-primary" id="avj-next">Next scene</button>
  </div>
  <p class="text-small text-muted">No binary GT · human audit required · each arm uses the same frozen evidence in both repeats</p>
  {"".join(panels)}
</div>

<style>
#ambiguous-vlm-audit {{
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
}}
#ambiguous-vlm-audit .viz-controls {{
  justify-content: space-between;
}}
#ambiguous-vlm-audit .avj-scene {{
  display: flex;
  flex-direction: column;
  gap: 18px;
}}
#ambiguous-vlm-audit .avj-scene[hidden] {{
  display: none;
}}
#ambiguous-vlm-audit .avj-event {{
  border-top: 1px solid var(--border);
  padding-top: 12px;
}}
#ambiguous-vlm-audit .avj-event-heading {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  flex-wrap: wrap;
}}
#ambiguous-vlm-audit .avj-audit-target {{
  border-left: 3px solid var(--accent);
  margin: 4px 0 12px;
  padding: 7px 10px;
}}
#ambiguous-vlm-audit .avj-arms {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}}
#ambiguous-vlm-audit .avj-arm {{
  min-width: 0;
}}
#ambiguous-vlm-audit img {{
  display: block;
  width: 100%;
  height: auto;
}}
#ambiguous-vlm-audit .avj-judgement {{
  border-top: 1px solid var(--border);
  margin-top: 8px;
  padding-top: 8px;
}}
#ambiguous-vlm-audit .avj-judgement .viz-row {{
  justify-content: flex-start;
}}
#ambiguous-vlm-audit .avj-judgement p {{
  margin-bottom: 0;
}}
@media (max-width: 620px) {{
  #ambiguous-vlm-audit .avj-arms {{
    grid-template-columns: 1fr;
  }}
}}
</style>

<script>
(() => {{
  const root = document.getElementById("ambiguous-vlm-audit");
  const panels = Array.from(root.querySelectorAll(".avj-scene"));
  const labels = {json.dumps(cases, ensure_ascii=False)};
  const previous = root.querySelector("#avj-prev");
  const next = root.querySelector("#avj-next");
  const position = root.querySelector("#avj-position");
  let active = 0;
  const update = () => {{
    panels.forEach((panel, index) => {{
      panel.hidden = index !== active;
    }});
    previous.disabled = active === 0;
    next.disabled = active === panels.length - 1;
    position.innerHTML = `<strong>${{active + 1}} / ${{panels.length}}</strong> · ${{labels[active]}}`;
  }};
  previous.addEventListener("click", () => {{
    if (active > 0) active -= 1;
    update();
  }});
  next.addEventListener("click", () => {{
    if (active < panels.length - 1) active += 1;
    update();
  }});
  update();
}})();
</script>
"""


def _event_identity_context(
    manifest: dict[str, Any],
    object_map: dict[str, str],
    report: dict[str, Any],
) -> dict[str, str]:
    metric = str(manifest["metric"])
    object_ids = [str(value) for value in manifest.get("object_ids") or []]
    metrics = report.get("metrics") or {}

    if metric == "collision":
        if len(object_ids) != 2:
            raise ValueError(f"collision event requires two objects: {manifest}")
        object_a, object_b = object_ids
        record = _find_collision_record(
            (metrics.get("collision") or {}).get("pairs") or [],
            object_a,
            object_b,
        )
        overlap = _millimetres(
            ((record.get("obb_evidence") or {}).get(
                "minimum_overlap_depth_proxy_m"
            ))
        )
        level = str(record.get("evidence_level") or "unknown")
        evidence_parts = [f"detector evidence: {level.upper()}"]
        if overlap is not None:
            evidence_parts.append(f"minimum overlap proxy {overlap}")
        return {
            "pair": f"{_object_label(object_a, object_map)} ↔ "
            f"{_object_label(object_b, object_map)}",
            "question": (
                "判断两者是否存在 physically implausible collision; "
                "正常接触、承载或刻意重叠不应判为 invalid."
            ),
            "evidence": " · ".join(evidence_parts),
        }

    if metric == "oob":
        if len(object_ids) != 1:
            raise ValueError(f"OOB event requires one object: {manifest}")
        object_id = object_ids[0]
        record = _find_object_record(
            (metrics.get("oob") or {}).get("objects") or [],
            object_id,
        )
        planes = [
            str(flag).removesuffix("_oob")
            for flag, active in (record.get("plane_flags") or {}).items()
            if active
        ]
        plane_labels = [f"{plane} room plane" for plane in planes]
        plane_text = " + ".join(plane_labels) or "room boundary"
        return {
            "pair": f"{_object_label(object_id, object_map)} ↔ {plane_text}",
            "question": (
                "判断该物体是否真正越过或不合理穿入这些 room planes; "
                "仅靠墙、贴墙或合理 wall attachment 不应判为 OOB."
            ),
            "evidence": (
                "detector plane flags: "
                + (", ".join(planes) if planes else "none recorded")
            ),
        }

    if metric == "support":
        if len(object_ids) != 1:
            raise ValueError(f"support event requires one subject: {manifest}")
        subject_id = object_ids[0]
        record = _find_object_record(
            (metrics.get("support") or {}).get("objects") or [],
            subject_id,
        )
        support_ids = [
            str(value) for value in record.get("candidate_support_object_ids") or []
        ]
        architecture = record.get("architecture_contact_candidates") or []
        counterparts = [
            _object_label(value, object_map) for value in support_ids
        ]
        counterparts.extend(
            f"{item.get('plane', 'unknown')} room plane"
            for item in architecture
        )
        if not counterparts:
            counterparts.append("candidate support surface")

        evidence_parts: list[str] = []
        gap = _millimetres(record.get("minimum_positive_clearance_m"))
        tolerance = _millimetres(record.get("contact_tolerance_m"))
        if gap is not None:
            evidence_parts.append(f"minimum positive gap {gap}")
        if tolerance is not None:
            evidence_parts.append(f"contact tolerance {tolerance}")
        band = record.get("gap_band")
        if band:
            evidence_parts.append(f"gap band {band}")
        for item in architecture:
            clearance = _millimetres(item.get("signed_clearance_m"))
            description = (
                f"{item.get('plane', 'unknown')} {item.get('mode', 'architecture contact')}"
            )
            if clearance is not None:
                description += f" clearance {clearance}"
            evidence_parts.append(description)

        return {
            "pair": f"{_object_label(subject_id, object_map)} ↔ "
            + " / ".join(counterparts),
            "question": (
                "判断 subject 是否获得 physically plausible support/attachment; "
                "可接受的微小 gap、支撑面接触或 wall attachment 不应误判."
            ),
            "evidence": " · ".join(evidence_parts) or "no detector detail recorded",
        }

    raise ValueError(f"unsupported metric in ambiguous audit: {metric}")


def _find_collision_record(
    records: list[dict[str, Any]],
    object_a: str,
    object_b: str,
) -> dict[str, Any]:
    expected = {object_a, object_b}
    for record in records:
        actual = {str(record.get("object_a")), str(record.get("object_b"))}
        if actual == expected:
            return record
    raise KeyError(f"collision pair not found: {object_a}, {object_b}")


def _find_object_record(
    records: list[dict[str, Any]],
    object_id: str,
) -> dict[str, Any]:
    for record in records:
        if str(record.get("object_id")) == object_id:
            return record
    raise KeyError(f"metric object not found: {object_id}")


def _object_label(object_id: str, object_map: dict[str, str]) -> str:
    return f"{object_map.get(object_id, 'unknown object')} ({object_id})"


def _millimetres(value: Any) -> str | None:
    if value is None:
        return None
    return f"{float(value) * 1000.0:.1f} mm"


def _contact_sheet_data_url(items: list[dict[str, Any]]) -> str:
    if len(items) != 4:
        raise ValueError(f"expected four evidence images, found {len(items)}")
    tile_size = CONTACT_SHEET_TILE_SIZE
    label_height = CONTACT_SHEET_LABEL_HEIGHT
    sheet = Image.new("RGB", (tile_size * 2, (tile_size + label_height) * 2))
    draw = ImageDraw.Draw(sheet)
    try:
        label_font = ImageFont.load_default(size=14)
    except TypeError:
        label_font = ImageFont.load_default()
    for index, item in enumerate(items):
        path = Path(str(item["path"])).expanduser().resolve()
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (tile_size, tile_size), (32, 32, 32))
            offset = (
                (tile_size - image.width) // 2,
                (tile_size - image.height) // 2,
            )
            tile.paste(image, offset)
        column = index % 2
        row = index // 2
        x = column * tile_size
        y = row * (tile_size + label_height)
        sheet.paste(tile, (x, y))
        label = f"{item.get('view_id')} · {item.get('role')}"
        draw.rectangle(
            (x, y + tile_size, x + tile_size, y + tile_size + label_height),
            fill=(32, 32, 32),
        )
        draw.text(
            (x + 6, y + tile_size + 6),
            label[:44],
            fill=(235, 235, 235),
            font=label_font,
        )
    buffer = io.BytesIO()
    sheet.save(buffer, format="WEBP", quality=80, method=6)
    return "data:image/webp;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


if __name__ == "__main__":
    main()
