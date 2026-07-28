from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageFilter


DEFAULT_BAND_WIDTH_PX = 7
DEFAULT_OUTLINE_WIDTH_PX = 2
DEFAULT_BAND_ALPHA = 0.30
DEFAULT_OUTLINE_ALPHA = 0.95


def compose_segmentation_contour_highlight(
    *,
    rgb_path: str | Path,
    targets: Iterable[dict[str, Any]],
    out_path: str | Path,
    band_width_px: int = DEFAULT_BAND_WIDTH_PX,
    outline_width_px: int = DEFAULT_OUTLINE_WIDTH_PX,
    band_alpha: float = DEFAULT_BAND_ALPHA,
    outline_alpha: float = DEFAULT_OUTLINE_ALPHA,
) -> dict[str, Any]:
    """Add an exterior color band and contour from Blender identity masks.

    The target interior is deliberately left as the original RGB render.  Only
    pixels outside the visible 2D segmentation silhouette are annotated:

    ``target | semi-transparent color band | opaque outer contour``.

    The masks may be lower resolution than the RGB render; nearest-neighbour
    resizing preserves their categorical object-ID semantics.
    """

    source_path = Path(rgb_path).expanduser().resolve()
    destination = Path(out_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"RGB image does not exist: {source_path}")
    band_width = _non_negative_int(band_width_px, "band_width_px")
    outline_width = _non_negative_int(outline_width_px, "outline_width_px")
    if band_width == 0 and outline_width == 0:
        raise ValueError("at least one of band_width_px or outline_width_px must be positive")
    band_opacity = _unit_float(band_alpha, "band_alpha")
    outline_opacity = _unit_float(outline_alpha, "outline_alpha")

    with Image.open(source_path) as opened:
        composite = opened.convert("RGBA")
    image_size = composite.size
    target_records: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        if not isinstance(target, dict):
            raise TypeError(f"targets[{index}] must be an object")
        target_id = str(target.get("id") or f"target_{index}")
        mask_path = Path(str(target.get("mask_path") or "")).expanduser().resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(f"segmentation mask for {target_id} does not exist: {mask_path}")
        color = _rgb8(target.get("color"))
        mask = _load_binary_mask(mask_path, image_size)
        visible_pixels = _foreground_pixels(mask)

        band_outer = _dilate(mask, band_width)
        band_mask = ImageChops.subtract(band_outer, mask)
        outline_outer = _dilate(mask, band_width + outline_width)
        outline_mask = ImageChops.subtract(outline_outer, band_outer)

        if band_width and band_opacity:
            composite = _composite_color(
                composite,
                color,
                band_mask.point(lambda value: round(value * band_opacity)),
            )
        if outline_width and outline_opacity:
            composite = _composite_color(
                composite,
                color,
                outline_mask.point(lambda value: round(value * outline_opacity)),
            )
        target_records.append(
            {
                "id": target_id,
                "mask_path": str(mask_path),
                "mask_sha256": _sha256_file(mask_path),
                "source_mask_size": list(_image_size(mask_path)),
                "composite_mask_size": list(image_size),
                "visible_pixels_at_composite_resolution": visible_pixels,
                "color_rgb8": list(color),
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    composite.convert("RGB").save(destination, format="PNG")
    return {
        "schema_version": "segmentation_contour_highlight_v1",
        "source_rgb": str(source_path),
        "source_rgb_sha256": _sha256_file(source_path),
        "output_path": str(destination),
        "output_sha256": _sha256_file(destination),
        "image_size": list(image_size),
        "target_interior_policy": "preserve_raw_rgb",
        "segmentation_source": "blender_visible_object_identity_mask",
        "band_width_px": band_width,
        "outline_width_px": outline_width,
        "band_alpha": band_opacity,
        "outline_alpha": outline_opacity,
        "targets": target_records,
    }


def compose_segmentation_contour_manifest(
    *,
    rgb_manifest: dict[str, Any],
    mask_manifest: dict[str, Any],
    overlay_spec: dict[str, Any],
    out_dir: str | Path,
    band_width_px: int = DEFAULT_BAND_WIDTH_PX,
    outline_width_px: int = DEFAULT_OUTLINE_WIDTH_PX,
    band_alpha: float = DEFAULT_BAND_ALPHA,
    outline_alpha: float = DEFAULT_OUTLINE_ALPHA,
) -> dict[str, Any]:
    """Compose every same-pose RGB/mask pair and write an auditable manifest."""

    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    rgb_views = _rgb_views(rgb_manifest)
    mask_views = {
        str(view.get("id")): view
        for view in mask_manifest.get("views", [])
        if isinstance(view, dict) and view.get("id") is not None
    }
    colors = {
        str(target.get("id")): target.get("color")
        for target in overlay_spec.get("targets", [])
        if isinstance(target, dict) and target.get("id") is not None
    }
    if not colors:
        for key in ("object_a", "object_b"):
            target = overlay_spec.get(key)
            if isinstance(target, dict) and target.get("id") is not None:
                colors[str(target["id"])] = target.get("color")
    if not colors:
        raise ValueError("overlay_spec does not define any target IDs/colors")

    outputs: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for index, rgb_view in enumerate(rgb_views):
        view_id = str(rgb_view.get("id") or "")
        rgb_path = rgb_view.get("path")
        mask_view = mask_views.get(view_id)
        if not view_id or not rgb_path or not isinstance(mask_view, dict):
            skipped.append({"view_id": view_id, "reason": "missing_same_pose_rgb_or_mask_record"})
            continue
        mask_targets = mask_view.get("targets")
        if not isinstance(mask_targets, dict):
            skipped.append({"view_id": view_id, "reason": "mask_record_has_no_targets"})
            continue
        targets = []
        missing = []
        for target_id, color in colors.items():
            target_mask = mask_targets.get(target_id)
            mask_path = target_mask.get("mask_path") if isinstance(target_mask, dict) else None
            if not mask_path:
                missing.append(target_id)
                continue
            targets.append({"id": target_id, "color": color, "mask_path": mask_path})
        if missing:
            skipped.append(
                {
                    "view_id": view_id,
                    "reason": f"missing_target_masks:{','.join(missing)}",
                }
            )
            continue
        output_path = destination / f"contour_{index:02d}_{_safe_name(view_id)}.png"
        record = compose_segmentation_contour_highlight(
            rgb_path=rgb_path,
            targets=targets,
            out_path=output_path,
            band_width_px=band_width_px,
            outline_width_px=outline_width_px,
            band_alpha=band_alpha,
            outline_alpha=outline_alpha,
        )
        record["view_id"] = view_id
        record["pose"] = rgb_view.get("pose")
        outputs.append(record)

    if not outputs:
        raise ValueError("no same-pose RGB/mask records could be composed")
    manifest = {
        "schema_version": "segmentation_contour_highlight_manifest_v1",
        "role": "metric_segmentation_contour_highlight",
        "source_policy": "raw_rgb_plus_blender_visible_object_identity_masks",
        "target_interior_policy": "preserve_raw_rgb",
        "band_width_px": int(band_width_px),
        "outline_width_px": int(outline_width_px),
        "band_alpha": float(band_alpha),
        "outline_alpha": float(outline_alpha),
        "views": outputs,
        "skipped_views": skipped,
    }
    manifest_path = destination / "segmentation_contour_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _rgb_views(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("rgb_views", "views"):
        values = manifest.get(key)
        if isinstance(values, list):
            return [value for value in values if isinstance(value, dict)]
    return []


def _load_binary_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as opened:
        mask = opened.convert("L")
    if mask.size != size:
        mask = mask.resize(size, resample=Image.Resampling.NEAREST)
    return mask.point(lambda value: 255 if value >= 128 else 0)


def _dilate(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.copy()
    return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))


def _composite_color(base: Image.Image, color: tuple[int, int, int], alpha: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", base.size, (*color, 255))
    layer.putalpha(alpha)
    return Image.alpha_composite(base, layer)


def _rgb8(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"target color must contain at least three channels, got {value!r}")
    channels = [float(channel) for channel in value[:3]]
    if all(0.0 <= channel <= 1.0 for channel in channels):
        channels = [channel * 255.0 for channel in channels]
    if any(channel < 0.0 or channel > 255.0 for channel in channels):
        raise ValueError(f"target color channels must be in [0, 1] or [0, 255], got {value!r}")
    return tuple(round(channel) for channel in channels)


def _foreground_pixels(mask: Image.Image) -> int:
    histogram = mask.histogram()
    return int(sum(histogram[128:]))


def _unit_float(value: Any, label: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {parsed}")
    return parsed


def _non_negative_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative, got {parsed}")
    return parsed


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as opened:
        return opened.size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
