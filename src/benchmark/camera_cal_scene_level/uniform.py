"""One protocol and explicit dependency graph for all Model floorplan modes.

No mode-specific evaluator/fallback, no process-global monkeypatch, no proxy
startup, and no automatic rerun, generation, publication or baseline promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / 'configs/runners/model_floorplan_unified_v1.json'


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def source_files(root: Path = ROOT) -> list[str]:
    paths = [p for directory in ('src', 'configs', 'schemas')
             for p in (root/directory).rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts)
             and p.suffix != '.pyc' and p.name != '.DS_Store']
    paths += [root/'scripts/run_uniform_model_evaluation.py', root/'scripts/freeze_uniform_model_evaluator.py',
              root/'scripts/replay_s100_regressions.py', root/'scripts/replay_placement_scoring.py',
              root/'scripts/preserve_placement_regressions.py', root/'pyproject.toml']
    paths += [root/'scripts/preserve_functional_regression.py', root/'scripts/replay_functional_scoring.py']
    return sorted(str(p.relative_to(root)) for p in paths)


def tree_digest(files: dict[str, str]) -> str:
    return hashlib.sha256(''.join(rel+'\0'+sha+'\n' for rel, sha in sorted(files.items())).encode()).hexdigest()


def protocol() -> dict:
    return read(PROTOCOL_PATH)


def verify_source(manifest: Path | None, *, required: bool) -> dict:
    if manifest is None:
        if required:
            raise ValueError('Live evaluation requires a sealed release manifest')
        return {'sealed': False, 'source_root': str(ROOT)}
    release = read(manifest)
    if release.get('schema_version') != 'uniform_model_evaluator_release_v1':
        raise ValueError('Unknown release contract')
    if Path(release['source_root']).resolve() != ROOT:
        raise ValueError('Release does not own the imported source root')
    if release.get('protocol_sha256') != digest(PROTOCOL_PATH):
        raise ValueError('Scientific protocol changed')
    if set(source_files()) != set(release['files']):
        raise ValueError('Release file inventory changed')
    if tree_digest(release['files']) != release.get('source_tree_sha256'):
        raise ValueError('Release tree digest is inconsistent')
    for rel, expected in release['files'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Unsafe release path')
        if not (ROOT/rel).resolve().is_relative_to(ROOT) or (ROOT/rel).is_symlink():
            raise ValueError('Release source escapes its root')
        if digest(ROOT / rel) != expected:
            raise ValueError('Release source changed: ' + rel)
    actual_python = {str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')}
    if not actual_python <= set(release['files']):
        raise ValueError('Unpinned Python source in release')
    for name, module in tuple(sys.modules.items()):
        if name == 'benchmark' or name.startswith('benchmark.'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(ROOT/'src'):
                raise ValueError('Mixed evaluator imports: ' + name)
    return {'sealed': True, 'source_root': str(ROOT),
            'release_manifest_sha256': digest(manifest),
            'source_tree_sha256': release['source_tree_sha256']}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', required=True, choices=('open-space', 'multi-room', 'non-rect'))
    p.add_argument('--dataset-root', required=True, type=Path)
    p.add_argument('--case-id', action='append', default=[])
    p.add_argument('--output-root', type=Path)
    p.add_argument('--release-manifest', type=Path)
    p.add_argument('--input-manifest', type=Path, help='Reviewed --check output; required for a live run')
    p.add_argument('--blender-bin', type=Path, default=Path('/Applications/Blender.app/Contents/MacOS/Blender'))
    p.add_argument('--max-workers', type=int, default=1)
    p.add_argument('--run', action='store_true', help='Explicit live run; default is input/source check only')
    return p


def native_argv(args: argparse.Namespace) -> list[str]:
    # All scientific options are fixed; mode is deliberately absent.
    result = ['--dataset-root', str(args.dataset_root.resolve()),
              '--output-root', str(args.output_root.resolve()),
              '--grouping-config', str(ROOT/'configs/grouping/vlm_visual_evidence_scope_v2.yaml'),
              '--evidence-resolution-policy', protocol()['evidence_resolution_policy'],
              '--max-workers', str(args.max_workers), '--no-resume',
              '--deduction-multiplier', '2', '--blender-bin', str(args.blender_bin),
              '--blender-timeout-seconds', '1800', '--render-width', '768', '--render-height', '768',
              '--preview-width', '256', '--preview-height', '256', '--terminal-progress']
    for case in args.case_id:
        result += ['--case-id', case]
    return result


def dependencies(argv: list[str]):
    """Inject policy through public dependency seams, including parallel cases."""
    from benchmark.camera_cal_scene_level import composition, planning, adapters, case_runtime, scheduling
    settings = protocol()
    base_factory = composition._adapter_factories()

    def model_config(route, *, role, **kwargs):
        if route.get('model') != settings['judge_model']:
            raise ValueError('Uniform protocol requires its fixed Judge model')
        config = planning.model_config(route, role=role, **kwargs)
        if role == 'judge':
            config.update(evidence_resolution_policy=settings['evidence_resolution_policy'],
                          terminal_evidence_policy=settings['terminal_evidence_policy'],
                          max_context_chars=settings['max_context_chars'])
        return config

    def build_adapters(**kwargs):
        return adapters.build_adapters(**kwargs, factories=replace(base_factory, model_config=model_config))

    def run_case(**kwargs):
        deps = composition.case_runtime_dependencies()
        original_evaluate = deps.external.run_evaluate

        def evaluate(**evaluation_kwargs):
            from benchmark.evaluator.judgement_coverage import apply_report
            result = original_evaluate(**evaluation_kwargs)
            return apply_report(result)

        deps = replace(deps, external=replace(deps.external, adapter_builder=build_adapters,
                                              run_evaluate=evaluate))
        return case_runtime.run_case_impl(**kwargs, deps=deps)

    def run_parallel(**kwargs):
        return scheduling.run_cases_parallel(**kwargs, run_case_fn=run_case,
                failure_recorder=composition._record_case_failure,
                cancellation_recorder=composition._record_case_cancellation)

    deps = composition.orchestrator_dependencies(argv)
    original_route = deps.planning.effective_model_route

    def fixed_route():
        route = original_route()
        if route['model'] != settings['judge_model']:
            raise ValueError('JUDGE_MODEL differs from the frozen Model protocol')
        route.update(max_retries=2, min_request_interval_seconds=1.0)
        return route

    return replace(deps, planning=replace(deps.planning, effective_model_route=fixed_route),
                   execution=replace(deps.execution, run_case=run_case, run_cases_parallel=run_parallel))


def inspect_inputs(args: argparse.Namespace) -> list[dict[str, Any]]:
    from benchmark.camera_cal_scene_level import discovery
    if args.max_workers < 1:
        raise ValueError('max-workers must be positive')
    dataset = args.dataset_root.resolve()
    cases = discovery.discover_cases(dataset, case_ids=args.case_id)
    # Native discovery skips unready cases; an all-cases run must never silently
    # shrink its planned population. Explicit selections remain auditable.
    manifests = {p.parent.name for p in dataset.glob('*/case_manifest.json')}
    if not args.case_id and {c['case_id'] for c in cases} != manifests:
        raise ValueError('Dataset contains unready cases; supply an explicit audited selection')
    result=[]
    for case in cases:
        root = Path(case['case_root'])
        manifest = read(root/'case_manifest.json')
        paths = discovery.case_paths(root, manifest)
        missing = [key for key, path in paths.items() if not path.is_file()]
        if missing:
            raise ValueError('Incomplete prepared case '+case['case_id']+': '+str(missing))
        input_hashes = {key: digest(path) for key, path in paths.items()}
        aliases = {'canonical_scene': 'scene', 'evidence_identity': 'identity',
                   'evidence_top': 'top', 'evidence_perspective': 'perspective'}
        for field, expected in (manifest.get('critical_artifact_hashes') or {}).items():
            key = aliases.get(field, field)
            if key not in input_hashes or input_hashes[key] != expected:
                raise ValueError('Prepared critical artifact identity mismatch: '+field)
        scene = read(paths['scene'])
        from benchmark.non_rectangular.geometry import polygon_geometry_from_scene
        polygon = polygon_geometry_from_scene(scene)
        if args.mode == 'non-rect' and polygon is None:
            raise ValueError('Non-rect inputs require authoritative polygon geometry; mode cannot fabricate it')
        from benchmark.scene_io.validate import validate_generated_scene
        validate_generated_scene(scene)
        # Resolve geometric validity before any output/model/render. Selection
        # labels never choose another scorer; geometry must be self-describing.
        from benchmark.evaluator.generic_validity.oob import _resolve_room
        if _resolve_room(scene) is None:
            raise ValueError('Unsupported or incomplete room geometry')
        ids = [o['id'] for o in scene['objects']]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Canonical object IDs must be nonempty and unique')
        from benchmark.evaluator.generic_validity.mesh_geometry import load_collision_geometry_manifest
        geometry = load_collision_geometry_manifest(paths['collision_geometry'])
        if set(geometry['objects']) != set(ids):
            raise ValueError('Collision geometry ownership differs from canonical scene')
        meshes = {}
        limitations = []
        for object_id, entry in geometry['objects'].items():
            if entry.get('complete') is not True:
                limitations.append({'object_id': object_id, 'reason': 'incomplete_original_geometry'})
            rel = entry.get('geometry_path')
            if rel:
                path = (paths['collision_geometry'].parent / rel).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError('Geometry path escapes the prepared case')
                if path.is_file():
                    meshes[object_id] = {'path': str(path), 'sha256': digest(path)}
                elif entry.get('complete'):
                    raise ValueError('Declared complete geometry file is missing')
        result.append({'case_id':case['case_id'], 'case_root':str(root),
                       'case_manifest_sha256':digest(root/'case_manifest.json'),
                       'input_hashes':input_hashes,
                       'mesh_files': meshes, 'object_count': len(ids), 'input_limitations': limitations})
    return result


def main(argv: list[str] | None = None) -> None:
    args=parser().parse_args(argv)
    identity=verify_source(args.release_manifest,required=args.run)
    cases=inspect_inputs(args)
    receipt={'schema_version':'uniform_model_evaluator_run_v1', 'mode':args.mode,
             'protocol':protocol(), 'protocol_sha256':digest(PROTOCOL_PATH),
             'identity':identity, 'cases':cases, 'live_run':False}
    if not args.run:
        print(json.dumps(receipt,indent=2))
        return
    if args.input_manifest is None:
        raise ValueError('Live evaluation requires a reviewed input manifest')
    approved = read(args.input_manifest)
    if approved.get('schema_version') != receipt['schema_version'] or approved.get('live_run') is not False:
        raise ValueError('Reviewed input manifest must be a pre-run check receipt')
    for field in ('mode', 'protocol_sha256', 'identity', 'cases'):
        if approved.get(field) != receipt[field]:
            raise ValueError('Input/source receipt differs from reviewed manifest: '+field)
    receipt['reviewed_input_manifest_sha256'] = digest(args.input_manifest)
    if args.output_root is None or args.output_root.exists():
        raise ValueError('A new output directory is required; no overwrite/resume')
    output = args.output_root.resolve()
    dataset = args.dataset_root.resolve()
    if output.is_relative_to(dataset) or dataset.is_relative_to(output) or output.is_relative_to(ROOT):
        raise ValueError('Output must be separate from input dataset and frozen source')
    if not args.blender_bin.is_file():
        raise FileNotFoundError(args.blender_bin)
    if os.environ.get('JUDGE_MODEL') != protocol()['judge_model']:
        raise ValueError('Set the frozen JUDGE_MODEL before launch')
    args.output_root.mkdir(parents=True, mode=0o700)
    receipt['live_run']=True
    (args.output_root/'uniform_evaluator_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    from benchmark.camera_cal_scene_level.orchestrator import run_main
    run_main(deps=dependencies(native_argv(args)))


if __name__ == '__main__':
    main()
