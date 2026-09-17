"""Seal an independent, hash-verified source copy. Never evaluates or publishes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from benchmark.camera_cal_scene_level.uniform import source_files, digest, tree_digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--validation-report', required=True, type=Path)
    args = parser.parse_args(argv)
    destination = args.destination.resolve()
    if destination.exists() or destination.is_relative_to(ROOT) or ROOT.is_relative_to(destination):
        raise ValueError('Freeze requires a new, independent destination')
    validation = json.loads(args.validation_report.read_text())
    if validation.get('status') != 'offline_acceptance_passed':
        raise ValueError('An explicit passing offline acceptance report is required')
    files = {rel: digest(ROOT/rel) for rel in source_files()}
    if validation.get('source_tree_sha256') != tree_digest(files):
        raise ValueError('Source changed after offline acceptance')
    for rel in files:
        if (ROOT/rel).is_symlink() or not (ROOT/rel).resolve().is_relative_to(ROOT):
            raise ValueError('Cannot freeze symlink/external source')
    destination.mkdir(parents=True)
    for rel, expected in files.items():
        target = destination/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, target)
        if digest(target) != expected:
            raise ValueError('Source changed while freezing: '+rel)
    manifest = {'schema_version': 'uniform_model_evaluator_release_v1',
        'source_root': str(destination), 'source_tree_sha256': tree_digest(files), 'files': files,
        'protocol_sha256': files['configs/runners/model_floorplan_unified_v1.json'],
        'created_at': datetime.now(timezone.utc).isoformat(),
        'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'parent_release_sha256': 'b1aad3d6cda533977ee251093f26e92fe9b9494d791bdc5a6e77ae8ac8d65de4',
        'validation_report_sha256': digest(args.validation_report),
        'runtime': {'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name)
            for name in ('numpy', 'shapely', 'Pillow', 'PyYAML', 'jsonschema', 'networkx')}},
        'baseline_promoted': False, 'live_evaluation_started': False,
        'performance_accepted': False, 'historical_results_relabelled': False}
    path = destination/'release_manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copyfile(args.validation_report, destination/'offline_acceptance.json')
    for rel in files:
        (destination/rel).chmod(0o444)
    path.chmod(0o444)
    print(json.dumps({'manifest': str(path), 'sha256': digest(path),
                      'source_tree_sha256': manifest['source_tree_sha256'], 'files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
