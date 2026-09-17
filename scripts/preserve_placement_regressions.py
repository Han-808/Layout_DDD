"""Preserve independent, transport-free S100-S102 aggregation replay fixtures.

Does not mutate the source campaign or call a model/renderer. Parsed residual
semantic candidates are retained, never the raw exchange or private bindings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

DROP = set('raw_response raw_request request messages headers api_key authorization request_metadata endpoint '
           'camera_control_audit global_camera_control_audit camera_acquisition_ledger '
           'camera_control_audits functional_prejudgement_evidence functional_group_evidence_window '
           'functional_group_evidence_window_audit functional_group_evidence_initial_window '
           'metric_prompt_context response_schema_audit evidence_handles '
           'vertices triangles faces points samples point_cloud representative_samples'.split())

def clean(value):
    if isinstance(value, dict):
        return {k: clean({'required_checks': v.get('required_checks', [])}
                        if k == 'functional_probe_evidence' and isinstance(v, dict) else v)
                for k, v in value.items()
                if k.lower() not in DROP and not k.endswith('_camera_control_audit')
                and not any(token in k.lower() for token in ('api_key', 'authorization', 'request_metadata'))}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str):
        if value.startswith(('/Users/', '/private/', '/tmp/', 'data:image', 'http://', 'https://')):
            return 'fixture-reference:' + hashlib.sha256(value.encode()).hexdigest()[:16]
        return re.sub(r'/Users/[^\s"\']+', 'fixture-reference:redacted', value)
    return value

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    for scene in ('S100', 'S101', 'S102'):
        evaluation = args.run / 'batch_01' / scene / 'shared/trial_00/evaluation_attempts/attempt_01/evaluation'
        case = evaluation / 'judgement/cases' / scene
        l3 = read(case / 'scene_quality_report.json')
        placement = l3['metrics']['semantic_placement_consistency']
        attempts = placement['residual_global_placement_review']['response_schema_audit']['attempts']
        candidates = [{'attempt': a['attempt'], 'validation_error': a['validation_error'],
                       'candidate': clean(json.loads(a['raw_response']))} for a in attempts]
        # Use the compact envelope for identity + outcome, never load its GB-sized duplicate.
        envelope = read(evaluation / 'evaluation_report.json')
        fixture = {'schema_version': 'placement_complete_scoring_replay_v1', 'case_id': scene,
                   'l3': clean(l3), 'l1': clean(read(case / 'l1_report.json')),
                   'case_manifest': clean(read(case / 'case_run_manifest.json')),
                   'compact_original': clean(envelope), 'residual_candidates': candidates,
                   'scope': 'Complete persisted scoring facts and residual semantic candidates; no images, transport, or live rejudgement.'}
        path = args.destination / (scene + '.json')
        path.write_text(json.dumps(fixture, separators=(',', ':'), allow_nan=False)+'\n')
        receipts.append({'case': scene, 'fixture_sha256': sha(path), 'bytes': path.stat().st_size,
                         'source_hashes': {p.name: sha(p) for p in [case/'scene_quality_report.json', case/'l1_report.json',
                                                                  case/'case_run_manifest.json', evaluation/'evaluation_report.json']}})
        print(json.dumps(receipts[-1]), flush=True)
    (args.destination/'inventory.json').write_text(json.dumps(receipts, indent=2)+'\n')

if __name__ == '__main__':
    main()
