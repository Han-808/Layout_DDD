"""Preserve S100 C-R1 ownership-reference repair and complete scoring inputs.

No source artifacts are modified. Keep parsed model decisions, never raw API
transport or credentials. These inputs support an offline implementation replay.
"""
import argparse
import json
from pathlib import Path

from preserve_placement_regressions import clean, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    case = args.evaluation/'judgement/cases/S100'
    scene = args.evaluation/'dataset/S100/scene/canonical_scene.json'
    l3 = read(case/'scene_quality_report.json')
    placement = l3['metrics']['semantic_placement_consistency']
    failed = [g for g in placement['group_results'] if g['status']=='failed']
    assert len(failed)==1 and failed[0]['group_id']=='group_002'
    attempts = failed[0]['judgement']['response_schema_audit']['attempts']
    assert len(attempts)==2
    fixture = {'schema_version':'placement_reference_repair_replay_v1',
        'case_id':'S100', 'trial_identity':'diagnostics/trial_01/attempt_01',
        'l1':clean(read(case/'l1_report.json')), 'l3':clean(l3),
        'case_manifest':clean(read(case/'case_run_manifest.json')),
        'compact_original':clean(read(args.evaluation/'evaluation_report.json')),
        'scene':clean(read(scene)), 'failed_group':clean(failed[0]),
        'model_candidates':[{'attempt':a['attempt'], 'validation_error':a['validation_error'],
                             'candidate':clean(json.loads(a['raw_response']))} for a in attempts],
        'scope':'Complete scoring inputs plus real initial and repair decisions. Image bytes/transport omitted. No new model judgement.'}
    path = args.destination/'S100_diagnostics_R1.json'
    path.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
    sources = [case/'scene_quality_report.json',case/'l1_report.json',case/'case_run_manifest.json',
               case/'evaluation_report.json',args.evaluation/'evaluation_report.json',scene,
               args.evaluation/'evaluation_receipt.json',args.evaluation/'process.json',
               args.evaluation/'judgement/progress.jsonl']
    inventory = {'fixture_sha256':sha(path),'bytes':path.stat().st_size,
        'source_hashes':{str(p.relative_to(args.evaluation)):sha(p) for p in sources}}
    (args.destination/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps(inventory,indent=2))


if __name__=='__main__': main()
