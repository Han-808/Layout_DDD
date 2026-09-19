"""Preserve independent Function cross-group acquisition and scoring inputs."""
import argparse
import json
from pathlib import Path
from preserve_placement_regressions import clean, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    case = args.evaluation/'judgement/cases/S100'
    def read(path): return json.loads(path.read_text())
    l3 = read(case/'scene_quality_report.json')
    function = l3['metrics']['functional_consistency']
    failed = [r for r in function['cross_group_relation_results'] if r['status']=='failed']
    assert len(failed)==1 and failed[0]['relation_id']=='functional_correspondence_02'
    audit = failed[0]['camera_control_audit']['audit']
    probe = audit['judge_request']['context']['functional_probe_evidence']
    fixture = {'schema_version':'crossgroup_acquisition_score_replay_v1','case_id':'S100',
        'l3':clean(l3),'l1':clean(read(case/'l1_report.json')),
        'case_manifest':clean(read(case/'case_run_manifest.json')),
        'compact_original':clean(read(args.evaluation/'evaluation_report.json')),
        'failed_relation':clean(failed[0]),'controller_audit':clean(audit),
        'failed_probe_evidence':clean(probe),
        'acquisition_audit':clean(function['functional_probe_acquisition']),
        'scope':'All scoring inputs and recorded typed acquisition outcomes; no historic terminal response exists. Test terminal decisions must be synthetic.'}
    target=args.destination/'S100.json'
    target.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
    receipt={'fixture_sha256':sha(target),'bytes':target.stat().st_size,
        'source_hashes':{str(p.relative_to(args.evaluation)):sha(p) for p in (
            case/'scene_quality_report.json',case/'l1_report.json',case/'case_run_manifest.json',
            args.evaluation/'evaluation_report.json',args.evaluation/'evaluation_receipt.json',
            args.evaluation/'process.json',args.evaluation/'judgement/progress.jsonl')}}
    (args.destination/'inventory.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
