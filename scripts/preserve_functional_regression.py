"""Preserve a transport-free independent Function acquisition/scoring fixture."""
import argparse
import json
from pathlib import Path
from preserve_placement_regressions import clean, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    case = args.evaluation/'judgement/cases/S100'
    def read(path): return json.loads(path.read_text())
    l3 = read(case/'scene_quality_report.json')
    function = l3['metrics']['functional_consistency']
    failed = [e for g in function['group_results'] for e in g['check_episodes'] if e['status']=='failed']
    assert len(failed)==1 and failed[0]['functional_check_episode_id']=='functional_check_032'
    audit = failed[0]['camera_control_audit']['audit']
    candidates = []
    def collect(value):
        if isinstance(value,dict):
            if isinstance(value.get('raw_response'),str):
                parsed = clean(json.loads(value['raw_response']))
                if parsed not in candidates: candidates.append(parsed)
            for key,item in value.items():
                if key!='raw_response': collect(item)
        elif isinstance(value,list):
            for item in value: collect(item)
    collect(audit)
    progress_path = args.evaluation/'judgement/progress.jsonl'
    progress = [clean(row) for line in progress_path.read_text().splitlines()
                if (row:=json.loads(line)).get('error_type')=='AcquisitionExhausted']
    fixture = {'schema_version':'functional_acquisition_score_replay_v1','case_id':'S100',
        'l3':clean(l3),'l1':clean(read(case/'l1_report.json')),
        'case_manifest':clean(read(case/'case_run_manifest.json')),
        'compact_original':clean(read(args.evaluation/'evaluation_report.json')),
        'failed_episode':clean(failed[0]),'controller_audit':clean(audit),
        # Preserve the contents separately: the generic cross-metric cleaner
        # deliberately compacts nested probe packets to their check roster.
        'failed_probe_evidence':clean(failed[0]['functional_probe_evidence']),
        'stored_judge_candidates':candidates,'acquisition_exhaustion_events':progress,
        'scope':'Stored failure and all eight scoring inputs; any new terminal verdict in tests is explicitly synthetic.'}
    target=args.destination/'S100.json'
    target.write_text(json.dumps(fixture,separators=(',',':'),allow_nan=False)+'\n')
    receipt={'fixture_sha256':sha(target),'bytes':target.stat().st_size,'parsed_candidate_count':len(candidates),
        'source_hashes':{str(p.relative_to(args.evaluation)):sha(p) for p in (
            case/'scene_quality_report.json',case/'l1_report.json',case/'case_run_manifest.json',
            args.evaluation/'evaluation_report.json',args.evaluation/'evaluation_receipt.json',
            args.evaluation/'process.json',progress_path)}}
    (args.destination/'inventory.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__': main()
