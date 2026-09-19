"""Resolve exact typed-defect references, without changing a model verdict.

Repeated findings stay owned by the typed component. Unknown IDs and
conflicting identities remain errors; no missing observation is defaulted.
"""
from copy import deepcopy


def resolve_typed_references(value: dict, context: dict | None) -> tuple[dict, list[dict]]:
    result = deepcopy(value)
    context = context or {}
    checks = {c['check_id']: c for c in context.get('typed_checks') or []}
    prior = {d.get('check_id'): d for d in context.get('typed_defects') or []}
    referenced = {}
    for defect in result.get('defects') or []:
        key = defect.get('check_id')
        if key not in checks or key not in prior:
            continue
        check, original = checks[key], prior[key]
        if check.get('conclusion') != 'invalid':
            raise ValueError('residual typed reference lacks a confirmed invalid check')
        if (defect.get('target_ids') != [check['subject_id']]
                or original.get('target_ids') != [check['subject_id']]
                or defect.get('check_type') != check['check_type']
                or original.get('check_type') != check['check_type']
                or 'context_ids' in defect and sorted(defect['context_ids']) != sorted(check['context_ids'])):
            raise ValueError('residual typed reference conflicts with its exact claim identity')
        if result.get('verdict') != 'invalid' or result.get('evidence_status') != 'sufficient':
            raise ValueError('residual typed reference requires an explicit final invalid observation')
        referenced[key] = deepcopy(check)
        rows = result.setdefault('placement_check_results', [])
        if not any(r.get('check_id') == key for r in rows):
            # The model explicitly reported this invalid defect. These redundant
            # linkage fields come from its exact ID, not an inferred judgement.
            rows.append({'check_id': key, 'subject_id': check['subject_id'],
                         'context_ids': deepcopy(check['context_ids']),
                         'observation_status': 'observed', 'conclusion': 'invalid',
                         'reason': defect.get('reason')})
    return result, list(referenced.values())


def repeated_typed_ids(value: dict, context: dict | None) -> set[str]:
    _, checks = resolve_typed_references(value, context)
    return {c['check_id'] for c in checks}
