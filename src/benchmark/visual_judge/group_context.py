"""Judge-facing grouping context, separate from grouping's routing audit.

Formation edges explain how evidence groups were assembled; they are not
metric observations or verdicts. Keep them in the original grouping artifact,
not repeated in every metric's mandatory prompt. Never truncate member IDs,
geometry, semantic context, ownership, or required metric checks.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

SCHEMA = 'judge_group_context_v1'
ROUTING_AUDIT_KEYS = frozenset({'formation_edges', 'edge_reasons'})


def project_group(group: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(group, dict):
        raise TypeError('Grouping context must be an object')
    omitted = {k: v for k, v in group.items() if k in ROUTING_AUDIT_KEYS}
    if not omitted:
        return deepcopy(group)
    result = {k: deepcopy(v) for k, v in group.items() if k not in ROUTING_AUDIT_KEYS}
    result['_routing_audit_projection'] = {
        'schema_version': SCHEMA,
        'omitted_keys': sorted(omitted),
        'source_group_sha256': hashlib.sha256(json.dumps(
            group, sort_keys=True, ensure_ascii=True, allow_nan=False,
            separators=(',', ':'),
        ).encode()).hexdigest(),
        'source': 'original grouping artifact; routing explanations are not metric evidence',
        'member_ids_and_nonrouting_fields_preserved': True,
    }
    return result


def project_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [project_group(group) for group in groups]
