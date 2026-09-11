RELATIONSHIP_MAPPING_DRAFT_PROMPT = """
Map natural-language placement and relationship intents into finite structural relation categories.
Separate object-object relations from object-architecture relations.
Use only the objects already present in object_plan; do not invent objects.
Do not output pose, coordinates, center, rotation, or deterministic placement.
One natural-language relation may map to multiple finite primitives later, but keep uncertain mappings explicit.
Return JSON only with oor_relations, oar_relations, unsupported_relations, and notes.

OOR types: left, right, in_front, behind, above, below, near, far, contact, on_top_of, within, contains, aligned, parallel, perpendicular, between, ordered, around.
OAR types: on_floor, against_wall, near_wall, at_corner, near_corner, room_center, room_region, along_wall, mounted_on_wall, attached_to_ceiling, hung_from_ceiling.
If an explicit relation matches none of these, assign it to oor_relations or oar_relations from its
explicit endpoints, preserve a concise snake_case type plus its raw wording and IDs, and never discard
or force-map it. The evaluator will send that family-routed claim to a VLM. Use unsupported_relations
only for malformed input whose family or required identities cannot be represented.
""".strip()
