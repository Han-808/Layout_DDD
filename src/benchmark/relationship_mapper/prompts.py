RELATIONSHIP_MAPPING_DRAFT_PROMPT = """
Map natural-language placement and relationship intents into finite structural relation categories.
Separate object-object relations from object-architecture relations.
Use only the objects already present in object_plan; do not invent objects.
Do not output pose, coordinates, center, rotation, or deterministic placement.
One natural-language relation may map to multiple finite primitives later, but keep uncertain mappings explicit.
Return JSON only with oor_relations, oar_relations, unsupported_relations, and notes.

OOR types: near, left, right, in_front, behind, above, below, aligned_with, contact, face_to, within, out_of, null.
OAR types: on_floor, against_wall, near_wall, below_wall, at_corner, null.
""".strip()
