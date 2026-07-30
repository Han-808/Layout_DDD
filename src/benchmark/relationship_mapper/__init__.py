"""Relationship intent mapping interfaces for future OOR/OAR relation extraction."""

from benchmark.relationship_mapper.mapper import map_relationships
from benchmark.relationship_mapper.schemas import OAR_RELATION_TYPES, OOR_RELATION_TYPES, relationship_intent_document

__all__ = ["OAR_RELATION_TYPES", "OOR_RELATION_TYPES", "map_relationships", "relationship_intent_document"]
