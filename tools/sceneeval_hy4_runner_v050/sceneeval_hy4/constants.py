"""Frozen experiment constants."""

MODEL_ALIAS = "Hy4-T3-A49B-DSA-1M-SFT0730-Opus5"
DEFAULT_ENDPOINT = (
    "http://infer-proxy-yb-test.production.polaris:8000/"
    "openapi/chat/completions"
)

PROMPT_VERSION = "sceneeval-abstract-layout-v4-room-geometry-positive-coordinates"
LAYOUT_SCHEMA_VERSION = "sceneeval-abstract-layout-v2-rooms-positive-coordinates"
RUN_MANIFEST_VERSION = 6
ARTIFACT_FORMAT_VERSION = 6

EXPECTED_SCENE_IDS = tuple(range(100))
DEFAULT_RETRY_DELAY_SECONDS = 30.0
MIN_VISIBLE_OUTPUT_TOKENS = 10
