"""Instance-scoped orchestration for the frozen two-stage generator.

The unchanged Stage A -> retrieval -> Stage C contract is documented in
``docs/generation_transport_compatibility.md``.  This orchestrator delegates
those semantics to the loaded frozen core and supplies only a route instance,
ordered run policy, safe progress, and write-once run-level artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from benchmark.scene_generation.frozen_two_stage.spec import (
    GenerationRunSpec,
    thaw_json,
)


SafeProgress = Callable[[Mapping[str, Any]], None]

_REQUIRED_CORE_ATTRIBUTES = (
    "DEFAULT_STAGE_A_PROMPT",
    "DEFAULT_STAGE_C_PROMPT",
    "initialize_run",
    "run_case",
    "utc_now",
    "write_json_exclusive",
)


class FrozenTwoStageOrchestrator:
    """Run a loaded frozen core through one explicit provider route instance."""

    def __init__(self, core: Any, provider_route: Any) -> None:
        missing = [name for name in _REQUIRED_CORE_ATTRIBUTES if not hasattr(core, name)]
        if missing:
            raise TypeError(f"frozen core is missing required attributes: {missing}")
        route_key = getattr(provider_route, "key", None)
        if not isinstance(route_key, str) or not route_key.strip():
            raise TypeError("provider_route.key must be a non-empty string")
        self._core = core
        self._provider_route = provider_route

    @property
    def provider_route(self) -> Any:
        """Return the instance-scoped route without mutating the frozen core."""

        return self._provider_route

    def run(
        self,
        *,
        spec: GenerationRunSpec,
        model: Any,
        briefs: Sequence[Mapping[str, Any]],
        retriever: Any,
        progress: SafeProgress | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """Run selected briefs exactly once and write a terminal summary.

        ``briefs`` must already be selected and must exactly match the ordered
        IDs in ``spec``.  This prevents a set-based compatibility shim from
        silently changing the caller's requested order.
        """

        if not isinstance(spec, GenerationRunSpec):
            raise TypeError("spec must be a GenerationRunSpec")
        self._validate_route_and_model(spec, model)
        selected_briefs = tuple(briefs)
        actual_ids = self._validate_briefs(selected_briefs)
        if actual_ids != spec.ordered_brief_ids:
            raise ValueError(
                "brief order/set mismatch: "
                f"expected={spec.ordered_brief_ids} actual={actual_ids}"
            )

        layout = spec.artifact_layout
        assert layout is not None  # normalized by GenerationRunSpec
        layout.require_fresh_output()
        self._emit(
            progress,
            event="run_starting",
            requested_briefs=len(selected_briefs),
        )
        self._core.initialize_run(
            output_root=spec.output_root,
            model=model,
            briefs_path=spec.briefs_path,
            models_path=spec.models_path,
            retriever=retriever,
            source_manifest=(
                None
                if spec.source_manifest is None
                else thaw_json(spec.source_manifest)
            ),
        )
        layout.verify_initialized()
        layout.write_execution_policy(
            self._core.write_json_exclusive,
            thaw_json(spec.execution_policy),
        )

        stage_a_prompt = self._core.DEFAULT_STAGE_A_PROMPT.read_text(
            encoding="utf-8"
        )
        stage_c_prompt = self._core.DEFAULT_STAGE_C_PROMPT.read_text(
            encoding="utf-8"
        )
        results: list[dict[str, Any]] = []
        stopped = False
        for index, brief in enumerate(selected_briefs, start=1):
            brief_id = actual_ids[index - 1]
            self._emit(
                progress,
                event="case_starting",
                brief_id=brief_id,
                case_index=index,
                requested_briefs=len(selected_briefs),
            )
            result = self._core.run_case(
                output_root=spec.output_root,
                model=model,
                brief=brief,
                retriever=retriever,
                stage_a_prompt=stage_a_prompt,
                stage_c_prompt=stage_c_prompt,
                provider_route=self._provider_route,
                retry_policy=spec.retry_policy,
            )
            result = self._validate_result(result, brief_id=brief_id)
            results.append(result)

            failed = result["status"] != "complete"
            stop_requested = bool(result["stop_batch"]) or (
                failed and not spec.retry_policy.continue_after_case_failure
            )
            continued = not stop_requested and index < len(selected_briefs)
            self._emit(
                progress,
                event="case_terminal",
                brief_id=brief_id,
                status=result["status"],
                eligible=bool(result["eligible_for_strict_one_shot_evaluation"]),
                stop_batch=bool(result["stop_batch"]),
                continued_after_case=continued,
            )
            if stop_requested:
                stopped = True
                break

        summary = self._build_summary(spec=spec, model=model, results=results, stopped=stopped)
        layout.write_summary(self._core.write_json_exclusive, summary)
        self._emit(
            progress,
            event="run_terminal",
            requested_briefs=summary["requested_briefs"],
            processed_briefs=summary["processed_briefs"],
            complete=summary["complete"],
            failed=summary["failed"],
            eligible=summary["eligible"],
            stopped_early=summary["stopped_early"],
        )
        return summary, stopped

    def _validate_route_and_model(self, spec: GenerationRunSpec, model: Any) -> None:
        route_key = self._provider_route.key
        if route_key != spec.provider_key:
            raise ValueError(
                f"provider route mismatch: expected={spec.provider_key!r} "
                f"actual={route_key!r}"
            )
        if getattr(model, "key", None) != spec.model_key:
            raise ValueError(
                f"model key mismatch: expected={spec.model_key!r} "
                f"actual={getattr(model, 'key', None)!r}"
            )
        if getattr(model, "wire_model", None) != spec.wire_model:
            raise ValueError(
                f"wire model mismatch: expected={spec.wire_model!r} "
                f"actual={getattr(model, 'wire_model', None)!r}"
            )
        actual_retries = getattr(model, "max_infrastructure_retries", None)
        if actual_retries != spec.expected_max_infrastructure_retries:
            raise ValueError(
                "model retry mismatch: "
                f"expected={spec.expected_max_infrastructure_retries} "
                f"actual={actual_retries}"
            )
        actual_delay = getattr(model, "retry_delay_seconds", None)
        if actual_delay is None or float(actual_delay) != spec.retry_policy.retry_delay_seconds:
            raise ValueError(
                "model retry delay mismatch: "
                f"expected={spec.retry_policy.retry_delay_seconds} "
                f"actual={actual_delay}"
            )

    @staticmethod
    def _validate_briefs(
        briefs: Sequence[Mapping[str, Any]],
    ) -> tuple[str, ...]:
        ids: list[str] = []
        for index, brief in enumerate(briefs):
            if not isinstance(brief, Mapping):
                raise TypeError(f"brief at index {index} must be a mapping")
            brief_id = brief.get("brief_id")
            if not isinstance(brief_id, str) or not brief_id:
                raise ValueError(f"brief at index {index} has invalid brief_id")
            ids.append(brief_id)
        if len(ids) != len(set(ids)):
            raise ValueError("briefs must not contain duplicate brief IDs")
        return tuple(ids)

    @staticmethod
    def _validate_result(result: Any, *, brief_id: str) -> dict[str, Any]:
        if not isinstance(result, Mapping):
            raise TypeError(f"case result for {brief_id} must be a mapping")
        normalized = dict(result)
        if normalized.get("brief_id") != brief_id:
            raise ValueError(f"case result brief identity mismatch for {brief_id}")
        status = normalized.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"case result for {brief_id} has invalid status")
        if not isinstance(normalized.get("stop_batch"), bool):
            raise ValueError(f"case result for {brief_id} has invalid stop_batch")
        if not isinstance(
            normalized.get("eligible_for_strict_one_shot_evaluation"), bool
        ):
            raise ValueError(f"case result for {brief_id} has invalid eligibility")
        if status == "complete" and normalized["stop_batch"]:
            raise ValueError(f"complete case result for {brief_id} cannot stop batch")
        if (
            normalized["eligible_for_strict_one_shot_evaluation"]
            and status != "complete"
        ):
            raise ValueError(
                f"incomplete case result for {brief_id} cannot be eligible"
            )
        return normalized

    def _build_summary(
        self,
        *,
        spec: GenerationRunSpec,
        model: Any,
        results: Sequence[Mapping[str, Any]],
        stopped: bool,
    ) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "schema_version": spec.summary_schema_version,
            "model_key": spec.model_key,
            "model_label": str(model.label),
            "requested_briefs": len(spec.ordered_brief_ids),
            "processed_briefs": len(results),
            "complete": sum(item["status"] == "complete" for item in results),
            "failed": sum(item["status"] != "complete" for item in results),
            "eligible": sum(
                bool(item["eligible_for_strict_one_shot_evaluation"])
                for item in results
            ),
            "stopped_early": stopped,
            "results": [dict(item) for item in results],
            "completed_at": self._core.utc_now(),
        }
        summary.update(thaw_json(spec.summary_extra))
        return summary

    @staticmethod
    def _emit(
        progress: SafeProgress | None,
        *,
        event: str,
        **fields: Any,
    ) -> None:
        if progress is None:
            return
        # The allowlisted calls above intentionally expose no prompts,
        # credentials, request IDs, response content, endpoints, or reasoning.
        progress({"event": event, **fields})
