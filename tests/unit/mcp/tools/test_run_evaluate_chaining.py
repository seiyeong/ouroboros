"""Run-to-evaluate chaining for PR-D.

Successful background ``ooo run`` results should enqueue the formal evaluator
as a separate bounded job.  Disabling the flag must leave the legacy run result
byte-for-byte at the metadata boundary.
"""

from __future__ import annotations

import asyncio
import faulthandler
import hashlib
import tempfile
import time
from typing import Any

import pytest

from ouroboros.core.types import Result
from ouroboros.events.base import BaseEvent
from ouroboros.mcp.errors import MCPToolError
from ouroboros.mcp.job_manager import JobManager, JobSnapshot, JobStatus
from ouroboros.mcp.tools import evaluation_handlers, execution_handlers
from ouroboros.mcp.tools.evaluation_handlers import StartEvaluateHandler
from ouroboros.mcp.tools.execution_handlers import (
    StartExecuteSeedHandler,
    _run_only_verification_meta,
    _run_only_verification_text,
)
from ouroboros.mcp.types import ContentType, MCPContentItem, MCPToolResult
from ouroboros.orchestrator.parallel_executor import render_parallel_verification_report
from ouroboros.orchestrator.parallel_executor_models import (
    ACExecutionResult,
    ParallelExecutionResult,
)
from ouroboros.persistence.event_store import EventStore


@pytest.fixture
async def event_store():
    store = EventStore("sqlite+aiosqlite:///:memory:")
    await store.initialize()
    yield store
    await store.close()


def _canonical_execution_summary(
    verification_report: str,
    task_results: list[dict[str, object]] | None = None,
    **overrides: object,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "acceptance_criteria_count": 1,
        "parallel_execution": True,
        "success_count": 1,
        "externally_satisfied_count": 0,
        "satisfied_count": 1,
        "failure_count": 0,
        "blocked_count": 0,
        "invalid_count": 0,
        "skipped_count": 0,
        "verification_report": verification_report,
        "verification_report_sha256": hashlib.sha256(
            verification_report.encode("utf-8")
        ).hexdigest(),
        "task_results": task_results
        if task_results is not None
        else [
            {
                "ac_index": 0,
                "outcome": "succeeded",
                "success": True,
                "evidence_present": True,
            }
        ],
    }
    summary.update(overrides)
    return summary


async def _wait_terminal(job_manager: JobManager, job_id: str) -> JobSnapshot:
    # 15s was too tight for loaded/shared CI runners (observed intermittent
    # timeouts on GitHub Actions despite the awaited work completing in <1s
    # locally); 60s gives ample headroom without weakening the assertion.
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        snapshot = await job_manager.get_snapshot(job_id)
        if snapshot.is_terminal:
            return snapshot
        task = job_manager._tasks.get(job_id)
        if task is not None and not task.done():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=min(0.05, remaining))
            except TimeoutError:
                pass
        else:
            await asyncio.sleep(0.01)
    diagnostics = _diagnose_stuck_job(job_manager, job_id)
    diagnostics += "\n" + await _dump_all_job_streams(job_manager)
    raise AssertionError(f"job {job_id} did not reach a terminal state\n{diagnostics}")


async def _dump_all_job_streams(job_manager: JobManager) -> str:
    """Dump every persisted job stream plus store/manager state (#1566 insurance).

    The append-void class of this flake (silently lost terminal events) is only
    diagnosable from the FULL persisted picture: the run job's stream, the
    chained evaluate job's stream, and the store connection/pool state.
    """
    lines: list[str] = ["--- all persisted job streams ---"]
    store = job_manager._event_store
    try:
        created_events = await store.query_events(event_type="mcp.job.created", limit=50)
        for job_id in [e.aggregate_id for e in created_events]:
            try:
                events, cursor = await store.get_events_after("job", job_id, 0)
            except Exception as exc:  # noqa: BLE001 - diagnostics must not mask failures
                lines.append(f"job {job_id}: <stream unavailable: {exc!r}>")
                continue
            lines.append(f"job {job_id} (cursor={cursor}):")
            for e in events:
                meta = e.data.get("result_meta")
                meta_keys = sorted(meta) if isinstance(meta, dict) else meta
                lines.append(
                    f"  {e.timestamp} {e.type} id={e.id} "
                    f"status={e.data.get('status')} meta_keys={meta_keys}"
                )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"<job discovery unavailable: {exc!r}>")
    try:
        engine = store._engine
        lines.append(f"engine pool: {engine.pool.status() if engine is not None else '<none>'}")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"<pool status unavailable: {exc!r}>")
    lines.append(f"manager tasks={sorted(job_manager._tasks)}")
    lines.append(f"manager runner_tasks={sorted(job_manager._runner_tasks)}")
    lines.append(f"manager backstops={sorted(getattr(job_manager, '_backstops', {}))}")
    lines.append(
        f"manager started_job_ids={sorted(getattr(job_manager, '_started_job_ids', set()))}"
    )
    return "\n".join(lines)


def _diagnose_stuck_job(job_manager: JobManager, job_id: str) -> str:
    """Capture where every task/thread is stuck when the job never terminalizes.

    This failure is CI-only (issue #1566) and has never reproduced locally, so
    the assertion message is the only diagnostic channel we get: dump the job
    task states plus every asyncio task stack and native thread stack.
    """
    lines: list[str] = ["--- stuck-job diagnostics (#1566) ---"]
    task = job_manager._tasks.get(job_id)
    runner = job_manager._runner_tasks.get(job_id)
    lines.append(f"job task: {task!r}")
    lines.append(f"runner task: {runner!r}")
    for t in asyncio.all_tasks():
        frames = t.get_stack(limit=6)
        where = " <- ".join(
            f"{f.f_code.co_name}:{f.f_code.co_filename.rsplit('/', 1)[-1]}:{f.f_lineno}"
            for f in reversed(frames)
        )
        lines.append(f"asyncio task {t.get_name()} done={t.done()}: {where or '<no stack>'}")
    # faulthandler needs a real file descriptor (StringIO raises
    # io.UnsupportedOperation: fileno), so dump native thread stacks to a
    # temp file and read them back into the assertion message.
    try:
        with tempfile.TemporaryFile(mode="w+") as buf:
            faulthandler.dump_traceback(file=buf)
            buf.seek(0)
            lines.append(buf.read())
    except Exception as exc:  # diagnostics must never mask the real failure
        lines.append(f"<faulthandler dump unavailable: {exc!r}>")
    return "\n".join(lines)


async def _wait_for_call(calls: list[Any]) -> None:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if calls:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("expected chained evaluate handler to be called")


class _SuccessfulExecuteHandler:
    agent_runtime_backend = None
    llm_backend = None

    def __init__(self, *, text: str = "run finished", worktree_path: str | None = None) -> None:
        self.text = text
        self.worktree_path = worktree_path
        self.returned_meta: dict[str, Any] | None = None

    async def handle(
        self,
        arguments: dict[str, Any],
        *,
        execution_id: str | None = None,
        session_id_override: str | None = None,
        synchronous: bool = False,
    ) -> Result[MCPToolResult, Any]:
        assert synchronous is True
        session_id = session_id_override or arguments.get("session_id") or "orch_fake"
        meta = {
            "seed_id": "seed-test",
            "session_id": session_id,
            "execution_id": execution_id,
            "launched": True,
            "status": "completed",
            "success": True,
            **_run_only_verification_meta(session_id),
        }
        if self.worktree_path is not None:
            meta["worktree_path"] = self.worktree_path
        self.returned_meta = dict(meta)
        return Result.ok(
            MCPToolResult(
                content=(
                    MCPContentItem(
                        type=ContentType.TEXT,
                        text=self.text + "\n" + _run_only_verification_text(session_id),
                    ),
                ),
                is_error=False,
                meta=meta,
            )
        )


class _ReceiptExecuteHandler(_SuccessfulExecuteHandler):
    def __init__(self, event_store: EventStore, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._event_store = event_store

    async def handle(
        self,
        arguments: dict[str, Any],
        *,
        execution_id: str | None = None,
        session_id_override: str | None = None,
        synchronous: bool = False,
    ) -> Result[MCPToolResult, Any]:
        result = await super().handle(
            arguments,
            execution_id=execution_id,
            session_id_override=session_id_override,
            synchronous=synchronous,
        )
        assert execution_id is not None
        assert session_id_override is not None
        await self._event_store.append(
            BaseEvent(
                type="execution.terminal",
                aggregate_type="execution",
                aggregate_id=execution_id,
                data={
                    "session_id": session_id_override,
                    "status": "completed",
                    "summary": _canonical_execution_summary(
                        "Parallel Execution Verification Report\n"
                        "Success: 1/1\n"
                        "\n## Task Results\n\n"
                        "### Task 1: [COMPLETED] canary\n"
                        "Result:\n"
                        "tests_passed: exit 0"
                    ),
                },
            )
        )
        return result


@pytest.mark.parametrize("receipt_query_raises", [False, True], ids=["missing", "query-failed"])
async def test_run_without_durable_receipt_fails_before_evaluation(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_query_raises: bool,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)
    evaluate_calls: list[dict[str, Any]] = []
    if receipt_query_raises:
        original_query_events = event_store.query_events

        async def _raise_for_execution_receipt(
            aggregate_id: str | None = None,
            event_type: str | None = None,
            limit: int = 50,
            offset: int = 0,
        ) -> list[BaseEvent]:
            if event_type == "execution.terminal":
                raise RuntimeError("receipt storage unavailable")
            return await original_query_events(aggregate_id, event_type, limit, offset)

        monkeypatch.setattr(event_store, "query_events", _raise_for_execution_receipt)

    class FakeEvaluateHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            evaluate_calls.append({"arguments": arguments, "kwargs": self.kwargs})
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="approved"),),
                    is_error=False,
                    meta={
                        "final_approved": True,
                        "session_id": arguments["session_id"],
                    },
                )
            )

    monkeypatch.setattr(evaluation_handlers, "EvaluateHandler", FakeEvaluateHandler)

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=_SuccessfulExecuteHandler(text="execution artifact"),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    seed_content = (
        "goal: Build a CLI task manager\n"
        "acceptance_criteria:\n"
        "  - Tasks can be created\n"
        "  - Tasks can be listed\n"
        "ontology_schema:\n"
        "  name: TaskManager\n"
        "  description: Task management domain\n"
        "metadata:\n"
        "  ambiguity_score: 0.15\n"
    )
    started = await handler.handle({"seed_content": seed_content, "cwd": str(tmp_path)})
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status is JobStatus.FAILED
    assert snapshot.result_meta["success"] is False
    assert snapshot.result_meta["verification_status"] == "evaluation_unavailable"
    assert snapshot.result_meta["evaluation_status"] == "receipt_unavailable"
    assert snapshot.result_meta["evaluated"] is False
    assert snapshot.result_meta["final_approved"] is False
    assert "chained_evaluate_job_id" not in snapshot.result_meta
    assert "Formal Evaluation: receipt_unavailable; run is not complete." in (
        snapshot.result_text or ""
    )
    assert not evaluate_calls


async def test_chained_evaluation_artifact_returns_none_without_execution_id(event_store) -> None:
    run_result = MCPToolResult(
        content=(MCPContentItem(type=ContentType.TEXT, text="ordinary run output"),),
        is_error=False,
        meta={},
    )

    artifact = await execution_handlers._chained_evaluation_artifact(
        event_store,
        run_result,
        "orch_receipt",
    )

    assert artifact is None


async def test_chained_evaluation_artifact_accepts_renderer_output(event_store) -> None:
    report = render_parallel_verification_report(
        ParallelExecutionResult(
            results=(
                ACExecutionResult(
                    ac_index=0,
                    ac_content="receipt",
                    success=True,
                    final_message="### Task 1: [FAILED] incidental markdown",
                ),
            ),
            success_count=1,
            failure_count=0,
        ),
        1,
    )
    await event_store.append(
        BaseEvent(
            type="execution.terminal",
            aggregate_type="execution",
            aggregate_id="exec_renderer_receipt",
            data={
                "session_id": "orch_receipt",
                "status": "completed",
                "summary": _canonical_execution_summary(report),
            },
        )
    )
    run_result = MCPToolResult(
        is_error=False,
        meta={"execution_id": "exec_renderer_receipt"},
    )

    artifact = await execution_handlers._chained_evaluation_artifact(
        event_store,
        run_result,
        "orch_receipt",
    )

    assert artifact is not None
    assert artifact.startswith("Run acceptance receipt:\n\n" + report)
    assert "## Durable Task Receipt" in artifact


async def test_chained_evaluation_artifact_requires_hash_bound_typed_task_receipt(
    event_store,
) -> None:
    report = (
        "Parallel Execution Verification Report\n"
        "Success: 1/1\n"
        "\n## Task Results\n\n"
        "### Task 1: [COMPLETED] receipt\n"
        "Result:\n"
        "tests_passed: exit 0"
    )
    summary = _canonical_execution_summary(report)
    del summary["verification_report_sha256"]
    del summary["task_results"]
    await event_store.append(
        BaseEvent(
            type="execution.terminal",
            aggregate_type="execution",
            aggregate_id="exec_missing_typed_receipt",
            data={
                "session_id": "orch_receipt",
                "status": "completed",
                "summary": summary,
            },
        )
    )

    artifact = await execution_handlers._chained_evaluation_artifact(
        event_store,
        MCPToolResult(is_error=False, meta={"execution_id": "exec_missing_typed_receipt"}),
        "orch_receipt",
    )

    assert artifact is None


@pytest.mark.parametrize(
    ("terminal_session_id", "terminal_status", "summary"),
    [
        (
            "orch_other",
            "completed",
            {
                "verification_report": (
                    "Parallel Execution Verification Report\n"
                    "Success: 1/1\n"
                    "\n## Task Results\n\n"
                    "### Task 1"
                )
            },
        ),
        ("orch_receipt", "completed", {"verification_report": "  \n"}),
        ("orch_receipt", "completed", "not a receipt mapping"),
        ("orch_receipt", "completed", {"verification_report": "garbage"}),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                (
                    "Parallel Execution Verification Report\n"
                    "Success: 0/1\n"
                    "\n## Task Results\n\n"
                    "### Task 1: [FAILED]"
                ),
                success_count=0,
                satisfied_count=0,
                failure_count=1,
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "### Task 1: [FAILED] injected contradiction",
                task_results=[
                    {
                        "ac_index": 0,
                        "outcome": "failed",
                        "success": False,
                        "evidence_present": True,
                    }
                ],
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "Result:",
                task_results=[
                    {
                        "ac_index": 0,
                        "outcome": "succeeded",
                        "success": True,
                        "evidence_present": False,
                    }
                ],
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "### Task 1: [COMPLETED] canary",
                failure_count=1,
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "### Task 1: [COMPLETED] canary"
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "### Task 1: [COMPLETED] canary",
                verification_report_sha256="0" * 64,
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "### Task 2: [COMPLETED] wrong index",
                task_results=[
                    {
                        "ac_index": 1,
                        "outcome": "succeeded",
                        "success": True,
                        "evidence_present": True,
                    }
                ],
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "### Task 1: [COMPLETED] canary",
                task_results=[
                    {
                        "ac_index": 0,
                        "outcome": "satisfied_externally",
                        "success": True,
                        "evidence_present": True,
                    }
                ],
            ),
        ),
        (
            "orch_receipt",
            "completed",
            _canonical_execution_summary(
                "Parallel Execution Verification Report\n"
                "Success: 1/1\n"
                "\n## Task Results\n\n"
                "### Task 1: [COMPLETED] parent\n"
                "Decomposed into 1 Subtasks\n\n"
                "#### Subtask 1.1: [FAILED] hidden failure",
                task_results=[
                    {
                        "ac_index": 0,
                        "outcome": "failed",
                        "success": False,
                        "evidence_present": True,
                    }
                ],
            ),
        ),
        (
            "orch_receipt",
            "completed",
            {"verification_report": "Parallel Execution Verification Report\nSuccess: 1/1"},
        ),
    ],
    ids=[
        "session-mismatch",
        "empty-report",
        "malformed-summary",
        "unstructured-report",
        "failed-report",
        "contradictory-task-result",
        "missing-typed-evidence",
        "typed-failure-count",
        "duplicate-success-count",
        "report-hash-mismatch",
        "wrong-task-index",
        "outcome-count-mismatch",
        "failed-subtask",
        "missing-task-results",
    ],
)
async def test_chained_evaluation_artifact_requires_matching_nonempty_receipt(
    event_store,
    terminal_session_id: str,
    terminal_status: str,
    summary,
) -> None:
    await event_store.append(
        BaseEvent(
            type="execution.terminal",
            aggregate_type="execution",
            aggregate_id="exec_receipt",
            data={
                "session_id": terminal_session_id,
                "status": terminal_status,
                "summary": summary,
            },
        )
    )
    run_result = MCPToolResult(
        content=(MCPContentItem(type=ContentType.TEXT, text="ordinary run output"),),
        is_error=False,
        meta={"execution_id": "exec_receipt"},
    )

    artifact = await execution_handlers._chained_evaluation_artifact(
        event_store,
        run_result,
        "orch_receipt",
    )

    assert artifact is None


async def test_chained_evaluate_uses_durable_execution_receipt(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)
    evaluate_calls: list[dict[str, Any]] = []

    class ReceiptExecuteHandler:
        agent_runtime_backend = None
        llm_backend = None

        async def handle(
            self,
            arguments: dict[str, Any],
            *,
            execution_id: str | None = None,
            session_id_override: str | None = None,
            synchronous: bool = False,
        ) -> Result[MCPToolResult, Any]:
            assert execution_id is not None
            assert session_id_override is not None
            assert synchronous is True
            await event_store.append(
                BaseEvent(
                    type="execution.terminal",
                    aggregate_type="execution",
                    aggregate_id=execution_id,
                    data={
                        "session_id": session_id_override,
                        "status": "completed",
                        "summary": _canonical_execution_summary(
                            "Parallel Execution Verification Report\n"
                            "Success: 1/1\n"
                            "\n## Task Results\n\n"
                            "### Task 1: [COMPLETED] receipt\n"
                            "File Changes:\n- lazycodex_canary.txt\n"
                            "tests_passed: verify_command exit 0"
                        ),
                    },
                )
            )
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="run-only warning"),),
                    is_error=False,
                    meta={
                        "session_id": session_id_override,
                        "execution_id": execution_id,
                        "success": True,
                    },
                )
            )

    class FakeEvaluateHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            evaluate_calls.append(arguments)
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="approved"),),
                    is_error=False,
                    meta={"final_approved": True, "session_id": arguments["session_id"]},
                )
            )

    monkeypatch.setattr(evaluation_handlers, "EvaluateHandler", FakeEvaluateHandler)

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=ReceiptExecuteHandler(),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle({"seed_content": "goal: receipt\n", "cwd": str(tmp_path)})
    assert started.is_ok
    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status is JobStatus.COMPLETED
    await _wait_for_call(evaluate_calls)
    artifact = evaluate_calls[0]["artifact"]
    assert artifact.startswith("Run acceptance receipt:")
    assert "verify_command exit 0" in artifact
    assert "run-only warning" not in artifact


async def test_chained_evaluate_rejection_fails_run(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)

    class RejectingEvaluateHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="rejected AC"),),
                    is_error=False,
                    meta={"final_approved": False, "session_id": arguments["session_id"]},
                )
            )

    monkeypatch.setattr(evaluation_handlers, "EvaluateHandler", RejectingEvaluateHandler)

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=_ReceiptExecuteHandler(event_store),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle({"seed_content": "goal: rejection\n", "cwd": str(tmp_path)})
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status is JobStatus.FAILED
    assert snapshot.result_meta["success"] is False
    assert snapshot.result_meta["evaluated"] is True
    assert snapshot.result_meta["verification_status"] == "evaluation_rejected"
    assert snapshot.result_meta["evaluation_status"] == "rejected"
    assert snapshot.result_meta["final_approved"] is False
    assert "Formal Evaluation: rejected; run is not complete." in (snapshot.result_text or "")


async def test_chained_evaluate_timeout_fails_run_with_specific_meta(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)

    class TimedOutEvaluateHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="evaluation timed out"),),
                    is_error=True,
                    meta={
                        "session_id": arguments["session_id"],
                        "evaluation_status": "timed_out",
                    },
                )
            )

    monkeypatch.setattr(evaluation_handlers, "EvaluateHandler", TimedOutEvaluateHandler)

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=_ReceiptExecuteHandler(event_store),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle({"seed_content": "goal: timeout\n", "cwd": str(tmp_path)})
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status is JobStatus.FAILED
    assert snapshot.result_meta["evaluated"] is False
    assert snapshot.result_meta["verification_status"] == "evaluation_unavailable"
    assert snapshot.result_meta["evaluation_status"] == "timed_out"
    assert snapshot.result_meta["final_approved"] is False
    assert snapshot.result_meta["chained_evaluate_job_id"].startswith("job_")


async def test_chained_evaluate_completed_without_verdict_fails_run(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)

    class InvalidEvaluateHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="missing verdict"),),
                    is_error=False,
                    meta={"session_id": arguments["session_id"]},
                )
            )

    monkeypatch.setattr(evaluation_handlers, "EvaluateHandler", InvalidEvaluateHandler)

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=_ReceiptExecuteHandler(event_store),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle({"seed_content": "goal: invalid\n", "cwd": str(tmp_path)})
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status is JobStatus.FAILED
    assert snapshot.result_meta["evaluated"] is False
    assert snapshot.result_meta["evaluation_status"] == "invalid_result"
    assert snapshot.result_meta["final_approved"] is False
    assert snapshot.result_meta["chained_evaluate_job_id"].startswith("job_")


async def test_run_job_stranded_without_terminal_event_still_terminalizes(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for this file's own CI flake signature (#1566 / PR #1576):
    the run job's task is fully released — job task AND runner task popped,
    persisted stream = created + running only — with no terminal event and no
    log. Whatever silently defeats the inline guards (modeled here by dropping
    every terminal append for the run job while its task lives), the run job
    must still reach a terminal state instead of zombying for the full wait
    deadline: first via the detached release-point backstop, and failing that
    via the get_snapshot in-process stranded-job net.
    """
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)

    class FakeEvaluateHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="approved"),),
                    is_error=False,
                    meta={"final_approved": True, "session_id": arguments["session_id"]},
                )
            )

    monkeypatch.setattr(evaluation_handlers, "EvaluateHandler", FakeEvaluateHandler)

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=_ReceiptExecuteHandler(event_store, text="execution artifact"),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    _terminal_types = {
        "mcp.job.completed",
        "mcp.job.failed",
        "mcp.job.cancelled",
        "mcp.job.interrupted",
    }
    # Identify the run job from its created event (allocated before handle()
    # returns) and drop its terminal appends while the drop flag is on. The
    # chained evaluate job is untouched.
    target: dict[str, str | None] = {"job_id": None}
    dropping = {"on": True}
    original_append_event = job_manager._append_event

    async def _drop_run_job_terminal_appends(
        event_type: str, job_id: str, data: dict, **kwargs: Any
    ) -> None:
        if event_type == "mcp.job.created" and data.get("job_type") == "execute_seed":
            target["job_id"] = job_id
        if dropping["on"] and job_id == target["job_id"] and event_type in _terminal_types:
            return  # silently lost, like the CI capture: no event, no exception
        await original_append_event(event_type, job_id, data, **kwargs)

    job_manager._append_event = _drop_run_job_terminal_appends

    started = await handler.handle({"seed_content": "goal: stranded\n", "cwd": str(tmp_path)})
    assert started.is_ok
    run_job_id = started.value.meta["job_id"]
    assert target["job_id"] == run_job_id

    # Wait for the run job's tasks to be fully released and its detached
    # backstop (whose appends are also dropped) to finish: the CI signature.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and (
        run_job_id in job_manager._tasks or getattr(job_manager, "_backstops", {})
    ):
        await asyncio.sleep(0.01)
    assert run_job_id not in job_manager._tasks
    assert run_job_id not in job_manager._runner_tasks
    events, _ = await event_store.get_events_after("job", run_job_id, 0)
    assert all(e.type in {"mcp.job.created", "mcp.job.updated"} for e in events)

    # From here the store is healthy again; the job must terminalize promptly
    # instead of zombying for the full 60s wait deadline.
    dropping["on"] = False
    deadline = time.monotonic() + 5.0
    snapshot = await job_manager.get_snapshot(run_job_id)
    while not snapshot.is_terminal and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        snapshot = await job_manager.get_snapshot(run_job_id)
    assert snapshot.is_terminal, f"run job stranded in {snapshot.status}"
    assert snapshot.result_meta.get("interrupted_from_stranded_job_task") is True


async def test_chained_evaluate_uses_execution_worktree_when_present(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)
    evaluate_calls: list[dict[str, Any]] = []

    class FakeEvaluateHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            evaluate_calls.append(arguments)
            return Result.ok(
                MCPToolResult(
                    content=(MCPContentItem(type=ContentType.TEXT, text="approved"),),
                    is_error=False,
                    meta={"final_approved": True, "session_id": arguments["session_id"]},
                )
            )

    monkeypatch.setattr(evaluation_handlers, "EvaluateHandler", FakeEvaluateHandler)
    execution_worktree = tmp_path / "task-worktree"
    execution_worktree.mkdir()
    original_cwd = tmp_path / "original"
    original_cwd.mkdir()

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=_ReceiptExecuteHandler(
            event_store,
            text="execution artifact",
            worktree_path=str(execution_worktree),
        ),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle({"seed_content": "goal: chain\n", "cwd": str(original_cwd)})
    assert started.is_ok

    await _wait_for_call(evaluate_calls)
    snapshot = await job_manager.get_snapshot(started.value.meta["job_id"])
    assert snapshot.job_type == "execute_seed"

    assert evaluate_calls
    assert evaluate_calls[0]["working_dir"] == str(execution_worktree)


async def test_auto_evaluate_override_false_preserves_legacy_run_meta_exactly(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)

    class UnexpectedStartEvaluateHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, _: dict[str, Any]) -> Result[MCPToolResult, Any]:
            raise AssertionError("auto_evaluate=false must not enqueue evaluation")

    monkeypatch.setattr(
        evaluation_handlers,
        "StartEvaluateHandler",
        UnexpectedStartEvaluateHandler,
    )

    execute_handler = _SuccessfulExecuteHandler(text="legacy")
    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=execute_handler,  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle(
        {
            "seed_content": "goal: legacy\n",
            "cwd": str(tmp_path),
            "auto_evaluate": False,
        }
    )
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status == JobStatus.COMPLETED
    assert execute_handler.returned_meta is not None
    assert snapshot.result_meta == execute_handler.returned_meta
    assert "chained_evaluate_job_id" not in snapshot.result_meta
    assert snapshot.result_meta["verification_status"] == "executed_unverified"


async def test_auto_evaluate_config_false_preserves_legacy_run_meta_exactly(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: False)

    class UnexpectedStartEvaluateHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, _: dict[str, Any]) -> Result[MCPToolResult, Any]:
            raise AssertionError("execution.auto_evaluate=false must not enqueue evaluation")

    monkeypatch.setattr(
        evaluation_handlers,
        "StartEvaluateHandler",
        UnexpectedStartEvaluateHandler,
    )

    execute_handler = _SuccessfulExecuteHandler(text="legacy config")
    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=execute_handler,  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle({"seed_content": "goal: config-off\n", "cwd": str(tmp_path)})
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status == JobStatus.COMPLETED
    assert execute_handler.returned_meta is not None
    assert snapshot.result_meta == execute_handler.returned_meta
    assert "chained_evaluate_job_id" not in snapshot.result_meta
    assert snapshot.result_meta["verification_status"] == "executed_unverified"


async def test_evaluate_enqueue_failure_fails_run(
    event_store,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(execution_handlers, "get_auto_evaluate_enabled", lambda: True)

    class FailingStartEvaluateHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, _: dict[str, Any]) -> Result[MCPToolResult, Any]:
            return Result.err(MCPToolError("enqueue boom", tool_name="ouroboros_start_evaluate"))

    monkeypatch.setattr(evaluation_handlers, "StartEvaluateHandler", FailingStartEvaluateHandler)

    job_manager = JobManager(event_store)
    handler = StartExecuteSeedHandler(
        execute_handler=_ReceiptExecuteHandler(event_store),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
    )

    started = await handler.handle({"seed_content": "goal: failure\n", "cwd": str(tmp_path)})
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status == JobStatus.FAILED
    assert snapshot.result_meta["success"] is False
    assert snapshot.result_meta["evaluation_status"] == "enqueue_failed"
    assert snapshot.result_meta["evaluation_error"] == "enqueue boom"
    assert snapshot.result_meta["next_step"].startswith("ooo evaluate orch_")
    assert "chained_evaluate_job_id" not in snapshot.result_meta
    assert "Formal Evaluation: enqueue_failed; run is not complete." in (snapshot.result_text or "")


async def test_start_evaluate_timeout_writes_terminal_event(
    event_store,
) -> None:
    class SlowEvaluateHandler:
        async def handle(self, _: dict[str, Any]) -> Result[MCPToolResult, Any]:
            await asyncio.sleep(1)
            return Result.ok(MCPToolResult())

    job_manager = JobManager(event_store)
    handler = StartEvaluateHandler(
        evaluate_handler=SlowEvaluateHandler(),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=job_manager,
        deadline_seconds=0.01,
    )

    started = await handler.handle({"session_id": "orch_timeout", "artifact": "code"})
    assert started.is_ok

    snapshot = await _wait_terminal(job_manager, started.value.meta["job_id"])

    assert snapshot.status == JobStatus.FAILED
    assert snapshot.result_meta["session_id"] == "orch_timeout"
    assert snapshot.result_meta["evaluation_status"] == "timed_out"
    assert snapshot.result_meta["status"] == "timed_out"
    assert "Evaluation timed out" in (snapshot.result_text or "")

    events, _ = await event_store.get_events_after("job", started.value.meta["job_id"])
    terminal = [event for event in events if event.type == "mcp.job.failed"]
    assert terminal
    assert terminal[-1].data["result_meta"]["evaluation_status"] == "timed_out"
