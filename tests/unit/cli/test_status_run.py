"""Tests for ``ouroboros status run`` — Wave-1 #946 S2 thin CLI surface.

These tests pin the contract that the CLI is a thin wrapper over
``ouroboros_query_projection``:

* The same arguments produce byte-identical JSON between the MCP handler
  and the CLI ``--json`` output (golden test).
* Exit codes follow the documented convention: ``0`` for success, ``2``
  for an unknown run anchor, ``64`` for malformed input.
* The positional ``RUN_ID`` argument is shorthand for ``--execution-id``.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from ouroboros.cli.main import app
from ouroboros.core.types import Result
from ouroboros.events.base import BaseEvent
from ouroboros.mcp.errors import MCPToolError
from ouroboros.mcp.tools.projection_handlers import ProjectionQueryHandler
from ouroboros.mcp.types import ContentType, MCPContentItem, MCPToolResult
from ouroboros.persistence.event_store import EventStore

runner = CliRunner(env={"COLUMNS": "240"})


def _golden_meta(run_id: str = "run_golden", execution_id: str = "exec_golden") -> dict:
    """Build a representative projection meta payload for golden comparisons."""

    return {
        "session_id": None,
        "execution_id": execution_id,
        "seed_id": "seed_golden",
        "seed_id_source": "event",
        "event_count": 3,
        "limit": None,
        "run": {
            "run_id": run_id,
            "seed_id": "seed_golden",
            "goal": "Golden goal",
            "schema_version": 1,
        },
        "stages": [
            {
                "stage_id": "stage_golden",
                "run_id": run_id,
                "kind": "execute",
            }
        ],
        "steps": [
            {
                "step_id": "step_golden",
                "run_id": run_id,
                "stage_id": "stage_golden",
                "kind": "tool_call",
                "name": "shell",
                "ok": True,
            }
        ],
        "artifacts": [],
        "verdicts": [],
    }


class _RecordingHandler:
    """Stand-in for ``ProjectionQueryHandler`` that records call args."""

    def __init__(self, result: Result):
        self.result = result
        self.last_arguments: dict | None = None


def _patched_runner(handler: _RecordingHandler):
    async def fake_handle(_self, arguments):  # noqa: D401 - test double
        handler.last_arguments = dict(arguments)
        return handler.result

    return patch(
        "ouroboros.cli.commands.status.ProjectionQueryHandler.handle",
        fake_handle,
    )


def test_status_run_json_matches_mcp_handler_output() -> None:
    """Golden test: CLI ``--json`` must reproduce the MCP meta payload exactly."""

    meta = _golden_meta()
    mcp_payload = MCPToolResult(
        content=(MCPContentItem(type=ContentType.TEXT, text="Run Projection\nRun: run_golden"),),
        meta=meta,
    )
    handler = _RecordingHandler(Result.ok(mcp_payload))

    with _patched_runner(handler):
        result = runner.invoke(
            app,
            ["status", "run", "exec_golden", "--json"],
        )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed == meta
    # The CLI must serialize via the same key ordering it documents.
    assert result.output.rstrip("\n") == json.dumps(meta, indent=2, sort_keys=True)


def test_status_run_positional_maps_to_execution_id() -> None:
    """The positional ``RUN_ID`` is shorthand for ``--execution-id``."""

    mcp_payload = MCPToolResult(content=(), meta=_golden_meta())
    handler = _RecordingHandler(Result.ok(mcp_payload))

    with _patched_runner(handler):
        result = runner.invoke(app, ["status", "run", "exec_golden", "--json"])

    assert result.exit_code == 0
    assert handler.last_arguments == {"execution_id": "exec_golden"}


def test_status_run_unknown_run_id_exits_with_code_2() -> None:
    """Unknown run anchors map to exit code ``2``."""

    handler = _RecordingHandler(Result.err(MCPToolError("No events found for projection query")))

    with _patched_runner(handler):
        result = runner.invoke(app, ["status", "run", "exec_missing", "--json"])

    assert result.exit_code == 2
    assert "No events found" in result.output


def test_status_run_missing_selector_exits_with_code_64() -> None:
    """Malformed input (no selector at all) maps to exit code ``64``."""

    result = runner.invoke(app, ["status", "run", "--json"])

    assert result.exit_code == 64
    assert "required" in result.output.lower()


def test_status_run_invalid_limit_exits_with_code_64() -> None:
    """Malformed numeric CLI input maps to exit code ``64`` before MCP dispatch."""

    mcp_payload = MCPToolResult(content=(), meta=_golden_meta())
    handler = _RecordingHandler(Result.ok(mcp_payload))

    with _patched_runner(handler):
        result = runner.invoke(app, ["status", "run", "exec_golden", "--limit", "0", "--json"])

    assert result.exit_code == 64
    assert "limit must be a positive integer" in result.output
    assert handler.last_arguments is None


def test_status_run_conflicting_selectors_exit_with_code_64() -> None:
    """RUN_ID combined with --session-id is malformed input."""

    result = runner.invoke(
        app,
        ["status", "run", "exec_golden", "--session-id", "session_xyz", "--json"],
    )

    assert result.exit_code == 64
    assert "session" in result.output.lower()


def test_status_run_conflicting_execution_id_exits_with_code_64() -> None:
    """RUN_ID and a different --execution-id must be flagged as malformed."""

    result = runner.invoke(
        app,
        [
            "status",
            "run",
            "exec_golden",
            "--execution-id",
            "exec_other",
            "--json",
        ],
    )

    assert result.exit_code == 64
    assert "different" in result.output.lower()


def test_status_run_session_id_only_still_supported() -> None:
    """Legacy ``--session-id`` invocations remain valid (no positional)."""

    mcp_payload = MCPToolResult(content=(), meta=_golden_meta())
    handler = _RecordingHandler(Result.ok(mcp_payload))

    with _patched_runner(handler):
        result = runner.invoke(
            app,
            ["status", "run", "--session-id", "session_legacy", "--json"],
        )

    assert result.exit_code == 0
    assert handler.last_arguments == {"session_id": "session_legacy"}


def _runtime_tool_events(session_id: str) -> tuple[BaseEvent, ...]:
    started = datetime.now(UTC)
    return (
        BaseEvent(
            id="evt_exec_start",
            type="execution.tool.started",
            timestamp=started,
            aggregate_type="execution",
            aggregate_id="exec_projected",
            data={
                "execution_id": "exec_projected",
                "session_id": session_id,
                "tool_call_id": "call_1",
                "tool_name": "Bash",
            },
        ),
        BaseEvent(
            id="evt_exec_done",
            type="execution.tool.completed",
            timestamp=started,
            aggregate_type="execution",
            aggregate_id="exec_projected",
            data={
                "execution_id": "exec_projected",
                "session_id": session_id,
                "tool_call_id": "call_1",
                "tool_name": "Bash",
                "is_error": False,
            },
        ),
    )


def _session_started_event(session_id: str) -> BaseEvent:
    return BaseEvent(
        id=f"evt_session_{session_id}",
        type="orchestrator.session.started",
        timestamp=datetime.now(UTC),
        aggregate_type="session",
        aggregate_id=session_id,
        data={
            "execution_id": "exec_projected",
            "seed_id": "seed_from_session",
            "seed_goal": "Ship the hello world script",
        },
    )


async def _projection_meta_for(tmp_path: Path, events: tuple[BaseEvent, ...]) -> dict:
    store = EventStore(f"sqlite+aiosqlite:///{tmp_path / 'events.db'}")
    await store.initialize()
    try:
        for event in events:
            await store.append(event)
        result = await ProjectionQueryHandler(event_store=store).handle(
            {"execution_id": "exec_projected"}
        )
    finally:
        await store.close()
    return result.unwrap().meta


@pytest.mark.asyncio
async def test_execution_projection_uses_session_goal_and_seed_id(tmp_path: Path) -> None:
    events = (*_runtime_tool_events("orch_single"), _session_started_event("orch_single"))

    meta = await _projection_meta_for(tmp_path, events)

    assert meta["seed_id"] == "seed_from_session"
    assert meta["seed_id_source"] == "session"
    assert meta["run"]["goal"] == "Ship the hello world script"
    assert len(meta["steps"]) == 1


@pytest.mark.asyncio
async def test_execution_projection_keeps_fallback_for_multiple_sessions(tmp_path: Path) -> None:
    events = (
        *_runtime_tool_events("orch_one"),
        *(
            event.model_copy(update={"id": f"{event.id}_b"})
            for event in _runtime_tool_events("orch_two")
        ),
        _session_started_event("orch_one"),
        _session_started_event("orch_two"),
    )

    meta = await _projection_meta_for(tmp_path, events)

    assert meta["seed_id"] == "exec_projected"
    assert meta["seed_id_source"] == "fallback"
    assert meta["run"]["goal"] == ""
