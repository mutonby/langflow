"""Tests for relocating knowledge base vectors between backends.

Runs against the real test database and real local Chroma directories. The
Chroma-to-Postgres move needs a pgvector database and is opt-in via
``LANGFLOW_RUN_PGVECTOR_INTEGRATION_TESTS=1`` and ``PGVECTOR_CONNECTION_STRING``.
"""

from __future__ import annotations

import contextlib
import gc
import math
import os
import uuid
from typing import TYPE_CHECKING

import pytest
from langflow.api.utils import knowledge_base_service
from langflow.api.utils.knowledge_base_relocation import (
    KBRelocationResult,
    relocate_knowledge_bases,
    validate_relocation_target_config,
)
from langflow.services.database.models.knowledge_base import KnowledgeBaseStatus
from langflow.services.deps import get_settings_service
from lfx.base.knowledge_bases.backends import ChromaLocalBackend, IngestedDocument, create_backend

if TYPE_CHECKING:
    from pathlib import Path

DIM = 8


@pytest.fixture
def kb_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "knowledge_bases"
    root.mkdir()
    monkeypatch.setattr(get_settings_service().settings, "knowledge_bases_dir", str(root))
    return root


@pytest.fixture
def during_copy(monkeypatch: pytest.MonkeyPatch):
    """Run ``action`` once, right after the relocation reads its first batch from Chroma.

    Wraps the real reader, so the copy itself is untouched; this only stands in
    for something else touching the knowledge base while it moves.
    """
    original = ChromaLocalBackend.iter_documents

    def install(action):
        async def iter_documents(self, **kwargs):
            first = True
            async for batch in original(self, **kwargs):
                yield batch
                if first:
                    first = False
                    await action()

        monkeypatch.setattr(ChromaLocalBackend, "iter_documents", iter_documents)

    return install


def _vector(i: int, *, unit: bool) -> list[float]:
    raw = [1.0] + [i / 100] * (DIM - 1)
    if not unit:
        return raw
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


async def _seed_chroma_kb(
    kb_root: Path, username: str, kb_name: str, n: int, *, unit: bool = True
) -> list[IngestedDocument]:
    """Chunks in a local Chroma KB. Unit-length vectors by default, like most hosted embedding models."""
    path = kb_root / username / kb_name
    path.mkdir(parents=True)
    docs = [
        IngestedDocument(id=f"chunk-{i}", content=f"doc {i}", metadata={"i": i}, embedding=_vector(i, unit=unit))
        for i in range(n)
    ]
    backend = ChromaLocalBackend(kb_name=kb_name, kb_path=path)
    try:
        await backend.add_embedded_documents(docs)
    finally:
        await backend.teardown()
        gc.collect()
    return docs


class TestRelocationWithoutATarget:
    async def test_stubbed_source_backend_is_reported_not_raised(self, active_user, kb_root):  # noqa: ARG002
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name="kb_on_astra", backend_type="astra", chunks=3
        )

        results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={})

        result = next(r for r in results if r.kb_id == record.id)
        assert result.status == "failed"
        assert "astra" in result.reason
        row = await knowledge_base_service.get_by_id(record.id)
        assert row.backend_type == "astra"

    async def test_kb_already_on_the_target_is_skipped(self, active_user, kb_root):  # noqa: ARG002
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name="kb_on_pg", backend_type="postgres"
        )

        results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={})

        assert next(r for r in results if r.kb_id == record.id).status == "skipped"

    async def test_dry_run_checks_the_target_is_reachable(self, active_user, kb_root, monkeypatch):
        monkeypatch.delenv("PGVECTOR_CONNECTION_STRING", raising=False)
        await _seed_chroma_kb(kb_root, active_user.username, "kb_dry_unreachable", 2)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name="kb_dry_unreachable", model_selection={"name": "m", "provider": "p"}, chunks=2
        )

        results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={}, dry_run=True)

        result = next(r for r in results if r.kb_id == record.id)
        assert result.status == "failed"
        assert "target" in result.reason

    async def test_kb_being_ingested_is_not_moved(self, active_user, kb_root):
        # An ingestion still running would keep writing to the source after the copy.
        await _seed_chroma_kb(kb_root, active_user.username, "kb_busy", 2)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name="kb_busy", model_selection={"name": "m", "provider": "p"}, chunks=2
        )
        await knowledge_base_service.update_status(record.id, status=KnowledgeBaseStatus.INGESTING)

        results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={})

        result = next(r for r in results if r.kb_id == record.id)
        assert result.status == "failed"
        assert "ingesting" in result.reason
        row = await knowledge_base_service.get_by_id(record.id)
        assert row.backend_type == "chroma"

    async def test_empty_model_selection_is_warned_about(self, active_user, kb_root):
        await _seed_chroma_kb(kb_root, active_user.username, "kb_no_model", 2)
        record = await knowledge_base_service.create_record(user_id=active_user.id, name="kb_no_model", chunks=2)

        results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={}, dry_run=True)

        warnings = next(r for r in results if r.kb_id == record.id).warnings
        assert any("model_selection" in w for w in warnings)

    async def test_source_holding_fewer_chunks_than_recorded_is_not_repointed(self, active_user, kb_root):
        # The row says 10, the store holds 4: that is what a truncated or
        # half-lost store looks like, so the relocation must refuse, not copy 4.
        await _seed_chroma_kb(kb_root, active_user.username, "kb_short", 4)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name="kb_short", model_selection={"name": "m", "provider": "p"}, chunks=10
        )

        results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={})

        result = next(r for r in results if r.kb_id == record.id)
        assert result.status == "failed"
        assert "4 of 10" in result.reason
        row = await knowledge_base_service.get_by_id(record.id)
        assert row.backend_type == "chroma"


@pytest.mark.parametrize(
    ("status", "copied", "expected"),
    [("relocated", 3, "chunks 3/3"), ("failed", 0, "chunks 0/3"), ("would_relocate", 0, "chunks 3")],
)
def test_relocation_line_shows_what_was_copied(status, copied, expected):
    from langflow.__main__ import relocation_line

    result = KBRelocationResult(
        kb_id=uuid.uuid4(),
        kb_name="kb",
        owner="alice",
        source_backend="chroma",
        target_backend="postgres",
        status=status,
        source_count=3,
        copied=copied,
    )

    assert relocation_line(result).endswith(expected)


@pytest.mark.parametrize(
    ("backend_type", "config", "override"),
    [
        ("opensearch", {"index_name": "shared"}, "index_name"),
        ("chroma", {"mode": "cloud", "collection_name": "shared"}, "collection_name"),
    ],
)
async def test_relocation_rejects_shared_target_collection(backend_type, config, override):
    with pytest.raises(ValueError, match=override):
        await relocate_knowledge_bases(target_backend_type=backend_type, target_backend_config=config)


@pytest.mark.parametrize(
    ("backend_type", "config"),
    [
        ("opensearch", {"index_name": ""}),
        ("chroma", {"mode": "cloud", "collection_name": ""}),
    ],
)
def test_relocation_allows_empty_collection_override(backend_type, config):
    validate_relocation_target_config(backend_type, config)


@pytest.mark.api_key_required
@pytest.mark.parametrize(
    ("backend_type", "config", "metric"),
    [
        ("chroma", {}, "l2"),
        ("postgres", {}, "cosine"),
        ("opensearch", {"url_variable": "OPENSEARCH_URL"}, "l2"),
        ("opensearch", {"url_variable": "OPENSEARCH_URL", "space_type": "cosinesimil"}, "cosine"),
        ("opensearch", {"url_variable": "OPENSEARCH_URL", "space_type": "innerproduct"}, "inner_product"),
    ],
)
def test_backends_report_the_metric_they_rank_by(tmp_path: Path, backend_type, config, metric):
    backend = create_backend(backend_type, kb_name="kb", kb_path=tmp_path, backend_config=config, user_id=uuid.uuid4())
    assert backend.distance_metric == metric


class TestRelocationToPostgresLive:
    @pytest.fixture(autouse=True)
    def _require_pgvector(self):
        if os.getenv("LANGFLOW_RUN_PGVECTOR_INTEGRATION_TESTS") != "1" or not os.getenv("PGVECTOR_CONNECTION_STRING"):
            pytest.skip("Set LANGFLOW_RUN_PGVECTOR_INTEGRATION_TESTS=1 and PGVECTOR_CONNECTION_STRING")
        pytest.importorskip("pgvector")

    async def test_moves_vectors_and_repoints_the_row(self, active_user, kb_root, tmp_path: Path):
        kb_name = f"kb_move_{uuid.uuid4().hex[:6]}"
        seeded = await _seed_chroma_kb(kb_root, active_user.username, kb_name, 12)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name=kb_name, model_selection={"name": "m", "provider": "p"}, chunks=12
        )
        target = create_backend(
            "postgres", kb_name=kb_name, kb_path=tmp_path, backend_config={}, user_id=active_user.id
        )
        try:
            results = await relocate_knowledge_bases(
                target_backend_type="postgres", target_backend_config={}, batch_size=5
            )
            result = next(r for r in results if r.kb_id == record.id)
            assert result.status == "relocated", result.reason
            assert (result.source_count, result.copied, result.target_count) == (12, 12, 12)

            row = await knowledge_base_service.get_by_id(record.id)
            assert row.backend_type == "postgres"

            moved = {}
            async for batch in target.iter_documents(include_embeddings=True):
                moved.update({d.id: d for d in batch})
            for doc in seeded:
                assert moved[doc.id].content == doc.content
                assert moved[doc.id].embedding == pytest.approx(doc.embedding)

            # The row now names the target, so a second run leaves it alone.
            rerun = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={})
            assert next(r for r in rerun if r.kb_id == record.id).status == "skipped"
        finally:
            with contextlib.suppress(Exception):
                await target.delete_collection()
            await target.teardown()

    async def test_vectors_that_are_not_unit_length_are_not_moved_to_another_metric(self, active_user, kb_root):
        # Chroma ranks by l2 and pgvector by cosine. For vectors that are not unit
        # length the two disagree on neighbours, and every count would still match.
        kb_name = f"kb_metric_{uuid.uuid4().hex[:6]}"
        await _seed_chroma_kb(kb_root, active_user.username, kb_name, 6, unit=False)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name=kb_name, model_selection={"name": "m", "provider": "p"}, chunks=6
        )

        for dry_run in (True, False):
            results = await relocate_knowledge_bases(
                target_backend_type="postgres", target_backend_config={}, dry_run=dry_run
            )
            result = next(r for r in results if r.kb_id == record.id)
            assert result.status == "failed"
            assert "ranks by l2 distance and the target by cosine" in result.reason
            assert "not unit length" in result.reason
            assert result.copied == 0

        assert (await knowledge_base_service.get_by_id(record.id)).backend_type == "chroma"

    async def test_unit_length_vectors_move_with_a_warning_about_scores(self, active_user, kb_root, tmp_path: Path):
        kb_name = f"kb_unit_{uuid.uuid4().hex[:6]}"
        await _seed_chroma_kb(kb_root, active_user.username, kb_name, 6)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name=kb_name, model_selection={"name": "m", "provider": "p"}, chunks=6
        )
        target = create_backend(
            "postgres", kb_name=kb_name, kb_path=tmp_path, backend_config={}, user_id=active_user.id
        )
        try:
            results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={})
            result = next(r for r in results if r.kb_id == record.id)
            assert result.status == "relocated", result.reason
            assert any("scores change scale" in warning for warning in result.warnings), result.warnings
        finally:
            with contextlib.suppress(Exception):
                await target.delete_collection()
            await target.teardown()

    async def test_dry_run_counts_without_writing(self, active_user, kb_root):
        kb_name = f"kb_dry_{uuid.uuid4().hex[:6]}"
        await _seed_chroma_kb(kb_root, active_user.username, kb_name, 7)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name=kb_name, model_selection={"name": "m", "provider": "p"}, chunks=7
        )

        results = await relocate_knowledge_bases(target_backend_type="postgres", target_backend_config={}, dry_run=True)

        result = next(r for r in results if r.kb_id == record.id)
        assert result.status == "would_relocate", result.reason
        assert result.source_count == 7
        row = await knowledge_base_service.get_by_id(record.id)
        assert row.backend_type == "chroma"

    async def _move(self, active_user, kb_root, tmp_path, kb_name, *, target_setup=None, on_record=None, batch_size=4):
        await _seed_chroma_kb(kb_root, active_user.username, kb_name, 12)
        record = await knowledge_base_service.create_record(
            user_id=active_user.id, name=kb_name, model_selection={"name": "m", "provider": "p"}, chunks=12
        )
        if on_record:
            on_record(record)
        target = create_backend(
            "postgres", kb_name=kb_name, kb_path=tmp_path, backend_config={}, user_id=active_user.id
        )
        try:
            if target_setup:
                await target_setup(target)
            results = await relocate_knowledge_bases(
                target_backend_type="postgres", target_backend_config={}, batch_size=batch_size
            )
            return record, next(r for r in results if r.kb_id == record.id)
        finally:
            with contextlib.suppress(Exception):
                await target.delete_collection()
            await target.teardown()

    async def test_chunk_written_to_the_source_mid_copy_stops_the_repoint(
        self, active_user, kb_root, tmp_path, during_copy
    ):
        kb_name = f"kb_late_{uuid.uuid4().hex[:6]}"

        async def write_late_chunk():
            source = ChromaLocalBackend(kb_name=kb_name, kb_path=kb_root / active_user.username / kb_name)
            try:
                await source.add_embedded_documents(
                    [IngestedDocument(id="late", content="late", metadata={}, embedding=[0.5] * DIM)]
                )
            finally:
                await source.teardown()

        during_copy(write_late_chunk)
        record, result = await self._move(active_user, kb_root, tmp_path, kb_name)

        assert result.status == "failed", (result.source_count, result.copied, result.target_count)
        assert "changed" in result.reason
        row = await knowledge_base_service.get_by_id(record.id)
        assert row.backend_type == "chroma"

    async def test_target_that_already_holds_other_chunks_is_not_repointed(self, active_user, kb_root, tmp_path):
        async def add_stale_rows(target):
            await target.ensure_ready()
            await target.add_embedded_documents(
                [
                    IngestedDocument(id=f"stale-{i}", content="stale", metadata={}, embedding=[0.9] * DIM)
                    for i in range(3)
                ]
            )

        kb_name = f"kb_stale_{uuid.uuid4().hex[:6]}"
        record, result = await self._move(active_user, kb_root, tmp_path, kb_name, target_setup=add_stale_rows)

        assert result.status == "failed", (result.source_count, result.copied, result.target_count)
        assert "15" in result.reason
        row = await knowledge_base_service.get_by_id(record.id)
        assert row.backend_type == "chroma"

    async def test_kb_deleted_mid_copy_is_not_reported_as_relocated(self, active_user, kb_root, tmp_path, during_copy):
        kb_name = f"kb_gone_{uuid.uuid4().hex[:6]}"
        records = []

        async def delete_row():
            await knowledge_base_service.delete_record(records[0].id)

        during_copy(delete_row)
        _, result = await self._move(active_user, kb_root, tmp_path, kb_name, on_record=records.append)

        assert result.status == "failed"
        assert "deleted" in result.reason
