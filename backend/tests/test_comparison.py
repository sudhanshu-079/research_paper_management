import json

import pytest

from app.services import comparison_service
from app.services import llm as llm_module
from tests.conftest import make_pdf
from tests.test_search import GENOMICS, VISION, upload


class ScriptedLLM:
    """Replies depend on which prompt it is given (field extraction vs observations)."""
    name = "scripted"

    def __init__(self, extract, observe):
        self.extract, self.observe, self.calls = extract, observe, []

    def generate(self, system, user):
        self.calls.append((system, user))
        reply = self.observe if "compare research papers" in system else self.extract
        reply = reply(user) if callable(reply) else reply
        return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture(autouse=True)
def reset_llm():
    yield
    llm_module.override(None)


@pytest.fixture()
def two(client, auth):
    g = upload(client, auth, "Tumour Genomics Study", "10.1/g", GENOMICS)
    v = upload(client, auth, "Image Classification with CNNs", "10.1/v", VISION)
    return g, v


def create(client, headers, ids, **kw):
    return client.post("/comparisons", json={"paper_ids": list(ids), **kw}, headers=headers)


GOOD_EXTRACT = {"fields": {
    "objectives": {"text": "Measure gene expression in tumours.", "citations": ["S1"]},
    "methodology": {"text": None, "citations": []},
    "limitations": {"text": "Proven beyond doubt.", "citations": ["S9"]},        # fake citation only
    "results": {"text": "Treatment groups differ.", "citations": ["S1", "S7"]},   # one real, one fake
}}
GOOD_OBSERVE = {"observations": [
    {"text": "Both papers state objectives.", "cites": [{"paper": 1, "field": "objectives"}, {"paper": 2, "field": "objectives"}]},
    {"text": "Cites a paper that doesn't exist.", "cites": [{"paper": 9, "field": "objectives"}]},
    {"text": "Cites an empty cell.", "cites": [{"paper": 1, "field": "methodology"}]},
    {"text": "Cites nothing.", "cites": []},
]}


def test_llm_comparison_validates_cells_and_observations(client, auth, two):
    fake = ScriptedLLM(GOOD_EXTRACT, GOOD_OBSERVE)
    llm_module.override(fake)
    r = create(client, auth, two)
    assert r.status_code == 201, r.text
    comp = r.json()
    assert len(fake.calls) == 3                                   # 2 papers + observations
    assert [c["title"] for c in comp["columns"]] == ["Tumour Genomics Study", "Image Classification with CNNs"]
    rows = {row["field"]: row for row in comp["rows"]}
    assert len(rows) == 10

    obj = rows["objectives"]["cells"][0]
    assert obj["status"] == "found" and obj["citations"][0]["page"] == 1
    assert obj["citations"][0]["paper_title"] == "Tumour Genomics Study"
    assert rows["methodology"]["cells"][0]["status"] == "not_found"
    lim = rows["limitations"]["cells"][0]
    assert lim["status"] == "unsupported" and lim["text"] is None          # fabricated citation => dropped
    res = rows["results"]["cells"][0]
    assert res["status"] == "found" and [c["label"] for c in res["citations"]] == ["S1"] and res["warnings"]
    assert rows["datasets"]["cells"][0]["status"] == "not_found"            # model didn't mention it

    assert len(comp["observations"]) == 1                                   # 3 bad ones dropped
    assert {c["paper_title"] for c in comp["observations"][0]["cites"]} == {"Tumour Genomics Study", "Image Classification with CNNs"}
    assert len(comp["warnings"]) == 3


def test_each_papers_prompt_contains_only_its_own_text(client, auth, two):
    fake = ScriptedLLM(GOOD_EXTRACT, GOOD_OBSERVE)
    llm_module.override(fake)
    create(client, auth, two)
    first, second = fake.calls[0][1], fake.calls[1][1]
    assert "tumour genomes" in first and "convolutional" not in first
    assert "convolutional" in second and "tumour genomes" not in second


def test_extractive_mode_without_llm(client, auth, two):
    comp = create(client, auth, two).json()
    assert comp["mode"] == "extractive" and comp["observations"] == []
    statuses = {c["status"] for row in comp["rows"] for c in row["cells"]}
    assert statuses <= {"extractive", "not_found"} and "extractive" in statuses
    for row in comp["rows"]:
        for c in row["cells"]:
            if c["status"] == "extractive":
                assert c["citations"][0]["paper_title"] in {"Tumour Genomics Study", "Image Classification with CNNs"}


def test_no_evidence_means_no_llm_call():
    class Boom:
        name = "boom"
        def generate(self, s, u): raise AssertionError("should not be called")
    from app.services.comparison_service import _extract_with_llm
    # direct guard check through compare(): patch evidence to empty
    import app.services.comparison_service as cs
    cs._paper_evidence, orig = (lambda *a: []), cs._paper_evidence
    llm_module.override(Boom())
    try:
        class FakeDB: pass
        out = cs.compare(FakeDB(), "owner", [{"_id": 1, "title": "T"}, {"_id": 2, "title": "U"}])
    finally:
        cs._paper_evidence = orig
    assert all(c["status"] == "not_found" for row in out["rows"] for c in row["cells"])


def test_long_papers_use_per_field_retrieval(client, auth, two, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "comparison_context_chars", 10)
    fake = ScriptedLLM(GOOD_EXTRACT, GOOD_OBSERVE)
    llm_module.override(fake)
    assert create(client, auth, two).status_code == 201   # path runs end-to-end without error


def test_bad_model_output_is_502_and_nothing_saved(client, auth, two, db):
    llm_module.override(ScriptedLLM("not json", "not json"))
    assert create(client, auth, two).status_code == 502
    assert db.research_comparisons.count_documents({}) == 0


# ---- validation and access control ----
def test_selection_rules(client, auth, other_auth, two):
    g, v = two
    assert create(client, auth, [g]).status_code == 422
    assert create(client, auth, [g, g]).status_code == 422
    assert create(client, auth, [g, v, g, v, g, v]).status_code == 422          # more than 5
    assert create(client, other_auth, [g, v]).status_code == 404                # not their papers
    assert create(client, auth, [g, "garbage"]).status_code == 404


def test_paper_without_text_is_rejected(client, auth, two):
    scanned = client.post("/papers/upload", files={"file": ("s.pdf", make_pdf(with_text=False), "application/pdf")},
                          headers=auth).json()["id"]
    r = create(client, auth, [two[0], scanned])
    assert r.status_code == 409


# ---- save / list / export / privacy ----
def test_save_list_get_export_delete(client, auth, two):
    llm_module.override(ScriptedLLM(GOOD_EXTRACT, GOOD_OBSERVE))
    comp = create(client, auth, two, title="My comparison").json()
    cid = comp["id"]
    assert client.get("/comparisons", headers=auth).json()[0]["title"] == "My comparison"
    assert client.get(f"/comparisons/{cid}", headers=auth).json()["rows"] == comp["rows"]

    md = client.get(f"/comparisons/{cid}/export", headers=auth)
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
    assert "| Field | Tumour Genomics Study | Image Classification with CNNs |" in md.text
    assert "Measure gene expression in tumours. (p. 1)" in md.text and "## Observations" in md.text
    js = client.get(f"/comparisons/{cid}/export", params={"format": "json"}, headers=auth)
    assert json.loads(js.text)["id"] == cid
    assert client.get(f"/comparisons/{cid}/export", params={"format": "pdf"}, headers=auth).status_code == 422

    assert client.delete(f"/comparisons/{cid}", headers=auth).status_code == 204
    assert client.get(f"/comparisons/{cid}", headers=auth).status_code == 404


def test_comparisons_are_private(client, auth, other_auth, two):
    cid = create(client, auth, two).json()["id"]
    for path in (f"/comparisons/{cid}", f"/comparisons/{cid}/export"):
        assert client.get(path, headers=other_auth).status_code == 404
    assert client.delete(f"/comparisons/{cid}", headers=other_auth).status_code == 404
    assert client.get("/comparisons", headers=other_auth).json() == []
    assert client.get("/comparisons").status_code == 401


def test_markdown_escapes_table_breaking_characters():
    comp = {"title": "T", "model": None, "columns": [{"title": "A|B"}, {"title": "C"}],
            "rows": [{"label": "Methodology", "cells": [
                {"text": "uses a|b\nnewline", "status": "found", "citations": [{"page": 3}]},
                {"text": None, "status": "not_found", "citations": []}]}], "observations": []}
    md = comparison_service.to_markdown(comp)
    assert "a\\|b newline (p. 3)" in md and "A/B" in md and "_not found in the text_" in md
