"""Sanity-check our $jsonSchema definitions offline.

This converts bsonType -> JSON Schema type and validates sample documents. It catches
typos and wrong required-field lists, but it is NOT a substitute for running against
a real MongoDB server, which is what actually enforces the validator.
"""
from datetime import datetime, timezone

import jsonschema
import pytest

from app.database.validation import PAPERS_SCHEMA, USERS_SCHEMA

TYPE_MAP = {"string": "string", "array": "array", "object": "object", "bool": "boolean",
            "int": "integer", "null": "null"}


def to_json_schema(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "bsonType":
                types = v if isinstance(v, list) else [v]
                mapped = [TYPE_MAP[t] for t in types if t in TYPE_MAP]
                if mapped and len(mapped) == len(types):
                    out["type"] = mapped if len(mapped) > 1 else mapped[0]
            else:
                out[k] = to_json_schema(v)
        return out
    if isinstance(node, list):
        return [to_json_schema(i) for i in node]
    return node


NOW = datetime.now(timezone.utc)
GOOD_PAPER = {"owner_id": "x", "title": "T", "authors": ["A"], "reading_status": "unread",
              "processing_status": "none", "created_at": NOW, "updated_at": NOW}


def check(schema, doc):
    # dates/objectIds have no JSON type; strip them before checking the rest
    skip = {"created_at", "updated_at", "owner_id"}
    jsonschema.validate({k: v for k, v in doc.items() if k not in skip}, to_json_schema(
        {**schema, "required": [r for r in schema["required"] if r not in skip]}))


def test_good_paper_passes():
    check(PAPERS_SCHEMA, GOOD_PAPER)


@pytest.mark.parametrize("bad", [
    {"title": ""}, {"reading_status": "done"}, {"year": 1500}, {"authors": "Smith"},
    {"processing_status": "weird"},
])
def test_bad_papers_fail(bad):
    with pytest.raises(jsonschema.ValidationError):
        check(PAPERS_SCHEMA, {**GOOD_PAPER, **bad})


def test_required_fields_enforced():
    doc = dict(GOOD_PAPER)
    del doc["title"]
    with pytest.raises(jsonschema.ValidationError):
        check(PAPERS_SCHEMA, doc)


def test_user_email_pattern():
    base = {"email": "a@b.com", "password_hash": "h", "full_name": "n", "created_at": NOW}
    check(USERS_SCHEMA, base)
    with pytest.raises(jsonschema.ValidationError):
        check(USERS_SCHEMA, {**base, "email": "nope"})
