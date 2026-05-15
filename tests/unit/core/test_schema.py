"""Validate that every registered method ships a MethodSchema and round-trips through JSON."""

from __future__ import annotations

import json

from nanook.core import METHOD_REGISTRY, list_method_schemas


def test_every_method_has_a_schema() -> None:
    for name, cls in METHOD_REGISTRY.items():
        s = getattr(cls, "_schema", None)
        assert s is not None, f"{name}: no @schema attached"
        assert s.name == name
        assert s.display_name, f"{name}: missing display_name"
        assert s.category, f"{name}: missing category"
        assert s.applicable_dtypes, f"{name}: missing applicable_dtypes"
        assert s.description, f"{name}: missing description"


def test_every_param_is_well_formed() -> None:
    for name, cls in METHOD_REGISTRY.items():
        for p in cls._schema.params:  # type: ignore[attr-defined]
            assert p.name, f"{name}: empty param name"
            assert p.display_name, f"{name}.{p.name}: missing display_name"
            assert p.param_type in {
                "INT",
                "FLOAT",
                "BOOL",
                "STRING",
                "CODE",
                "RANGE",
                "LIST",
                "MAP",
            }, f"{name}.{p.name}: invalid param_type {p.param_type!r}"
            if p.param_type == "CODE":
                assert p.code_options, f"{name}.{p.name}: CODE param must declare code_options"


def test_catalogue_is_json_serialisable() -> None:
    catalogue = list_method_schemas()
    payload = json.dumps(catalogue)
    parsed = json.loads(payload)
    assert isinstance(parsed, list)
    assert len(parsed) == len(METHOD_REGISTRY)
    names = {entry["code"] for entry in parsed}
    assert names == set(METHOD_REGISTRY.keys())


def test_catalogue_entries_have_expected_keys() -> None:
    for entry in list_method_schemas():
        assert set(entry.keys()) == {
            "code",
            "displayName",
            "category",
            "applicableTypes",
            "requiresPreScan",
            "dropsColumn",
            "requiresQuasiIdentifiers",
            "requiresSensitive",
            "description",
            "params",
        }
        for p in entry["params"]:
            assert set(p.keys()) == {
                "code",
                "displayName",
                "paramType",
                "defaultValue",
                "codeOptions",
                "required",
                "description",
            }


def test_quasi_identifier_required_for_qi_methods() -> None:
    qi_required = {
        entry["code"] for entry in list_method_schemas() if entry["requiresQuasiIdentifiers"]
    }
    assert qi_required == {"massc", "local_suppression"}


def test_no_method_currently_requires_sensitive() -> None:
    sensitive_required = {
        entry["code"] for entry in list_method_schemas() if entry["requiresSensitive"]
    }
    assert sensitive_required == set()
