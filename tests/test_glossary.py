"""Glossary: every option a user can pick has an entry, so recommendations can always be explained."""
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algorithms  # noqa: E402
import glossary  # noqa: E402
import rewards  # noqa: E402
import vision as visions  # noqa: E402


def test_every_registry_option_has_an_entry():
    ids = {e["id"] for e in glossary.entries()}
    for algo in algorithms.ALGORITHMS.values():
        assert algo.name in ids
        assert {f"{algo.name}.{p['name']}" for p in algo.params} <= ids
    for v in visions.VISIONS.values():
        assert f"vision.{v.name}" in ids and {f"vision.{v.name}.{p['name']}" for p in v.params} <= ids
    assert {f"reward.{f}" for f in rewards.FEATURES} <= ids
    assert {f"preset.{p}" for p in rewards.PRESETS} <= ids


def test_entries_are_complete_and_unambiguous():
    entries = glossary.entries()
    ids = {e["id"] for e in entries}
    aliases = collections.Counter(a.lower() for e in entries for a in e["aliases"])
    assert [a for a, n in aliases.items() if n > 1] == [], "an alias must point to one entry"
    assert not aliases.keys() & glossary.GENERIC, "everyday words would link terms in unrelated sentences"
    for e in entries:
        assert e["short"] and e["short"].lower() != e["term"].lower(), e["id"]
        assert e["long"] or e["up"], f"{e['id']} needs an explanation"
        assert set(e["related"]) <= ids
        if e["kind"] == "setting" and e["group"] in algorithms.ALGORITHMS:
            assert e["up"] and e["down"], f"{e['id']} should say what raising / lowering it does"
