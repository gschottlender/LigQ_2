from argparse import Namespace

import pytest

from run_ligq_2 import parse_args, resolve_search_threshold


def _args(
    search_representation="morgan_1024_r2",
    search_threshold=None,
    known_only=False,
    bsi=False,
):
    return Namespace(
        search_representation=search_representation,
        search_threshold=search_threshold,
        known_only=known_only,
        bsi=bsi,
    )


def test_resolves_morgan_default_threshold():
    args = _args(search_representation="morgan_1024_r2")

    resolve_search_threshold(args)

    assert args.search_threshold == pytest.approx(0.415094)


def test_resolves_chemberta_default_threshold():
    args = _args(search_representation="chemberta_zinc_base_768")

    resolve_search_threshold(args)

    assert args.search_threshold == pytest.approx(0.936140)


def test_explicit_threshold_overrides_representation_default():
    args = _args(
        search_representation="morgan_1024_r2",
        search_threshold=0.2,
    )

    resolve_search_threshold(args)

    assert args.search_threshold == pytest.approx(0.2)


def test_legacy_zinc_threshold_alias_overrides_representation_default(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ligq_2.py",
            "--input-fasta",
            "queries.fasta",
            "--output-dir",
            "results",
            "--zinc-search-threshold",
            "0.2",
        ],
    )
    args = parse_args()

    resolve_search_threshold(args)

    assert args.search_threshold == pytest.approx(0.2)


def test_unknown_representation_requires_explicit_threshold():
    args = _args(search_representation="custom_rep")

    with pytest.raises(ValueError, match="Pass --search-threshold explicitly"):
        resolve_search_threshold(args)


def test_known_only_does_not_require_threshold():
    args = _args(
        search_representation="custom_rep",
        known_only=True,
    )

    resolve_search_threshold(args)

    assert args.search_threshold is None


def test_bsi_does_not_use_search_threshold_defaults():
    args = _args(
        search_representation="custom_rep",
        bsi=True,
    )

    resolve_search_threshold(args)

    assert args.search_threshold is None
