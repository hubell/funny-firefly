# SPDX-License-Identifier: GPL-3.0-only
"""The two properties the re-opened encoding decision rests on.

The decision that ships the shrunk encoding rests on a quantity no table
printed before it -- the strata pooled at the *ranking* set's bucket mix
rather than at December's -- and on the encoding counting as geometry when
an ablation asks what removing the geometry costs. Both are one line of
arithmetic or one set membership, and both are load-bearing, which is the
combination that gets a check here rather than a comment.

The pooling identity is the important one. At a split's own bucket row
counts the pooled figure *is* that split's overall RMSE: the weighted mean
of the per-bucket mean squared errors is the pooled mean squared error. That
makes the December column a reconciliation against a number the ladder
already printed, and it is what licenses reading the ranking column as the
same quantity at different weights. Get the pooling wrong -- pool the RMSEs
linearly rather than in quadrature, say -- and the December column stops
agreeing with the ladder, which is exactly the failure this asserts.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


_MODULE = None


def _run_baselines():
    """Load the report generator as a module, without running the pipeline.

    ``scripts/`` is not an importable package, so the generator is loaded from
    its path. Cached, because executing it appends to ``sys.path`` and the
    tests below would otherwise stack a duplicate entry per call.
    """
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location(
            "_run_baselines", REPO / "scripts" / "run_baselines.py"
        )
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


# December 2025's strata, as reports/baseline.md prints them: the bucket row
# counts, and the control rung's RMSE in each bucket. The ladder reports that
# rung at 235.7 s ex-flagged over the same rows.
DECEMBER_ROWS = (476.0, 1063.0, 13048.0, 151002.0)
CONTROL_RMSE = (235.4, 353.4, 274.4, 231.0)
CONTROL_POOLED = 235.7

# The ranking set, as reports/geometry_gaps.md counts it in *How much of the
# ranking set is thin*: rows per frequency bucket, and their total.
RANKING_ROWS = (1115.0, 2946.0, 17153.0, 194662.0)
RANKING_DEPARTURES = 215876


def test_pooling_at_a_splits_own_mix_reproduces_that_splits_rmse():
    """The December column is a reconciliation, not a second measurement."""
    m = _run_baselines()
    from taxiout.validation import FREQUENCY_BUCKETS

    rmse = dict(zip(FREQUENCY_BUCKETS, CONTROL_RMSE))
    december = dict(zip(FREQUENCY_BUCKETS, DECEMBER_ROWS))
    assert round(m._pooled_at(rmse, december), 1) == CONTROL_POOLED


def test_pooling_is_quadrature_and_not_a_weighted_mean_of_rmses():
    """The failure mode worth naming: pooling the RMSEs linearly.

    A weighted mean of the per-bucket RMSEs is close enough to look right and
    is wrong, because RMSE is not linear in the errors. Here it lands at
    235.2 against the ladder's 235.7 -- a fifth of the 2.4 s the decision
    turns on, so not a rounding difference, and it fails the reconciliation
    that makes the December column worth printing at all.
    """
    m = _run_baselines()
    from taxiout.validation import FREQUENCY_BUCKETS

    rmse = dict(zip(FREQUENCY_BUCKETS, CONTROL_RMSE))
    december = dict(zip(FREQUENCY_BUCKETS, DECEMBER_ROWS))
    total = sum(DECEMBER_ROWS)
    linear = sum(w * r for w, r in zip(DECEMBER_ROWS, CONTROL_RMSE)) / total
    assert round(linear, 1) != CONTROL_POOLED
    assert round(m._pooled_at(rmse, december), 1) == CONTROL_POOLED
    # Quadrature is the larger of the two whenever the buckets differ, by
    # Jensen; that ordering is the sign the right one is being used.
    assert m._pooled_at(rmse, december) > linear


def test_the_ranking_mix_matches_the_report_it_was_taken_from():
    """The constant against its source, so neither can drift alone.

    ``RANKING_MIX`` is the one quantity in the encoding section that
    ``run_baselines.py`` does not compute; it is read from
    ``reports/geometry_gaps.md``, which a different generator writes. A
    sidecar file shared between the two would couple their run order for very
    little -- these are raw-triple counts and do not move when the OSM
    matrices are rebuilt. Parsing the report instead is one-way and enough: a
    re-issued ranking parquet, or a regenerated report, fails here rather than
    leaving the two quietly disagreeing.
    """
    m = _run_baselines()
    from taxiout.validation import FREQUENCY_BUCKETS

    text = (REPO / "reports" / "geometry_gaps.md").read_text(encoding="utf-8")
    heading = "### How much of the ranking set is thin"
    assert heading in text, "the ranking-mix table has moved or been renamed"
    counts: dict[str, float] = {}
    for line in text.split(heading, 1)[1].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in FREQUENCY_BUCKETS:
            counts[cells[0]] = float(cells[1].replace(",", ""))
        elif counts and not line.startswith("|"):
            break
    assert tuple(counts) == tuple(FREQUENCY_BUCKETS), counts
    assert counts == m.RANKING_MIX


def test_the_ranking_mix_is_the_row_counts_its_source_report_prints():
    """It is an external constant, so it must at least be well formed.

    Keyed to ``FREQUENCY_BUCKETS`` so it cannot drift from the labels the
    strata table prints, and held as ranking *row counts* rather than as the
    rounded percentages printed beside them in `geometry_gaps.md`: those are
    two-figure and need not sum to one, while the counts sum to the ranking
    set exactly and are what those percentages are computed from.
    """
    m = _run_baselines()
    from taxiout.validation import FREQUENCY_BUCKETS

    assert tuple(m.RANKING_MIX) == tuple(FREQUENCY_BUCKETS)
    # Per bucket, not only in total: a permutation of the two largest weights
    # sums the same and moves every printed pooled figure by tens of seconds.
    assert tuple(m.RANKING_MIX.values()) == RANKING_ROWS
    assert sum(RANKING_ROWS) == RANKING_DEPARTURES
    assert all(v == int(v) for v in m.RANKING_MIX.values())
    # Normalised at the point of use: a constant RMSE pools to itself at any
    # weights only if the weights are divided by their sum.
    flat = dict.fromkeys(FREQUENCY_BUCKETS, 1.0)
    for w in (m.RANKING_MIX, flat):
        assert math.isclose(
            m._pooled_at(dict.fromkeys(FREQUENCY_BUCKETS, 300.0), w), 300.0
        )


def test_the_ranking_mix_is_thinner_than_decembers_where_it_matters():
    """The reason the column exists, asserted rather than left to prose."""
    m = _run_baselines()
    from taxiout.validation import FREQUENCY_BUCKETS

    dec_total, rank_total = sum(DECEMBER_ROWS), sum(m.RANKING_MIX.values())
    for b, n in zip(FREQUENCY_BUCKETS[:2], DECEMBER_ROWS[:2]):
        assert m.RANKING_MIX[b] / rank_total > n / dec_total


def test_the_encoding_counts_as_geometry_for_an_ablation():
    """Condition on shipping the encoding: "geometry out" must remove it.

    The encoding shrinks toward ``g_hat``, so a rung carrying it carries the
    geometry with an extra step in front. An ablation that called itself
    geometry-free while the encoding stayed in would report a contrast that
    shrank by its own definition rather than by a finding.
    """
    from taxiout.encoding import ENCODING_FEATURES
    from taxiout.features import GEOMETRY_FEATURES
    from taxiout.models import ENCODING_ARMS, LADDER_BY_KEY

    # The encoding is its own block: this is why the classification needed
    # saying rather than following from the feature lists.
    assert not set(ENCODING_FEATURES) & set(GEOMETRY_FEATURES)
    for key in ENCODING_ARMS:
        assert LADDER_BY_KEY[key].uses_geometry
    # A rung carrying the encoding and nothing else geometric still counts.
    bare = LADDER_BY_KEY["identifiers"]
    assert not bare.uses_geometry
    assert bare.__class__(
        key="probe",
        label="probe",
        kind="booster",
        features=tuple(bare.features) + tuple(ENCODING_FEATURES),
    ).uses_geometry


def test_the_claim_contrast_carries_no_encoding_either_side():
    """Which is why the +14.7 s figure does not move with this decision."""
    m = _run_baselines()
    from taxiout.encoding import ENCODING_FEATURES
    from taxiout.models import LADDER_BY_KEY

    for key in m.CLAIM:
        assert not set(LADDER_BY_KEY[key].features) & set(ENCODING_FEATURES)
