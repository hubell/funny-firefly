#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The plan-shifted departures: the rule of the pre-registration of 2 October 2026.

The rule, as the pre-registration of 2 October fixes it
(``reports/preregistered-2026-10-02.md``):

- **The class.** A departure, training or ranking, not covered by the
  record-quality term of ``taxiout.tails``, whose take-off minus ``EOBT_1_flt``
  and take-off minus ``LOBT_flt`` both exceed ``TAIL_EXCLUSION_ADOPTED_H``.
  Both estimates must be present.
- **Long.** A training departure of the class whose recorded taxi-out exceeds
  the same four hours.
- **The share.** ``(long + 1) / (n + 2)`` over the class's departures in the
  rows a fit is given.
- **The lag.** Take-off minus ``LOBT_flt``.
- **The rest mean.** The mean recorded taxi-out of the training departures at
  the same airport that have a flight-plan record and a recorded taxi-out of
  four hours or less, in the rows a fit is given.
- **The prediction.** ``share * lag + (1 - share) * rest_mean``, substituted
  for the booster's prediction on the class's rows, after the term and before
  the rounding and the clip at zero.

It reads ``EOBT_1_flt``, ``LOBT_flt``, the take-off time, the departure airport
and, at fit time, the recorded taxi-out; never ``AOBT_3_flt``. "A flight-plan
record" is read as ``taxiout.tails`` reads it: ``LOBT_flt`` present.

The three functions below are the rule as it was evaluated, before it moved
into ``taxiout.tails`` (``plan_shifted`` and ``PlanShiftModel``, which the
submission uses). Here they are evaluated on the twelve held-out months and
the two forward legs, on the stored residuals of ``scripts/report_monthly.py``
(the held-out files ``scripts/report_monthly_ladder.py`` reproduced, and the
forward files checked against the ladder's), and the report says what that
evaluation is: the arithmetic on three training rows, not independent
evidence.

Usage::

    python scripts/report_planshift.py --stops    the pre-registration's first two stops
    python scripts/report_planshift.py --report   the evaluation, to reports/planshift.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import monthly as c  # noqa: E402
from monthly import rm, rb  # noqa: E402
from taxiout.models import TAIL_EXCLUSION_ADOPTED_H  # noqa: E402
from taxiout.provenance import FROZEN_DOCUMENTS  # noqa: E402
from taxiout.tails import TailModel  # noqa: E402

# The pre-registration as the repository publishes it, frozen at the digest
# ``taxiout.provenance.FROZEN_DOCUMENTS`` records, and the digest of the working
# note it was copied from, which it records and which fixed the order.
PREREG_NAME = "preregistered-2026-10-02.md"
PREREG_SHA256 = "d62b48f32d3ba9da07aa346cb04a2560feaff9664a9734228d7523e669dda9a1"
THRESHOLD_S = float(TAIL_EXCLUSION_ADOPTED_H) * 3600.0
# What the pre-registration was written knowing: a different count is a stop.
EXPECTED_TRAINING = {"n": 3, "long": 2}
EXPECTED_RANKING = {"n": 1, "icao": "LIRF", "month": "2026-07"}
DEFAULT_OUT = c.REPO / "reports" / "planshift.md"
READS = ("EOBT_1_flt", "LOBT_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt")


# --------------------------------------------------------------------------
# the rule


def planshift_class(frame: pd.DataFrame, covered) -> pd.Series:
    """The class: not covered by the term, both estimates present, both lags over 4 h."""
    t = c.utc(frame["MVT_TIME_UTC_mvt"])
    lag_eobt = (t - c.utc(frame["EOBT_1_flt"])).dt.total_seconds()
    lag_lobt = (t - c.utc(frame["LOBT_flt"])).dt.total_seconds()
    present = frame["EOBT_1_flt"].notna() & frame["LOBT_flt"].notna()
    out = (
        ~pd.Series(np.asarray(covered, dtype=bool), index=frame.index)
        & present
        & (lag_eobt > THRESHOLD_S)
        & (lag_lobt > THRESHOLD_S)
    )
    return out.fillna(False).astype(bool).rename("planshift")


@dataclass
class PlanShift:
    n: int = 0
    long: int = 0
    share: float = float("nan")
    rest_mean: dict[str, float] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, covered, y) -> "PlanShift":
        """On the rows a fit is given: ``frame`` is those rows, ``covered`` the
        term's mask on them, ``y`` their recorded taxi-out."""
        y = pd.to_numeric(
            pd.Series(np.asarray(y, dtype=float), index=frame.index), errors="coerce"
        )
        cls = planshift_class(frame, covered)
        self.n = int(cls.sum())
        self.long = int((cls & (y > THRESHOLD_S)).sum())
        self.share = (self.long + 1.0) / (self.n + 2.0)
        record = frame["LOBT_flt"].notna()
        rest = record & y.notna() & (y <= THRESHOLD_S)
        icao = frame["ADEP_mvt"].astype(str)
        self.rest_mean = {
            str(k): float(v) for k, v in y[rest].groupby(icao[rest]).mean().items()
        }
        return self

    def predict(self, frame: pd.DataFrame, covered) -> pd.Series:
        """The rule on the class's rows, NaN elsewhere."""
        cls = planshift_class(frame, covered)
        lag = (
            c.utc(frame["MVT_TIME_UTC_mvt"]) - c.utc(frame["LOBT_flt"])
        ).dt.total_seconds()
        rest = frame["ADEP_mvt"].astype(str).map(self.rest_mean).astype(float)
        value = self.share * lag + (1.0 - self.share) * rest
        return pd.Series(
            np.where(cls, value, np.nan), index=frame.index, name="planshift", dtype=float
        )


def apply_rule(prediction, rule) -> np.ndarray:
    """The rule substituted on its rows, after the term: ``prediction`` is the
    booster's or the term's, ``rule`` NaN outside the class."""
    r = np.asarray(rule, dtype=float)
    return np.where(np.isfinite(r), r, np.asarray(prediction, dtype=float))


# --------------------------------------------------------------------------
# the stops


def full_year_term(dep: pd.DataFrame) -> TailModel:
    return TailModel().fit(dep)


def class_rows(frame: pd.DataFrame, covered, y=None) -> pd.DataFrame:
    """Per class departure: airport, month, record, the three lags in hours, taxi-out."""
    cls = planshift_class(frame, covered)
    f = frame.loc[cls]
    t = c.utc(f["MVT_TIME_UTC_mvt"])
    out = pd.DataFrame(
        {
            "icao": f["ADEP_mvt"].astype(str).to_numpy(),
            "month": t.dt.strftime("%Y-%m").to_numpy(),
            "record": f["LOBT_flt"].notna().to_numpy(),
            "lag_eobt_h": (
                (t - c.utc(f["EOBT_1_flt"])).dt.total_seconds() / 3600.0
            ).to_numpy(),
            "lag_lobt_h": (
                (t - c.utc(f["LOBT_flt"])).dt.total_seconds() / 3600.0
            ).to_numpy(),
            "lag_sched_h": (
                (t - c.utc(f["SCHED_TIME_UTC_mvt"])).dt.total_seconds() / 3600.0
            ).to_numpy(),
        },
        index=f.index,
    )
    if y is not None:
        yy = np.asarray(y, dtype=float)[cls.to_numpy()]
        out["taxi_s"] = yy
        out["long"] = yy > THRESHOLD_S
    return out


def prereg_copy() -> Path:
    """The published copy of the pre-registration."""
    return c.REPO / "reports" / PREREG_NAME


def stops(log=print) -> tuple[bool, dict]:
    ok = True
    frozen = {f.path: f.sha256 for f in FROZEN_DOCUMENTS}[f"reports/{PREREG_NAME}"]
    digest = c.sha256(prereg_copy())
    same = digest == frozen and PREREG_SHA256 in prereg_copy().read_text(encoding="utf-8")
    log(
        f"    [{'ok ' if same else 'STOP'}] reports/{PREREG_NAME} sha256 {digest}, "
        f"recording the working note's {PREREG_SHA256}"
    )
    ok &= same
    dep = c.departures()
    y = c.target(dep)
    term = full_year_term(dep)
    covered = term.predict(dep).notna()
    train = class_rows(dep, covered, y)
    n, long = len(train), int(train["long"].sum())
    good = n == EXPECTED_TRAINING["n"] and long == EXPECTED_TRAINING["long"]
    log(
        f"    [{'ok ' if good else 'STOP'}] the class in the training year: {n} departures, {long} long "
        f"(the pre-registration: {EXPECTED_TRAINING['n']}, {EXPECTED_TRAINING['long']})"
    )
    ok &= good
    rank = c.ranking_departures()
    rank_covered = term.predict(rank).notna()
    ranked = class_rows(rank, rank_covered)
    good = (
        len(ranked) == EXPECTED_RANKING["n"]
        and set(ranked["icao"]) == {EXPECTED_RANKING["icao"]}
        and set(ranked["month"]) == {EXPECTED_RANKING["month"]}
    )
    log(
        f"    [{'ok ' if good else 'STOP'}] the class in the ranking set: {len(ranked)} departure(s), "
        f"{sorted(set(ranked['icao']))} {sorted(set(ranked['month']))} (the pre-registration: one, LIRF, July 2026)"
    )
    ok &= good
    return ok, {
        "dep": dep,
        "y": y,
        "term": term,
        "covered": covered,
        "train": train,
        "rank": rank,
        "rank_covered": rank_covered,
        "ranked": ranked,
    }


# --------------------------------------------------------------------------
# the evaluation


def legs() -> list[rm.Leg]:
    return rm.leg_plan()


def evaluate_leg(dep, y, month, leg: rm.Leg) -> dict:
    """One leg: the term and the rule fitted on its training rows, applied to its
    validation rows, against the held-out-month fits' stored residuals."""
    train = month.isin(leg.train_months).to_numpy()
    valid = (month == leg.valid_month).to_numpy()
    term = TailModel().fit(dep.loc[train])
    tail = term.predict(dep)
    covered = tail.notna()
    rule = PlanShift().fit(dep.loc[train], covered[train], y[train])
    pred_rule = rule.predict(dep.loc[valid], covered[valid])
    stored = pd.read_parquet(c.MONTHLY / f"{leg.stem}.parquet")
    ids = dep.loc[valid, "MVT_ID_mvt"].to_numpy()
    if len(stored) != len(ids) or not np.array_equal(stored["mvt_id"].to_numpy(), ids):
        raise SystemExit(f"{leg.name}: the stored rows are not the validation rows")
    if not np.array_equal(
        stored["flagged"].to_numpy(dtype=bool), covered[valid].to_numpy(dtype=bool)
    ):
        raise SystemExit(f"{leg.name}: the term's rows differ from the stored flags")
    yv = y[valid].to_numpy(dtype=float)
    booster = yv - stored["residual"].to_numpy(dtype=float)
    with_term = yv - stored[rm.TERM_COLUMN].to_numpy(dtype=float)
    r = pred_rule.to_numpy(dtype=float)
    cls = np.isfinite(r)
    d4 = yv <= THRESHOLD_S
    out = {
        "leg": leg.name,
        "kind": leg.kind,
        "valid_month": leg.valid_month,
        "rows": int(valid.sum()),
        "term_rows": int(covered[valid].sum()),
        "fit": {"n": rule.n, "long": rule.long, "share": rule.share},
        "class_rows": int(cls.sum()),
        "class": [],
        "d4_rows": int(d4.sum()),
    }
    for i in np.flatnonzero(cls):
        out["class"].append(
            {
                "icao": str(dep.loc[valid, "ADEP_mvt"].astype(str).to_numpy()[i]),
                "taxi_s": float(yv[i]),
                "rule": float(r[i]),
                "stored": float(booster[i]),
                "rest_mean": float(
                    rule.rest_mean.get(
                        str(dep.loc[valid, "ADEP_mvt"].astype(str).to_numpy()[i]), np.nan
                    )
                ),
            }
        )
    for name, base in (("term", with_term), ("booster", booster)):
        withr = apply_rule(base, r)
        out[f"sse_{name}"] = float(np.sum((yv - base) ** 2))
        out[f"sse_{name}_rule"] = float(np.sum((yv - withr) ** 2))
        out[f"sse_{name}_d4"] = float(np.sum((yv[d4] - base[d4]) ** 2))
        out[f"sse_{name}_rule_d4"] = float(np.sum((yv[d4] - withr[d4]) ** 2))
    return out


def rm_(sse: float, n: int) -> float:
    return float(np.sqrt(sse / n)) if n else float("nan")


def build_report(
    st: dict, results: list[dict], v6_pred: float, rank_rule: dict, inputs: list[Path]
) -> str:
    L: list[str] = []
    add = L.append
    add("# The plan-shifted departures, evaluated")
    add("")
    add(
        "Generated by `scripts/report_planshift.py` with `--report` from the training and ranking files, the "
        "record-quality term of `taxiout.tails` and the held-out-month fits' stored residuals "
        "(`reports/monthly.md`); every figure is computed there, and "
        "the generator refuses to write a movement identifier or a literal stand identifier. **It decides nothing.**"
    )
    add("")
    add("## The rule, as fixed on 2 October")
    add("")
    add(
        f"Implemented from the pre-registration of 2 October (the working note's sha256 `{PREREG_SHA256}`, published as "
        f"`reports/{PREREG_NAME}`, which was checked at its frozen digest before anything was read): the class is a departure not covered by the record-quality term whose take-off minus "
        f"`EOBT_1_flt` and minus `LOBT_flt` both exceed {TAIL_EXCLUSION_ADOPTED_H:g} h "
        "(`TAIL_EXCLUSION_ADOPTED_H`), both estimates present; *long* is a training departure of the class whose "
        "recorded taxi-out exceeds the same 4 h; the share is `(long + 1) / (n + 2)` over the class's departures "
        "in the rows a fit is given; the prediction is `share x lag + (1 - share) x rest_mean`, the lag being "
        "take-off minus `LOBT_flt` and the rest mean the mean recorded taxi-out of that airport's training "
        "departures with a flight-plan record and a recorded taxi-out of 4 h or less. It is substituted for the "
        "booster's prediction on the class's rows, after the term. It reads "
        + ", ".join(f"`{x}`" for x in READS)
        + " and, at fit time, the recorded taxi-out; never `AOBT_3_flt`. A flight-plan record is read as "
        "`taxiout.tails` reads it, `LOBT_flt` present, and the rest mean is taken over every such training row "
        "of 4 h or less, the term's rows included, as the pre-registration words it."
    )
    add("")
    # 1
    add("## The class in the training year")
    add("")
    add(
        "The record-quality term fitted on the whole training year, as the submission fits it; the class is read "
        "on the rows it does not cover. One row per departure, without identifiers:"
    )
    add("")
    rows = []
    for _, r in st["train"].sort_values(["month", "icao"]).iterrows():
        rows.append(
            [
                r["icao"],
                c.month_label(r["month"]) + " " + r["month"][:4],
                "yes" if r["record"] else "no",
                f"{r['lag_eobt_h']:.2f}",
                f"{r['lag_lobt_h']:.2f}",
                f"{r['lag_sched_h']:.2f}",
                f"{r['taxi_s']:,.0f}",
                "yes" if r["long"] else "no",
            ]
        )
    L += c.table(
        [
            "airport",
            "month",
            "flight-plan record",
            "take-off - EOBT_1 (h)",
            "take-off - LOBT (h)",
            "take-off - schedule (h)",
            "recorded taxi-out (s)",
            "long",
        ],
        rows,
        "llcrrrrc",
    )
    add("")
    n, long = len(st["train"]), int(st["train"]["long"].sum())
    add(
        f"{n} departures, {long} long: the pre-registration's count, so its second stop does not fire. "
        "The first stop, the ranking set, is in section 3."
    )
    add("")
    # 2
    fy = st["full_rule"]
    add("## The rule fitted on the full year")
    add("")
    add(
        f"`n` = {fy.n}, `long` = {fy.long}, share = ({fy.long} + 1) / ({fy.n} + 2) = {fy.share:.4f}. "
        "The rest mean per airport, over the training departures with a flight-plan record and a recorded "
        "taxi-out of 4 h or less:"
    )
    add("")
    L += c.table(
        ["airport", "rest mean (s)"],
        [[k, f"{v:,.1f}"] for k, v in sorted(fy.rest_mean.items())],
        "lr",
    )
    add("")
    # 3
    add("## The one ranking departure")
    add("")
    rr = rank_rule
    add(
        f"One departure of the ranking set is in the class: {rr['icao']}, {c.month_label(rr['month'])} "
        f"{rr['month'][:4]}, take-off minus `LOBT_flt` {rr['lag_lobt_s']:,.0f} s ({rr['lag_lobt_s'] / 3600:.2f} h), "
        f"minus `EOBT_1_flt` {rr['lag_eobt_s'] / 3600:.2f} h, minus the schedule {rr['lag_sched_s'] / 3600:.2f} h. "
        f"The rule fitted on the full year predicts {fy.share:.4f} x {rr['lag_lobt_s']:,.0f} + "
        f"(1 - {fy.share:.4f}) x {rr['rest_mean']:,.1f} = **{rr['rule']:,.1f} s** "
        f"({round(rr['rule']):,.0f} s after the submission's rounding). v6 predicts **{v6_pred:,.0f} s** for it, "
        "read from `data/submissions/funny-firefly_v6.parquet` through the template's identifier, which is not "
        f"printed. The difference is {rr['rule'] - v6_pred:+,.0f} s. What the departure recorded is hidden, "
        "and nothing here says which is right."
    )
    alt = st["alt_rest_lirf"]
    add(
        f"One reading the pre-registration leaves open is whether the rest mean counts the term's own rows. "
        f"Read literally it does, and that is what is implemented. Leaving them out would give a {rr['icao']} rest mean "
        f"of {alt:,.1f} s instead of {rr['rest_mean']:,.1f} s, and move the prediction by "
        f"{(1 - fy.share) * (alt - rr['rest_mean']):+,.1f} s."
    )
    add("")
    # 4
    add("## Per held-out month and on the two forward legs")
    add("")
    add(
        "On every leg the term and the rule are fitted on the leg's training rows and applied to its validation "
        "rows. *Stored* is the booster's prediction on the class's row, read from the held-out-month fit's "
        "residual file (the recorded taxi-out minus the stored residual). The held-out-month ladder "
        "reproduced the twelve held-out files to 0 s, and the forward files reproduce the ladder's stored December and July residuals to 0 s. The "
        "RMSEs are "
        "every row, with the term on its rows (what the submission scores) and with the booster on them, each "
        "without and with the rule. The second reading, over the rows of 4 h or less, is a diagnostic: it leaves "
        "out exactly the rows the rule is for when they are long, so it shows only what the rule does to an "
        "ordinary class row."
    )
    add("")
    rows = []
    for r in results:
        cls_txt = (
            "; ".join(
                f"{x['icao']}: recorded {x['taxi_s']:,.0f}, rule {x['rule']:,.0f}, stored {x['stored']:,.0f}"
                for x in r["class"]
            )
            or "none"
        )
        rows.append(
            [
                ("forward " if r["kind"] == "forward" else "held out ")
                + c.month_label(r["valid_month"]),
                f"{r['rows']:,}",
                f"{r['fit']['n']} / {r['fit']['long']} / {r['fit']['share']:.3f}",
                f"{r['class_rows']}",
                cls_txt,
            ]
        )
    L += c.table(
        ["leg", "rows", "fitted n / long / share", "class rows", "the class's rows (s)"],
        rows,
        "lrrrl",
    )
    add("")
    add(
        "Every row, with and without the rule (s), and the change in squared error (million s²). In brackets, the "
        "same over the rows whose recorded taxi-out is 4 h or less, a diagnostic, with the count it leaves out:"
    )
    add("")

    def line(label: str, r: dict, n: int, n4: int) -> list[str]:
        out = [label]
        for base in ("term", "booster"):
            a, b = f"sse_{base}", f"sse_{base}_rule"
            out += [
                f"{c.fmt(rm_(r[a], n))} [{c.fmt(rm_(r[a + '_d4'], n4))}]",
                f"{c.fmt(rm_(r[b], n))} [{c.fmt(rm_(r[b + '_d4'], n4))}]",
                f"{(r[b] - r[a]) / 1e6:+,.1f} [{(r[b + '_d4'] - r[a + '_d4']) / 1e6:+,.1f}; {n - n4:,} out]",
            ]
        return out

    rows = []
    for r in results:
        label = ("forward " if r["kind"] == "forward" else "held out ") + c.month_label(
            r["valid_month"]
        )
        rows.append(line(label, r, r["rows"], r["d4_rows"]))
    held = [r for r in results if r["kind"] == "held-out"]
    tot = {k: sum(r[k] for r in held) for k in held[0] if k.startswith("sse_")}
    N = sum(r["rows"] for r in held)
    N4 = sum(r["d4_rows"] for r in held)
    rows.append(line("**the year, twelve held-out months pooled**", tot, N, N4))
    L += c.table(
        [
            "leg",
            "with the term",
            "with the term and the rule",
            "change in squared error",
            "booster only",
            "booster only and the rule",
            "change in squared error",
        ],
        rows,
        "lrrrrrr",
    )
    add("")
    moved = [r for r in held if r["class_rows"]]
    add(
        f"The rule touches {sum(r['class_rows'] for r in held)} validation row(s) over the twelve held-out months "
        f"({', '.join(c.month_label(r['valid_month']) for r in moved) or 'none'}) and "
        f"{sum(r['class_rows'] for r in results if r['kind'] == 'forward')} on the two forward legs. Over the year, every row "
        f"with the term moves from {c.fmt(rm_(tot['sse_term'], N))} to {c.fmt(rm_(tot['sse_term_rule'], N))} s "
        f"({c.fmt(rm_(tot['sse_term_d4'], N4))} to {c.fmt(rm_(tot['sse_term_rule_d4'], N4))} s over the {N4:,} rows of "
        f"4 h or less, a diagnostic, {N - N4:,} left out)."
    )
    add("")
    # 5
    add("## What this evaluation can show")
    add("")
    add(
        "In the pre-registration's own words: \"In the training year the class holds three departures, so this "
        "evaluation shows what the rule does to those rows and nothing more: it is a statement of the arithmetic "
        'and not independent evidence that the rule is right." Each held-out month that holds one of the three '
        "fits the rule on the other two, so the share it applies was computed from the very rows the rule was "
        "written after seeing. Nothing here settles whether the rule is right for the 2026 departure, which no "
        "data available to the entry can settle."
    )
    add("")
    L += c.inputs_footer(inputs)
    return "\n".join(L) + "\n"


def report(out: Path) -> int:
    before = c.fingerprint()
    ok, st = stops()
    if not ok:
        print("a pre-registered stop fired; nothing is evaluated", file=sys.stderr)
        return 1
    dep, y = st["dep"], st["y"]
    st["full_rule"] = PlanShift().fit(dep, st["covered"], y)
    lirf = (
        (dep["ADEP_mvt"].astype(str) == "LIRF")
        & dep["LOBT_flt"].notna()
        & (y <= THRESHOLD_S)
        & ~st["covered"]
    )
    st["alt_rest_lirf"] = float(y[lirf].mean())
    # the ranking departure
    rank, rc = st["rank"], st["rank_covered"]
    cls = planshift_class(rank, rc)
    pred = st["full_rule"].predict(rank, rc)
    i = int(np.flatnonzero(cls.to_numpy())[0])
    r = rank.iloc[i]
    t = c.utc(pd.Series([r["MVT_TIME_UTC_mvt"]]))[0]
    rank_rule = {
        "icao": str(r["ADEP_mvt"]),
        "month": t.strftime("%Y-%m"),
        "lag_lobt_s": (t - c.utc(pd.Series([r["LOBT_flt"]]))[0]).total_seconds(),
        "lag_eobt_s": (t - c.utc(pd.Series([r["EOBT_1_flt"]]))[0]).total_seconds(),
        "lag_sched_s": (
            t - c.utc(pd.Series([r["SCHED_TIME_UTC_mvt"]]))[0]
        ).total_seconds(),
        "rule": float(pred.iloc[i]),
        "rest_mean": st["full_rule"].rest_mean[str(r["ADEP_mvt"])],
    }
    template = pd.read_parquet(c.TEMPLATE)
    v6 = pd.read_parquet(c.V6)
    if not np.array_equal(
        v6["MVT_ID_mvt"].to_numpy(dtype=float),
        template["MVT_ID_mvt"].to_numpy(dtype=float),
    ):
        raise SystemExit("v6 is not in the template's order")
    the_id = float(r["MVT_ID_mvt"])
    hit = v6.loc[v6["MVT_ID_mvt"].to_numpy(dtype=float) == the_id, "TAXITIME_SEC_mvt"]
    if len(hit) != 1:
        raise SystemExit(f"{len(hit)} v6 rows for the class's departure")
    v6_pred = float(hit.iloc[0])
    month = rm.month_of(dep["MVT_TIME_UTC_mvt"])
    results = [evaluate_leg(dep, y, month, leg) for leg in legs()]
    files = [c.MONTHLY / f"{leg.stem}.parquet" for leg in legs()]
    inputs = (
        rb._inputs()
        + files
        + [c.V6, c.TEMPLATE, c.DATA / "ranking.parquet", prereg_copy()]
    )
    text = build_report(st, results, v6_pred, rank_rule, inputs)
    stands = c.stands_of(dep, rank)
    ids = c.movement_ids(dep["MVT_ID_mvt"], rank["MVT_ID_mvt"])
    c.write(out, text, stands, ids, before)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stops", action="store_true")
    mode.add_argument("--report", action="store_true")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    if args.stops:
        ok, _ = stops()
        return 0 if ok else 1
    return report(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
