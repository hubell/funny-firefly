#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The three candidates of 28 September, under the rule of 3 October.

Three variations of the submission rung, each measured once before and left
on the table, are fitted on the legs the ladder uses and read against the
rung's own fit under a rule written before any of them was fitted:
``reports/preregistered-2026-10-03.md``, which keeps the candidates, the legs,
the reference and the four conditions of ``reports/preregistered-2026-09-28.md``
and adds a fifth. The candidates:

1. **Saturation**: ``taxiout.saturation.SATURATION_FEATURES`` appended to the
   features of ``+ plan differences``.
2. **The mild recency schedule**: row weights halving every twelve months over
   the fitting rows, by ``preregistered.recency_weights``, the function the
   schedule was first fitted with.
3. **More trees**: ``n_estimators`` 8,000, every other parameter of
   ``BOOSTER_PARAMS`` unchanged.

Each is passed to the existing fitting code as an argument: the feature tuple
``Booster`` is given, the ``sample_weight`` its ``fit`` takes, or the
``n_estimators`` of its parameters. **Nothing in ``taxiout.models`` changes
here, and this script writes no submission.**

**The rows every figure is over** are the validation rows that neither the
record-quality term nor the plan-shift rule covers, both fitted on the leg's
training rows as ``taxiout.tails`` fits them, because those are the rows the
booster predicts in what ships. Beside every figure, the same reading over
those of them whose recorded taxi-out is four hours or less, a diagnostic that
decides nothing.

**The reproduction check runs first and a failure stops the run.** The
reference, the submission rung's own fit at the ladder's seed, must reproduce
the ladder's stored December and July residuals with a largest absolute
difference of zero, joined on the movement identifier, which is never printed,
and October's figure as ``reports/baseline.md`` prints it. Before the held-out
months are read, one of them is refitted for the reference and compared with
the stored file, every column, to the bit.

**Every verdict is computed** from the rule's constants, which ``the_rule``
holds, and the floors are read from the reports that print them and compared
with those constants before anything is fitted.

Like ``preregistered.py``, this writes under one gitignored directory of its
own, ``data/processed/candidates/``, and to the one report it generates, and it
fingerprints the stored residuals and every report before a run and verifies
them after it.

Usage
-----
    python scripts/candidates.py --seeds-report PATH --dry-run
    python scripts/candidates.py --seeds-report PATH --fit legs
    python scripts/candidates.py --seeds-report PATH --fit months
    python scripts/candidates.py --seeds-report PATH --fit combination
    python scripts/candidates.py --seeds-report PATH --fit seed1
    python scripts/candidates.py --seeds-report PATH --report

``--seeds-report`` is the five-seed report of the held-out months, which
prints the year's floor and is not in this repository until it is landed in
``reports/``. Each ``--fit`` stage fits only what the rule asks for given the
records already on disk, and does nothing when the rule does not reach it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

# The repository root, resolved without a subscript so that this module holds
# no numeric constant, as ``check_paper.py`` resolves it: the register's check 1
# reads tracked files only, so a constant here would be a stale entry until the
# commit that adds this file. The rule's numbers are in ``the_rule`` instead.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import check_licence_hygiene as hygiene  # noqa: E402
import preregistered as pr  # noqa: E402
import run_baselines as rb  # noqa: E402
from taxiout.encoding import encode_split, with_encoding  # noqa: E402
from taxiout.features import TARGET, airport_fallback, build_features  # noqa: E402
from taxiout.models import (  # noqa: E402
    BOOSTER_PARAMS,
    LADDER_BY_KEY,
    SUBMISSION_KEY,
    Booster,
    fit_rows as fitting_rows,
)
from taxiout.provenance import FROZEN_DOCUMENTS, digest, report_inputs_footer  # noqa: E402
from taxiout.saturation import SATURATION_FEATURES  # noqa: E402
from taxiout.tails import (  # noqa: E402
    PlanShiftModel,
    TailModel,
    combine,
    plan_shift_threshold_s,
)
from taxiout.validation import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    DECEMBER_TRAIN,
    DECEMBER_VALID,
    JULY_AIRPORTS,
    JULY_TRAIN,
    JULY_VALID,
    cluster_bootstrap,
    cluster_sums,
    month_of,
)

WORK = REPO / "data" / "processed" / "candidates"
REPORT = REPO / "reports" / "candidates-2026-10-03.md"

# What this script may never write, checked rather than promised: every report,
# the ladder's stored residuals, every other directory of stored residuals and
# records, and the submissions. Its own report is exempted by name in the one
# mode that writes it.
FORBIDDEN = (
    REPO / "reports",
    REPO / "data" / "processed" / "residuals",
    REPO / "data" / "processed" / "monthly",
    REPO / "data" / "processed" / "monthly_ladder",
    REPO / "data" / "processed" / "monthly_seeds",
    REPO / "data" / "processed" / "monthly_pairs",
    REPO / "data" / "processed" / "preregistered",
    REPO / "data" / "submissions",
)

# Read, never written. Repository-relative strings rather than path
# expressions, so that no module-level expression here names a report this
# script would be taken to write.
BASELINE = "reports/baseline.md"
RULE_TEXT = "reports/preregistered-2026-10-03.md"
CANDIDATES_TEXT = "reports/preregistered-2026-09-28.md"
LADDER_RESIDUALS = "data/processed/residuals"
MONTHLY = "data/processed/monthly"
MONTHLY_SEEDS = "data/processed/monthly_seeds"

# The digests of the working notes the two published copies were made from,
# which each copy names as the proof of its order.
NOTE_DIGESTS = {
    CANDIDATES_TEXT: "eee5b947e4c449eb7df5765ca62d25a3ca954f201ecefa23311eec8918813e60",
    RULE_TEXT: "91e3306ef2b5e0efd6ab8d0a5c0ed9e2fc29ed0b0015bb4af8e96c6a2a601662",
}

ATTRIBUTION = (
    "Distance matrices derived from OpenStreetMap: data (c) OpenStreetMap"
    " contributors, ODbL 1.0. Movement and flight data supplied by EUROCONTROL"
    " PRC and the OpenSky Network under the terms of the PRC Data Challenge 2026."
)

# Patterns, not names, for the guard every generated text passes: a numbered
# work block with or without a letter after its number, and the words for where
# the work was discussed. Bracketed so this file does not match itself.
WORK_BLOCK_PATTERN = r"\bBlock\s+\d+[A-Za-z]*\b"
DISCUSSION_PATTERN = r"\bcha[t]\b|\bsessio[n]\b"


# --------------------------------------------------------------------------
# the rule, as constants


@dataclass(frozen=True)
class Rule:
    """The numbers the rule fixes, each with the words it is fixed in."""

    floor_december: float
    floor_july: float
    floor_july_three: float
    floor_october: float
    floor_year: float
    contrast_decimals: int
    more_trees: int
    half_life_months: int
    plan_shift_rows: dict
    plan_shift_july_recorded_s: float
    deadline_utc: str


def the_rule() -> Rule:
    """The rule of 3 October and the candidates of 28 September, as numbers.

    Every value is copied from ``reports/preregistered-2026-10-03.md`` or the
    note of 28 September it keeps, and ``tests/test_candidates.py`` checks each
    against the published copies' words:

    - the four floors, "December, all ten: better by more than 0.23 s",
      "July, all ten: better by more than 0.54 s", "July, the three: not worse
      by more than 1.02 s", "October: not worse by more than 0.21 s", and the
      year's, "the year's floor, 0.16 s";
    - the precision the contrast is read at, "at the precision
      `reports/baseline.md` prints": that report prints every contrast it reads
      against a floor, and every floor, to two decimals;
    - the candidates, "`n_estimators` 8,000" and "row weights halving every
      12 months";
    - the plan-shifted rows, "On the July leg the rule covers one validation
      row, recorded 87,002 s" and "December and October hold no such row";
    - the deadline, "If the block has no verdict by 7 October 2026, 12:00 UTC,
      nothing is adopted".

    Held in a function and not at module level for the reason ``REPO`` is
    resolved without a subscript.
    """
    return Rule(
        floor_december=0.23,
        floor_july=0.54,
        floor_july_three=1.02,
        floor_october=0.21,
        floor_year=0.16,
        contrast_decimals=2,
        more_trees=8000,
        half_life_months=12,
        plan_shift_rows={"december": 0, "july": 1, "october": 0},
        plan_shift_july_recorded_s=87002.0,
        deadline_utc="2026-10-07T12:00:00+00:00",
    )


@dataclass(frozen=True)
class Condition:
    """One of the five, read on one set of rows of one fit."""

    key: str
    label: str
    kind: str  # "better" or "not worse"
    floor: float
    split: str
    rows: str  # "all" or "three"


def conditions(rule: Rule) -> tuple[Condition, ...]:
    """The five, in the order the rule lists them."""
    return (
        Condition(
            "december",
            "December, all ten",
            "better",
            rule.floor_december,
            "december",
            "all",
        ),
        Condition("july", "July, all ten", "better", rule.floor_july, "july", "all"),
        Condition(
            "july_three",
            "July, the three",
            "not worse",
            rule.floor_july_three,
            "july",
            "three",
        ),
        Condition(
            "october",
            "October, all ten",
            "not worse",
            rule.floor_october,
            "october",
            "all",
        ),
        Condition(
            "year",
            "the twelve held-out months",
            "not worse",
            rule.floor_year,
            "year",
            "all",
        ),
    )


def hundredths(x: float, decimals: int) -> int:
    """``x`` as printed at ``decimals`` places, in units of the last place.

    The figure is formatted first and the integer taken from the printed
    string, so the reading is the printed figure's and not the binary one's.
    """
    return int(Decimal(f"{x:.{decimals}f}").scaleb(decimals))


def holds(kind: str, contrast: float, floor: float, decimals: int) -> bool:
    """Arm minus reference, read at the printed precision against the floor.

    Higher RMSE is worse. "Better by more than the floor": the printed gain
    exceeds the printed floor. "Not worse by more than the floor": the printed
    loss does not exceed it, so a gain of any size holds.
    """
    c, f = hundredths(contrast, decimals), hundredths(floor, decimals)
    if kind == "better":
        return -c > f
    if kind == "not worse":
        return c <= f
    raise ValueError(f"not a kind of condition: {kind!r}")


def holds_unrounded(kind: str, contrast: float, floor: float) -> bool:
    """The same reading on the unrounded contrast, for the report's comparison."""
    if kind == "better":
        return -contrast > floor
    return contrast <= floor


def holds_at_figure_precision(
    kind: str, arm: float, reference: float, floor: float
) -> bool:
    """The same reading with each figure first rounded to one decimal."""
    c = Decimal(f"{arm:.1f}") - Decimal(f"{reference:.1f}")
    f = Decimal(f"{floor:.2f}")
    if kind == "better":
        return -c > f
    return c <= f


def holds_at_contrast_precision(kind: str, contrast: float, floor: float) -> bool:
    """The same reading with the contrast and the floor both rounded to one decimal."""
    c = Decimal(f"{contrast:.1f}")
    f = Decimal(f"{floor:.1f}")
    if kind == "better":
        return -c > f
    return c <= f


# The other readings the report sets beside the rule's, so that a reader can see
# whether any verdict turns on the precision it was read at. They decide nothing.
OTHER_READINGS = {
    "unrounded": "on the unrounded contrast",
    "figures": "with each figure first rounded to one decimal",
    "contrast": "with the contrast and the floor both rounded to one decimal",
}


def holds_read(reading: str, kind: str, r: dict, floor: float, decimals: int) -> bool:
    """One condition under one reading; ``rule`` is the rule's own."""
    if reading == "rule":
        return holds(kind, r["contrast"], floor, decimals)
    if reading == "unrounded":
        return holds_unrounded(kind, r["contrast"], floor)
    if reading == "figures":
        return holds_at_figure_precision(kind, r["arm"], r["reference"], floor)
    if reading == "contrast":
        return holds_at_contrast_precision(kind, r["contrast"], floor)
    raise ValueError(f"not a reading: {reading!r}")


def same_sign(seed0: float, seed1: float, decimals: int) -> bool:
    """Both contrasts negative or both positive, at the printed precision.

    A contrast that prints as zero has no sign, so it never matches one that
    does: the check asks whether the gain was there again, and a zero is not.
    """
    a, b = hundredths(seed0, decimals), hundredths(seed1, decimals)
    return (a < 0 and b < 0) or (a > 0 and b > 0)


# --------------------------------------------------------------------------
# the candidates


@dataclass(frozen=True)
class Arm:
    """A variation of the submission rung, passed to the fitting code."""

    key: str
    saturation: bool = False
    half_life_months: int | None = None
    n_estimators: int | None = None


LABELS = {
    "reference": "the reference, `+ plan differences` as it ships",
    "saturation": "saturation on top of `+ plan differences`",
    "recency": "the mild recency schedule",
    "trees": "more trees",
    "combination": "the combination",
}


def reference_arm() -> Arm:
    return Arm("reference")


def candidate_arms(rule: Rule) -> tuple[Arm, ...]:
    """The three, in the order the pre-registration of 28 September lists them."""
    return (
        Arm("saturation", saturation=True),
        Arm("recency", half_life_months=rule.half_life_months),
        Arm("trees", n_estimators=rule.more_trees),
    )


def combined(arms: list[Arm]) -> Arm:
    """The clearing candidates combined into one arm."""
    if len(arms) < 2:
        raise ValueError("a combination needs at least two arms")
    half_lives = {a.half_life_months for a in arms if a.half_life_months is not None}
    trees = {a.n_estimators for a in arms if a.n_estimators is not None}
    if len(half_lives) > 1 or len(trees) > 1:
        raise ValueError("two arms set the same variation differently")
    return Arm(
        "combination",
        saturation=any(a.saturation for a in arms),
        half_life_months=next(iter(half_lives), None),
        n_estimators=next(iter(trees), None),
    )


def arm_from(record: dict) -> Arm:
    return Arm(**record["arm"])


def columns_of(arm: Arm) -> tuple[str, ...]:
    """The rung's features, with the saturation block appended if the arm has it."""
    base = tuple(LADDER_BY_KEY[SUBMISSION_KEY].features)
    if not arm.saturation:
        return base
    overlap = sorted(set(SATURATION_FEATURES) & set(base))
    if overlap:
        raise RuntimeError(f"the rung already reads {overlap}")
    return base + tuple(SATURATION_FEATURES)


def params_of(arm: Arm, seed: int) -> dict:
    """``BOOSTER_PARAMS`` with the seed offset, and the tree count if the arm sets it."""
    params = dict(BOOSTER_PARAMS)
    params["random_state"] = int(BOOSTER_PARAMS["random_state"]) + int(seed)
    if arm.n_estimators is not None:
        params["n_estimators"] = int(arm.n_estimators)
    return params


def weights_of(
    arm: Arm, month: pd.Series, rows: pd.Series, last: str
) -> np.ndarray | None:
    """The recency weights over the fitting rows, or ``None`` for an unweighted fit."""
    if arm.half_life_months is None:
        return None
    return pr.recency_weights(month, rows, last, int(arm.half_life_months))


# --------------------------------------------------------------------------
# the splits


@dataclass
class Split:
    """One fit's training months and validation month."""

    name: str
    train_months: tuple[str, ...]
    valid_month: str
    description: str

    @property
    def last(self) -> str:
        """The last training month, where a row's age is measured from."""
        return max(self.train_months)


# The twelve months of 2025, from the module that owns the December split.
MONTHS: tuple[str, ...] = tuple(DECEMBER_TRAIN) + tuple(DECEMBER_VALID)


def leg_plan() -> list[Split]:
    """The three legs of the pre-registration of 28 September."""
    legs = [
        Split(
            "december",
            tuple(DECEMBER_TRAIN),
            DECEMBER_VALID[0],
            "train January to November 2025, validate December, all ten",
        ),
        Split(
            "july",
            tuple(JULY_TRAIN),
            JULY_VALID[0],
            "train January to June 2025, validate July, all ten and the three",
        ),
        Split(
            "october",
            tuple(pr.OCTOBER_TRAIN),
            pr.OCTOBER_VALID[0],
            "train January to September 2025, validate October, all ten",
        ),
    ]
    for leg in legs:
        if leg.last != pr.LAST_TRAIN_MONTH[leg.name]:
            raise RuntimeError(
                f"{leg.name}: the last training month is not the schedule's"
            )
    return legs


def month_plan() -> list[Split]:
    """The held-out-month splits: each month of 2025 held out, the other eleven training."""
    plan = [
        Split(
            f"heldout-{m}",
            tuple(o for o in MONTHS if o != m),
            m,
            f"train the other eleven months of 2025, validate {m}",
        )
        for m in MONTHS
    ]
    for split in plan:
        if (
            split.valid_month in split.train_months
            or len(split.train_months) != len(MONTHS) - 1
        ):
            raise RuntimeError(f"{split.name}: not one month against eleven")
    return plan


def check_month() -> str:
    """The held-out month refitted for the reference before the others are read."""
    return min(MONTHS)


# --------------------------------------------------------------------------
# what this script may not write


def fingerprint(exempt: Path | None = None) -> dict[str, tuple[int, int, str]]:
    """Size, modification time and sha256 of every file this script must not write."""
    out: dict[str, tuple[int, int, str]] = {}
    for directory in FORBIDDEN:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != exempt:
                stat = path.stat()
                out[str(path)] = (stat.st_size, stat.st_mtime_ns, digest(path))
    return out


def touched(before: dict, after: dict) -> list[str]:
    return sorted(
        {k for k in before if after.get(k) != before[k]} | (set(after) - set(before))
    )


def ignored(path: Path) -> bool:
    """Whether git ignores ``path``, asked of git and not assumed."""
    got = subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        cwd=REPO,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        capture_output=True,
        check=False,
    )
    return got.returncode == 0


def suspended_s() -> float:
    """Seconds the machine has slept since boot, which a wall clock would count."""
    return time.clock_gettime(time.CLOCK_BOOTTIME) - time.clock_gettime(
        time.CLOCK_MONOTONIC
    )


def peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576


# --------------------------------------------------------------------------
# what is read before anything is fitted


class Checks:
    """Every check, with its expectation, printed as it is made."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, what: str, expected: str, got: str, ok: bool) -> bool:
        self.rows.append({"what": what, "expected": expected, "got": got, "ok": bool(ok)})
        print(
            f"  [{'ok ' if ok else 'STOP'}] {what}: expected {expected}, got {got}",
            flush=True,
        )
        return bool(ok)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.rows if not r["ok"]]


def _section(text: str, heading: str) -> str:
    start = text.find(f"\n{heading}\n")
    if start < 0:
        raise ValueError(f"no {heading!r} heading")
    rest = text[start + len(heading) + 2 :]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_cell(section: str, header_starts: str, column: str, row: str) -> str:
    """One cell of the first table whose header starts as given."""
    lines = section.split("\n")
    for i, line in enumerate(lines):
        if line.startswith(header_starts):
            header = _cells(line)
            if column not in header:
                raise ValueError(f"no {column!r} column")
            for body in lines[i + 2 :]:
                if not body.startswith("|"):
                    break
                cells = _cells(body)
                if cells and cells[0] == row:
                    return cells[header.index(column)]
            raise ValueError(f"no {row!r} row")
    raise ValueError(f"no table headed {header_starts!r}")


def read_floors(baseline_text: str) -> dict[str, str]:
    """The four floors, as ``reports/baseline.md`` prints them in *The noise floor*."""
    section = _section(baseline_text, "## The noise floor")
    return {
        key: table_cell(section, "| leg |", "floor (s)", label)
        for key, label in (
            ("december", "December, all ten"),
            ("july", "July, all ten"),
            ("july_three", "July, the three"),
            ("october", "October, all ten"),
        )
    }


def read_october(baseline_text: str) -> str:
    """October's figure for the ladder's seed, as *Seed averaging* prints it."""
    section = _section(baseline_text, "## Seed averaging")
    return table_cell(section, "| column |", "the ladder's seed (s)", "October, all ten")


def read_year_floor(seeds_text: str) -> str:
    """The year's floor, the ``floor`` cell of the year row of the per-month table."""
    section = _section(seeds_text, "## Per month and over the year")
    return table_cell(section, "| month | rows | seed 0 |", "floor", "year")


def frozen_copies(checks: Checks) -> bool:
    """Both published copies at their frozen digests, each naming its note's."""
    held = {f.path: f.sha256 for f in FROZEN_DOCUMENTS}
    ok = True
    for path, note in NOTE_DIGESTS.items():
        got = digest(REPO / path)
        ok &= checks.add(
            f"`{path}` at the digest `taxiout.provenance.FROZEN_DOCUMENTS` holds",
            str(held.get(path)),
            got,
            held.get(path) is not None and got == held.get(path),
        )
        text = (REPO / path).read_text(encoding="utf-8") if (REPO / path).exists() else ""
        named = set(re.findall(r"`([0-9a-f]{64})`", text))
        ok &= checks.add(
            f"`{path}` names its working note's digest as the proof of its order",
            note,
            ", ".join(sorted(named)) or "none",
            named == {note},
        )
    return ok


def floor_checks(checks: Checks, rule: Rule, seeds_report: Path) -> dict:
    """The five floors read where they are printed, against the rule's constants."""
    baseline_text = (REPO / BASELINE).read_text(encoding="utf-8")
    floors = read_floors(baseline_text)
    want = {
        "december": rule.floor_december,
        "july": rule.floor_july,
        "july_three": rule.floor_july_three,
        "october": rule.floor_october,
    }
    labels = {c.key: c.label for c in conditions(rule)}
    d = rule.contrast_decimals
    for key, value in want.items():
        checks.add(
            f"the floor of {labels[key]} in `{BASELINE}`, *The noise floor*",
            f"{value:.{d}f}",
            floors[key],
            floors[key] == f"{value:.{d}f}",
        )
    seeds_text = seeds_report.read_text(encoding="utf-8")
    year = read_year_floor(seeds_text)
    checks.add(
        "the year's floor in the five-seed report of the held-out months, the year row",
        f"{rule.floor_year:.{d}f}",
        year,
        year == f"{rule.floor_year:.{d}f}",
    )
    october = read_october(baseline_text)
    return {
        "baseline": floors,
        "year": year,
        "october_printed": october,
        "seeds_report_sha256": digest(seeds_report),
    }


# --------------------------------------------------------------------------
# one fit


def coverage(dep: pd.DataFrame, train: pd.Series, valid: pd.Series) -> dict:
    """What the term and the plan-shift rule cover, both fitted on ``train``.

    ``taxiout.tails`` fitted as ``make_submission.py`` fits it on the full
    year: the record-quality term on the training rows, and the plan-shift
    rule defined against that term and fitted on the same rows. ``flagged`` is
    the term's rows and ``shift`` the rule's, which ``in_class`` already keeps
    apart from the term's. The rule takes no row out of the fit; it only says
    which rows it predicts.
    """
    term = TailModel().fit(dep.loc[train])
    tail = term.predict(dep)
    shift_model = PlanShiftModel(term).fit(dep.loc[train])
    shift = shift_model.in_class(dep).astype(bool)
    # The rule must be able to predict every row it claims; it refuses one it
    # has no rest mean for, and a refusal here is a stop.
    shift_model.predict(dep.loc[valid & shift])
    return {
        "tail": tail,
        "flagged": tail.notna(),
        "shift": shift,
        "plan_shift": {
            "n": shift_model.n,
            "long": shift_model.long,
            "share": shift_model.share,
        },
    }


class Frame:
    """The feature frame and the target, built as ``preregistered.Frame`` builds it."""

    def __init__(self) -> None:
        t0 = time.monotonic()
        matrix = rb.load_matrix()
        self.dep = rb.load_departures(None)
        self.y = pd.to_numeric(self.dep[TARGET], errors="coerce")
        movements = rb.load_all_movements()
        self.features = build_features(
            self.dep,
            matrix,
            fallback=airport_fallback(matrix),
            movements=movements,
            edges=rb.load_edge_sets(),
            weather=rb.load_weather(),
            airports=rb.load_airport_table(),
        )
        del movements
        self.build_seconds = time.monotonic() - t0
        self.month = month_of(self.dep["MVT_TIME_UTC_mvt"])
        print(
            f"frame: {len(self.features):,} rows in {self.build_seconds:.0f} s, "
            f"peak {peak_gib():.1f} GiB",
            flush=True,
        )

    def stands(self) -> list[str]:
        values = self.dep["STAND_mvt"].dropna().astype(str).str.strip()
        return sorted(set(values) - {""})

    def prepare(self, split: Split) -> dict:
        """One split's term, plan-shift rule, fitting rows and encoded frame.

        What ``run_baselines.run`` does inside its split loop, and what
        ``make_submission.py`` does on the full year: the record-quality term
        and the plan-shift rule fitted on the split's training rows, the
        fitting rows over train, unflagged and the four-hour exclusion, and the
        shrunk encoding recomputed from those rows and overlaid. The rule takes
        no row out of the fit; it only says which validation rows are its.
        """
        t0 = time.monotonic()
        train = self.month.isin(split.train_months)
        valid = self.month == split.valid_month
        if bool((train & valid).any()) or not bool(valid.any()):
            raise RuntimeError(f"{split.name}: the split is not a split")
        covered = coverage(self.dep, train, valid)
        flagged, shift = covered["flagged"], covered["shift"]
        rows = fitting_rows(train, flagged, self.y)
        encoding = encode_split(self.features, self.y, rows)
        got = {
            "split": split,
            "train": train,
            "valid": valid,
            "tail": covered["tail"],
            "flagged": flagged,
            "shift": shift,
            "rows": rows,
            "frame": with_encoding(self.features, encoding),
            "plan_shift": covered["plan_shift"],
            "seconds": time.monotonic() - t0,
        }
        print(
            f"  {split.name:>16s}: {int(train.sum()):,} train, {int(valid.sum()):,} validate, "
            f"{int(rows.sum()):,} fit rows, {int((flagged & valid).sum()):,} term rows, "
            f"{int((shift & valid).sum()):,} plan-shift rows ({got['seconds']:.0f} s)",
            flush=True,
        )
        return got

    def counts(self, prep: dict) -> dict:
        valid = prep["valid"]
        booster = valid & ~prep["flagged"] & ~prep["shift"]
        short = booster & (self.y <= plan_shift_threshold_s())
        return {
            "train": int(prep["train"].sum()),
            "valid": int(valid.sum()),
            "fit_rows": int(prep["rows"].sum()),
            "term": int((prep["flagged"] & valid).sum()),
            "plan_shift": int((prep["shift"] & valid).sum()),
            "booster": int(booster.sum()),
            "four_hours_or_less": int(short.sum()),
            "plan_shift_fit": prep["plan_shift"],
        }


def record_name(split: str, arm: str, seed: int) -> str:
    """Underscores throughout, so that no part of a name reads as a stand identifier."""
    return f"{split}__{arm}__seed{int(seed)}"


def fit_one(frame: Frame, prep: dict, arm: Arm, seed: int) -> tuple[dict, pd.DataFrame]:
    """One fit of the submission rung under one arm, scored on the split's month."""
    split: Split = prep["split"]
    columns = columns_of(arm)
    missing = [c for c in columns if c not in prep["frame"].columns]
    if missing:
        raise RuntimeError(f"the frame lacks {missing}")
    params = params_of(arm, seed)
    rows = prep["rows"]
    weight = weights_of(arm, frame.month, rows, split.last)
    t0 = time.monotonic()
    model = Booster(columns, params).fit(
        prep["frame"].loc[rows], frame.y[rows], sample_weight=weight
    )
    fit_s = time.monotonic() - t0
    mask = prep["valid"]
    t1 = time.monotonic()
    pred = np.asarray(model.predict(prep["frame"].loc[mask]), dtype=float)
    predict_s = time.monotonic() - t1
    raw = frame.dep.loc[mask]
    y = np.asarray(frame.y[mask], dtype=float)
    kept = pd.DataFrame(
        {
            "mvt_id": raw["MVT_ID_mvt"].to_numpy(),
            "icao": raw["ADEP_mvt"].astype(str).to_numpy(),
            "runway": raw["RUNWAY_mvt"].astype(str).to_numpy(),
            "takeoff": pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).to_numpy(),
            "term": np.asarray(prep["flagged"][mask], dtype=bool),
            "shift": np.asarray(prep["shift"][mask], dtype=bool),
            "y": y,
            "pred": pred,
            "with_term": combine(pred, prep["tail"][mask]).to_numpy(dtype=float),
        }
    )
    record = {
        "name": record_name(split.name, arm.key, seed),
        "split": split.name,
        "train_months": list(split.train_months),
        "valid_month": split.valid_month,
        "arm": asdict(arm),
        "seed": int(seed),
        "random_state": int(params["random_state"]),
        "n_estimators": int(params["n_estimators"]),
        "columns": len(columns),
        "fit_rows": int(rows.sum()),
        "weight_sum": None if weight is None else float(np.sum(weight)),
        "weight_last": split.last if weight is not None else None,
        "valid_rows": int(mask.sum()),
        "prepare_s": float(prep["seconds"]),
        "fit_s": fit_s,
        "predict_s": predict_s,
        "peak_rss_gib": peak_gib(),
        "pred_sha256": hashlib.sha256(pred.tobytes()).hexdigest(),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    del model
    return record, kept


# --------------------------------------------------------------------------
# the records on disk


def record_path(name: str) -> Path:
    return WORK / f"{name}.json"


def kept_path(name: str) -> Path:
    return WORK / f"{name}.parquet"


def write_record(record: dict, kept: pd.DataFrame) -> None:
    """The predictions first and the figures last, so a record on disk is whole."""
    WORK.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(kept_path(record["name"]), index=False)
    record_path(record["name"]).write_text(
        json.dumps(record, indent=1, sort_keys=True), encoding="utf-8"
    )


def read_records() -> dict[str, dict]:
    if not WORK.exists():
        return {}
    out = {}
    for path in sorted(WORK.glob("*__*__seed*.json")):
        got = json.loads(path.read_text(encoding="utf-8"))
        out[got["name"]] = got
    return out


def read_kept(name: str) -> pd.DataFrame:
    return pd.read_parquet(kept_path(name))


def stage_path(stage: str) -> Path:
    return WORK / f"stage_{stage}.json"


def read_stage(stage: str) -> dict | None:
    path = stage_path(stage)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------
# the readings


def booster_rows(kept: pd.DataFrame, which: str = "all") -> np.ndarray:
    """The rows neither the term nor the plan-shift rule covers; ``three`` narrows to them."""
    rows = ~kept["term"].to_numpy(dtype=bool) & ~kept["shift"].to_numpy(dtype=bool)
    if which == "three":
        rows &= kept["icao"].isin(JULY_AIRPORTS).to_numpy()
    elif which != "all":
        raise ValueError(f"not a row set: {which!r}")
    return rows


def short_rows(kept: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Of ``rows``, those whose recorded taxi-out is four hours or less."""
    return rows & (kept["y"].to_numpy(dtype=float) <= plan_shift_threshold_s())


def rmse(residual) -> float:
    """Over the finite residuals, as ``taxiout.validation.rmse`` drops the rest."""
    r = np.asarray(residual, dtype=float)
    r = r[np.isfinite(r)]
    return float(np.sqrt(np.mean(r**2))) if len(r) else float("nan")


def aligned(arm: pd.DataFrame, base: pd.DataFrame) -> None:
    """Two predictions of the same rows, in the same order, or nothing is read."""
    same = (
        len(arm) == len(base)
        and np.array_equal(arm["mvt_id"].to_numpy(), base["mvt_id"].to_numpy())
        and np.array_equal(arm["term"].to_numpy(), base["term"].to_numpy())
        and np.array_equal(arm["shift"].to_numpy(), base["shift"].to_numpy())
        and np.array_equal(arm["y"].to_numpy(), base["y"].to_numpy(), equal_nan=True)
    )
    if not same:
        raise RuntimeError(
            "the two predictions are not of the same rows in the same order"
        )


def paired(arm: pd.DataFrame, base: pd.DataFrame, rows: np.ndarray) -> dict:
    """Arm minus base over ``rows``, with the day-clustered interval, one draw for both."""
    aligned(arm, base)
    a = (arm["y"] - arm["pred"]).to_numpy(dtype=float)[rows]
    b = (base["y"] - base["pred"]).to_numpy(dtype=float)[rows]
    days = pr.day_of(arm.loc[rows])
    sums = cluster_sums({"arm": a, "base": b}, days)
    got = cluster_bootstrap(
        [{"s": sums}], lambda r: r["s"]["arm"] - r["s"]["base"], draws=BOOTSTRAP_DRAWS
    )
    return {
        "rows": int(rows.sum()),
        "days": int(len(sums["arm"].n)),
        "reference": sums["base"].rmse,
        "arm": sums["arm"].rmse,
        "contrast": got.point,
        "low": got.low,
        "high": got.high,
        "draws": got.draws,
    }


def reading(arm: pd.DataFrame, base: pd.DataFrame, which: str) -> dict:
    """The contrast over the booster's rows, and over those of four hours or less."""
    rows = booster_rows(arm, which)
    return {
        "rows": paired(arm, base, rows),
        "short": paired(arm, base, short_rows(arm, rows)),
    }


def base_month(name: str, kept: pd.DataFrame) -> pd.DataFrame:
    """The base rung's stored seed-0 fit of one held-out month, as a kept frame.

    The stored file holds ``y - pred`` beside the movement identifier and the
    term's flag, in the order of the departures; the arm's own kept frame holds
    the target and both flags for the same rows. The two must agree row for
    row, identifier and term flag, or the month is not read.
    """
    month = name.removeprefix("heldout-")
    stored = pd.read_parquet(REPO / MONTHLY / f"heldout-{month}_{SUBMISSION_KEY}.parquet")
    same = (
        len(stored) == len(kept)
        and np.array_equal(stored["mvt_id"].to_numpy(), kept["mvt_id"].to_numpy())
        and np.array_equal(
            stored["flagged"].to_numpy(dtype=bool), kept["term"].to_numpy()
        )
    )
    if not same:
        raise RuntimeError(f"{name}: the stored file is not of the same rows and term")
    base = kept.copy()
    base["pred"] = kept["y"].to_numpy(dtype=float) - stored["residual"].to_numpy(
        dtype=float
    )
    base["with_term"] = kept["y"].to_numpy(dtype=float) - stored[
        "residual_with_term"
    ].to_numpy(dtype=float)
    return base


def measure(records: dict, rule: Rule) -> dict:
    """Every contrast the rule reads, from the kept predictions on disk.

    Keyed by arm, then by condition. A condition whose fits are not on disk is
    absent, which the verdict reads as "not reached".
    """
    out: dict[str, dict] = {}
    arms = {r["arm"]["key"] for r in records.values() if r["arm"]["key"] != "reference"}
    for seed in sorted({r["seed"] for r in records.values()}):
        for leg in leg_plan():
            ref = record_name(leg.name, "reference", seed)
            if ref not in records:
                continue
            base = read_kept(ref)
            for key in sorted(arms):
                name = record_name(leg.name, key, seed)
                if name not in records:
                    continue
                kept = read_kept(name)
                slot = out.setdefault(key, {}).setdefault(f"seed{seed}", {})
                slot[f"{leg.name}:all"] = reading(kept, base, "all")
                slot[f"{leg.name}:three"] = reading(kept, base, "three")
    # The fifth condition, at the ladder's seed only: each month's arm against
    # the base rung's stored fit, then the year pooled over every row.
    for key in sorted(arms):
        names = [record_name(s.name, key, 0) for s in month_plan()]
        if not all(n in records for n in names):
            continue
        arm_frames, base_frames, months = [], [], {}
        for split, name in zip(month_plan(), names):
            kept = read_kept(name)
            base = base_month(split.name, kept)
            months[split.valid_month] = reading(kept, base, "all")
            arm_frames.append(kept)
            base_frames.append(base)
        arm_year = pd.concat(arm_frames, ignore_index=True)
        base_year = pd.concat(base_frames, ignore_index=True)
        out.setdefault(key, {}).setdefault("seed0", {})["year:all"] = {
            **reading(arm_year, base_year, "all"),
            "months": months,
        }
    return out


# --------------------------------------------------------------------------
# the verdict


def condition_verdicts(m: dict, rule: Rule) -> dict:
    """Each condition at the ladder's seed: holds, fails, or not reached."""
    out = {}
    for c in conditions(rule):
        got = m.get("seed0", {}).get(f"{c.split}:{c.rows}")
        if got is None:
            out[c.key] = None
            continue
        out[c.key] = holds(
            c.kind, got["rows"]["contrast"], c.floor, rule.contrast_decimals
        )
    return out


def clears(verdicts: dict, keys: tuple[str, ...]) -> bool:
    return all(verdicts.get(k) is True for k in keys)


FOUR = ("december", "july", "july_three", "october")
FIVE = FOUR + ("year",)


def latest_written(records: dict) -> str | None:
    """When the last fit on disk was recorded, which is when a verdict from them exists."""
    times = [r["written_utc"] for r in records.values()]
    return max(times, key=datetime.fromisoformat) if times else None


def past_deadline(as_of: str | None, rule: Rule) -> bool:
    if as_of is None:
        return False
    return datetime.fromisoformat(as_of) > datetime.fromisoformat(rule.deadline_utc)


def decide(measured: dict, rule: Rule, as_of: str | None = None) -> dict:
    """What the rule asks for next, or what it adopts, from what has been measured.

    The sequence is the rule's: the four conditions on the three candidates;
    the fifth on those that hold the four; the combination if more than one
    holds all five, which ships only if it holds them too, else the single
    best-clearing candidate by the December contrast; then the seed-1 refit of
    whatever clears, whose contrasts on December and July must have the sign
    they had at seed 0, else nothing is adopted. ``as_of`` is when the verdict's
    last measurement was made: past the rule's deadline, nothing is adopted.
    """
    singles = [a.key for a in candidate_arms(rule)]
    verdicts = {k: condition_verdicts(measured.get(k, {}), rule) for k in singles}
    out: dict = {"verdicts": verdicts}
    if past_deadline(as_of, rule):
        out.update(next=None, adopted=None, outcome="past the deadline")
        return out
    missing = [k for k in singles if any(verdicts[k][c] is None for c in FOUR)]
    if missing:
        out.update(next="legs", adopted=None)
        return out
    four = [k for k in singles if clears(verdicts[k], FOUR)]
    out["clears_four"] = four
    if not four:
        out.update(next=None, adopted=None, outcome="none clears four")
        return out
    if any(verdicts[k]["year"] is None for k in four):
        out.update(next="months", adopted=None, fifth_for=four)
        return out
    five = [k for k in four if verdicts[k]["year"] is True]
    out["clears_five"] = five
    if not five:
        out.update(next=None, adopted=None, outcome="none clears five")
        return out
    winner = five[0]
    if len(five) > 1:
        combo = condition_verdicts(measured.get("combination", {}), rule)
        out["combination"] = combo
        if any(v is None for v in combo.values()):
            out.update(next="combination", adopted=None, combine=five)
            return out
        if clears(combo, FIVE):
            winner = "combination"
        else:
            winner = min(
                five,
                key=lambda k: measured[k]["seed0"]["december:all"]["rows"]["contrast"],
            )
    out["winner"] = winner
    seed1 = measured.get(winner, {}).get("seed1", {})
    if "december:all" not in seed1 or "july:all" not in seed1:
        out.update(next="seed1", adopted=None)
        return out
    signs = {
        leg: same_sign(
            measured[winner]["seed0"][f"{leg}:all"]["rows"]["contrast"],
            seed1[f"{leg}:all"]["rows"]["contrast"],
            rule.contrast_decimals,
        )
        for leg in ("december", "july")
    }
    out["signs"] = signs
    if all(signs.values()):
        out.update(next=None, adopted=winner, outcome="adopted")
    else:
        out.update(next=None, adopted=None, outcome="sign check fails")
    return out


def verdict_line(decision: dict) -> str:
    """The one line the report states, in the rule's words."""
    outcome = decision.get("outcome")
    if outcome == "adopted":
        winner = decision["adopted"]
        return (
            f"**The rule adopts {LABELS[winner]}.** It holds all five conditions on the"
            " paired contrast against the reference, and its paired contrasts at seed 1"
            " have the same sign as at seed 0 on December and July, so it is adopted:"
            " v8 is the five-seed average of this arm, and the one upload is v8."
        )
    if outcome == "none clears four":
        return (
            "**The rule adopts nothing.** No candidate holds the four conditions of"
            " 28 September on the paired contrast against the reference, so nothing"
            " changes: v7 ships."
        )
    if outcome == "none clears five":
        held = ", ".join(LABELS[k] for k in decision.get("clears_four", []))
        return (
            f"**The rule adopts nothing.** Of the candidates, {held} held the four"
            " conditions of 28 September, and none held the fifth, the twelve held-out"
            " months, so nothing changes: v7 ships."
        )
    if outcome == "sign check fails":
        winner = decision["winner"]
        return (
            f"**The rule adopts nothing.** {LABELS[winner].capitalize()} cleared, and its"
            " paired contrasts at seed 1 do not have the same sign as at seed 0 on both"
            " December and July, so nothing is adopted: v7 ships."
        )
    if outcome == "past the deadline":
        when = datetime.fromisoformat(the_rule().deadline_utc)
        return (
            "**The rule adopts nothing.** The rule had no verdict by"
            f" {when.day} {when:%B %Y}, {when:%H:%M} UTC, so nothing is adopted, and the"
            " paper says why: v7 ships."
        )
    return f"**No verdict yet**: the rule asks for the {decision.get('next')} fits next."


def stop_line(failed: list[dict]) -> str:
    """The line the report states instead of a verdict when a check disagreed."""
    return (
        "**No verdict: a check disagreed with its expectation, so nothing is read and"
        " nothing is adopted.** The checks that stopped it: "
        + "; ".join(r["what"] for r in failed)
        + "."
    )


# --------------------------------------------------------------------------
# the checks that are re-made from the files


def reproduces(kept: pd.DataFrame, stored: Path) -> tuple[str, bool]:
    """``preregistered.reproduces`` on a kept frame: every row, joined, to zero."""
    frame = kept[["mvt_id", "y", "pred"]]
    got, ok, _ = pr.reproduces(frame, stored)
    return got, ok


def reproduction_checks(checks: Checks, records: dict, rule: Rule, october: str) -> bool:
    """The reference at the ladder's seed against what the ladder stored and printed."""
    ok = True
    for leg in ("december", "july"):
        name = record_name(leg, "reference", 0)
        if name not in records:
            continue
        got, passed = reproduces(
            read_kept(name), REPO / LADDER_RESIDUALS / f"{leg}_{SUBMISSION_KEY}.parquet"
        )
        ok &= checks.add(
            f"the {leg.capitalize()} reference against the ladder's"
            f" `{leg}_{SUBMISSION_KEY}.parquet`",
            "a largest absolute difference of 0 on every row",
            got,
            passed,
        )
    name = record_name("october", "reference", 0)
    if name in records:
        kept = read_kept(name)
        figure = rmse((kept["y"] - kept["pred"]).to_numpy()[~kept["term"].to_numpy()])
        d = rule.contrast_decimals
        ok &= checks.add(
            f"the October reference, excluding the term's rows, against `{BASELINE}`",
            f"{october} s",
            f"{figure:.{d}f} s",
            f"{figure:.{d}f}" == october,
        )
    for leg in ("december", "july", "october"):
        name = record_name(leg, "reference", 0)
        if name not in records:
            continue
        kept = read_kept(name)
        shifted = kept.loc[kept["shift"].to_numpy(dtype=bool)]
        want = int(rule.plan_shift_rows[leg])
        ok &= checks.add(
            f"{leg.capitalize()}: validation rows the plan-shift rule covers",
            f"{want}",
            f"{len(shifted)}",
            len(shifted) == want,
        )
        if leg == "july" and len(shifted) == want:
            recorded = [float(v) for v in shifted["y"]]
            ok &= checks.add(
                "July: the plan-shifted row's recorded taxi-out",
                f"{rule.plan_shift_july_recorded_s:,.0f} s",
                ", ".join(f"{v:,.0f} s" for v in recorded),
                recorded == [rule.plan_shift_july_recorded_s],
            )
    return ok


def month_check(checks: Checks, records: dict) -> bool:
    """The reference refitted on one held-out month, against the stored fit, every column."""
    month = check_month()
    name = record_name(f"heldout-{month}", "reference", 0)
    if name not in records:
        return True
    kept = read_kept(name)
    stored = pd.read_parquet(REPO / MONTHLY / f"heldout-{month}_{SUBMISSION_KEY}.parquet")
    got, ok = reproduces(
        kept, REPO / MONTHLY / f"heldout-{month}_{SUBMISSION_KEY}.parquet"
    )
    same = len(stored) == len(kept) and np.array_equal(
        stored["mvt_id"].to_numpy(), kept["mvt_id"].to_numpy()
    )
    parts = [f"residual {got}", f"rows in the same order {same}"]
    if same:
        flags = np.array_equal(
            stored["flagged"].to_numpy(dtype=bool), kept["term"].to_numpy()
        )
        icao = np.array_equal(
            stored["icao"].astype(str).to_numpy(), kept["icao"].to_numpy()
        )
        runway = "runway" in kept and np.array_equal(
            stored["runway"].astype(str).to_numpy(), kept["runway"].to_numpy()
        )
        takeoff = np.array_equal(
            pd.to_datetime(stored["takeoff"], utc=True).to_numpy(),
            pd.to_datetime(kept["takeoff"], utc=True).to_numpy(),
        )
        worst = float(
            np.max(
                np.abs(
                    stored["residual_with_term"].to_numpy(dtype=float)
                    - (kept["y"] - kept["with_term"]).to_numpy(dtype=float)
                )
            )
        )
        parts += [
            f"term flags identical {flags}",
            f"airports identical {icao}",
            f"runways identical {runway}",
            f"take-off times identical {takeoff}",
            f"residual with the term {worst:.6g} s",
        ]
        ok = ok and flags and icao and runway and takeoff and worst == 0.0
    else:
        ok = False
    return checks.add(
        f"the reference refitted on held-out {month} against the stored fit of that month",
        "every column identical, both residuals to a largest absolute difference of 0",
        "; ".join(parts),
        ok,
    )


def seed1_check(checks: Checks, records: dict) -> bool:
    """The December reference at seed 1, against the held-out December seed-1 file.

    Held-out December trains on January to November, which is the December
    leg, so its seed-1 fit is this one. Read when the file is on disk, and a
    disagreement is a stop like any other.
    """
    name = record_name("december", "reference", 1)
    path = REPO / MONTHLY_SEEDS / f"{max(MONTHS)}_seed1.parquet"
    if name not in records or not path.exists():
        return True
    got, ok = reproduces(read_kept(name), path)
    return checks.add(
        "the December reference at seed 1 against the held-out December seed-1 fit",
        "a largest absolute difference of 0 on every row",
        got,
        ok,
    )


# --------------------------------------------------------------------------
# the stages


def stage_start(name: str) -> dict:
    return {
        "stage": name,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "t0": time.monotonic(),
        "suspended_before_s": suspended_s(),
        "fits": [],
    }


def stage_end(
    stage: dict, checks: Checks, before: dict, extra: dict | None = None
) -> int:
    after = fingerprint()
    moved = touched(before, after)
    stage.update(
        {
            "wall_s": time.monotonic() - stage.pop("t0"),
            "suspended_s": suspended_s() - stage.pop("suspended_before_s"),
            "peak_rss_gib": peak_gib(),
            "checks": checks.rows,
            "fingerprint": {
                "files": len(before),
                "touched": [str(Path(m).relative_to(REPO)) for m in moved],
            },
            **(extra or {}),
        }
    )
    WORK.mkdir(parents=True, exist_ok=True)
    path = stage_path(stage["stage"])
    if path.exists():
        # A stage run again, or resumed, keeps the record of every earlier run.
        prior = json.loads(path.read_text(encoding="utf-8"))
        earlier = prior.pop("previous", [])
        stage["previous"] = earlier + [prior]
    path.write_text(json.dumps(stage, indent=1, sort_keys=True), encoding="utf-8")
    if moved:
        print("TOUCHED: " + ", ".join(stage["fingerprint"]["touched"]), file=sys.stderr)
        return 1
    if checks.failed:
        print("STOPPED: a check disagreed with its expectation", file=sys.stderr)
        return 1
    print(f"nothing touched; {len(before):,} files checked\nDONE", flush=True)
    return 0


def fit_and_record(frame: Frame, prep: dict, arm: Arm, seed: int, stage: dict) -> None:
    name = record_name(prep["split"].name, arm.key, seed)
    if record_path(name).exists():
        print(f"  {name}: on disk, not refitted", flush=True)
        return
    print(f"\nfitting {name} ({datetime.now(timezone.utc):%H:%M:%S}Z)", flush=True)
    record, kept = fit_one(frame, prep, arm, seed)
    write_record(record, kept)
    rows = booster_rows(kept)
    stage["fits"].append(name)
    print(
        f"  fit {record['fit_s']:.0f} s, predict {record['predict_s']:.0f} s, "
        f"{rows.sum():,} rows read at {rmse((kept['y'] - kept['pred']).to_numpy()[rows]):.2f} s, "
        f"peak {record['peak_rss_gib']:.1f} GiB",
        flush=True,
    )


def preflight(checks: Checks, rule: Rule, seeds_report: Path) -> dict:
    """What is read before anything is fitted: the copies, the floors, the directory."""
    print("\nbefore anything is fitted", flush=True)
    frozen_copies(checks)
    floors = floor_checks(checks, rule, seeds_report)
    checks.add(
        "the records' directory is ignored by git",
        "ignored",
        "ignored" if ignored(WORK) else "not ignored",
        ignored(WORK),
    )
    return floors


def run_legs(rule: Rule, seeds_report: Path) -> int:
    before = fingerprint()
    stage = stage_start("legs")
    checks = Checks()
    floors = preflight(checks, rule, seeds_report)
    if checks.failed:
        return stage_end(stage, checks, before, {"floors": floors})
    frame = Frame()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "stands.json").write_text(json.dumps(frame.stands()), encoding="utf-8")
    preps = {leg.name: frame.prepare(leg) for leg in leg_plan()}
    counts = {k: frame.counts(p) for k, p in preps.items()}
    print("\nthe reference on each leg, and the reproduction check", flush=True)
    for leg in leg_plan():
        fit_and_record(frame, preps[leg.name], reference_arm(), 0, stage)
    if not reproduction_checks(checks, read_records(), rule, floors["october_printed"]):
        return stage_end(stage, checks, before, {"floors": floors, "counts": counts})
    print("\nthe candidates", flush=True)
    for leg in leg_plan():
        for arm in candidate_arms(rule):
            fit_and_record(frame, preps[leg.name], arm, 0, stage)
    return stage_end(
        stage,
        checks,
        before,
        {"floors": floors, "counts": counts, "frame_build_s": frame.build_seconds},
    )


def run_months(rule: Rule, seeds_report: Path) -> int:
    records = read_records()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decision = decide(measure(records, rule), rule, now)
    if decision["next"] != "months":
        print(f"the rule does not ask for the held-out months: {decision.get('outcome')}")
        return 0
    arms = [a for a in candidate_arms(rule) if a.key in decision["fifth_for"]]
    before = fingerprint()
    stage = stage_start("months")
    checks = Checks()
    floors = preflight(checks, rule, seeds_report)
    if checks.failed:
        return stage_end(stage, checks, before, {"floors": floors})
    frame = Frame()
    counts = {}
    for split in month_plan():
        prep = frame.prepare(split)
        counts[split.name] = frame.counts(prep)
        if split.valid_month == check_month():
            fit_and_record(frame, prep, reference_arm(), 0, stage)
            if not month_check(checks, read_records()):
                return stage_end(
                    stage, checks, before, {"floors": floors, "counts": counts}
                )
        for arm in arms:
            fit_and_record(frame, prep, arm, 0, stage)
        del prep
    return stage_end(
        stage,
        checks,
        before,
        {
            "floors": floors,
            "counts": counts,
            "arms": [a.key for a in arms],
            "frame_build_s": frame.build_seconds,
        },
    )


def run_combination(rule: Rule, seeds_report: Path) -> int:
    records = read_records()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decision = decide(measure(records, rule), rule, now)
    if decision["next"] != "combination":
        print(f"the rule does not ask for a combination: {decision.get('outcome')}")
        return 0
    arm = combined([a for a in candidate_arms(rule) if a.key in decision["combine"]])
    before = fingerprint()
    stage = stage_start("combination")
    checks = Checks()
    floors = preflight(checks, rule, seeds_report)
    if checks.failed:
        return stage_end(stage, checks, before, {"floors": floors})
    frame = Frame()
    for split in leg_plan() + month_plan():
        prep = frame.prepare(split)
        fit_and_record(frame, prep, arm, 0, stage)
        del prep
    return stage_end(
        stage,
        checks,
        before,
        {"floors": floors, "arm": asdict(arm), "frame_build_s": frame.build_seconds},
    )


def run_seed1(rule: Rule, seeds_report: Path) -> int:
    records = read_records()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    decision = decide(measure(records, rule), rule, now)
    if decision["next"] != "seed1":
        print(f"the rule does not ask for the seed-1 refit: {decision.get('outcome')}")
        return 0
    winner = decision["winner"]
    arm = next(
        (arm_from(r) for r in records.values() if r["arm"]["key"] == winner),
        None,
    )
    if arm is None:
        raise SystemExit(f"no record of {winner}")
    before = fingerprint()
    stage = stage_start("seed1")
    checks = Checks()
    floors = preflight(checks, rule, seeds_report)
    if checks.failed:
        return stage_end(stage, checks, before, {"floors": floors})
    frame = Frame()
    for leg in leg_plan()[:2]:
        prep = frame.prepare(leg)
        fit_and_record(frame, prep, reference_arm(), 1, stage)
        if leg.name == "december" and not seed1_check(checks, read_records()):
            return stage_end(stage, checks, before, {"floors": floors, "winner": winner})
        fit_and_record(frame, prep, arm, 1, stage)
        del prep
    return stage_end(
        stage,
        checks,
        before,
        {"floors": floors, "winner": winner, "frame_build_s": frame.build_seconds},
    )


def dry_run(rule: Rule, seeds_report: Path) -> int:
    """The legs' row counts, the floors as read, and the fit list. Nothing is fitted."""
    before = fingerprint()
    checks = Checks()
    preflight(checks, rule, seeds_report)
    frame = Frame()
    print("\nthe legs, and nothing fitted:", flush=True)
    for leg in leg_plan():
        prep = frame.prepare(leg)
        got = frame.counts(prep)
        print(
            f"  {leg.name:>8s}: {got['valid']:,} validate, {got['term']:,} the term's, "
            f"{got['plan_shift']:,} the plan-shift rule's, {got['booster']:,} read, "
            f"{got['four_hours_or_less']:,} of them 4 h or less",
            flush=True,
        )
        del prep
    print("\nthe first round's fits, in order:", flush=True)
    for leg in leg_plan():
        print(f"  {record_name(leg.name, 'reference', 0)}", flush=True)
    for leg in leg_plan():
        for arm in candidate_arms(rule):
            print(
                f"  {record_name(leg.name, arm.key, 0)}: {len(columns_of(arm))} columns, "
                f"{params_of(arm, 0)['n_estimators']:,} trees, "
                f"{'weighted' if arm.half_life_months else 'unweighted'}",
                flush=True,
            )
    print(
        f"then, for each candidate that holds the four: {len(month_plan())} held-out months,"
        " and the reference refitted on one of them first",
        flush=True,
    )
    moved = touched(before, fingerprint())
    print(
        "\n" + ("nothing was touched" if not moved else f"TOUCHED: {moved}"), flush=True
    )
    return 1 if (moved or checks.failed) else 0


# --------------------------------------------------------------------------
# the report


def guard(text: str, stands: set[str], movements: set[str]) -> None:
    """Every generated text: no identifier, no work block, no assistant, no dash."""
    pr.check_no_identifiers(text, stands, movements)
    if re.search(WORK_BLOCK_PATTERN, text) or re.search(DISCUSSION_PATTERN, text, re.I):
        raise SystemExit(
            "refusing to write the report: it names a work block or where work was discussed"
        )
    if hygiene.attribution_in(text):
        raise SystemExit("refusing to write the report: it names an assistant or a tool")
    if "\u2014" in text or "\u2013" in text:
        raise SystemExit(
            "refusing to write the report: it carries a dash that is not a hyphen"
        )
    if hygiene.ATTRIBUTION not in text:
        raise SystemExit("refusing to write the report: it lost its OpenStreetMap credit")


def table(header: list[str], rows: list[list[str]], align: str) -> list[str]:
    out = ["| " + " | ".join(header) + " |"]
    out.append("| " + " | ".join("---:" if a == "r" else "---" for a in align) + " |")
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def fmt(x: float, d: int) -> str:
    return f"{x:,.{d}f}"


def signed(x: float, d: int) -> str:
    return f"{x:+,.{d}f}"


def interval(got: dict, d: int) -> str:
    return f"[{signed(got['low'], d)}, {signed(got['high'], d)}]"


def mark(v: bool | None) -> str:
    return "not reached" if v is None else ("holds" if v else "**fails**")


def environment() -> dict:
    return pr.environment()


def build_report(
    records: dict, rule: Rule, checks: Checks, floors: dict, generated: str
) -> str:
    d = rule.contrast_decimals
    measured = measure(records, rule)
    decision = decide(measured, rule, latest_written(records))
    stopped = bool(checks.failed)
    verdict = stop_line(checks.failed) if stopped else verdict_line(decision)
    stages = {s: read_stage(s) for s in ("legs", "months", "combination", "seed1")}
    env = environment()
    lines = [
        "# The three candidates, under the rule of 3 October",
        "",
        f"Generated by `scripts/candidates.py` on {generated} from the fits it recorded under"
        " `data/processed/candidates/`; nothing is refitted to write it, every figure and"
        " every verdict is computed from those records, and the generator refuses to write a"
        " movement identifier or a literal stand identifier.",
        "",
        verdict,
        "",
        "## What was measured",
        "",
        "Three candidates, each measured once before and left on the table, fitted on top of"
        " `+ plan differences` (`SUBMISSION_KEY`) and read against that rung's own fit under"
        " the pre-registration of 3 October, which keeps the candidates, the legs, the"
        " reference and the four conditions of the pre-registration of 28 September and adds"
        " a fifth. Both are in this repository as published copies of the working notes they"
        " were fixed in, frozen at their digests, and the checks below read them first.",
        "",
    ]
    lines += table(
        ["candidate", "what is passed to the fitting code"],
        [
            [
                LABELS["saturation"],
                "the features of the rung with `SATURATION_FEATURES` appended",
            ],
            [
                LABELS["recency"],
                f"row weights halving every {rule.half_life_months} months over the fitting"
                " rows, by `preregistered.recency_weights`, measured from each split's last"
                " training month",
            ],
            [
                LABELS["trees"],
                f"`n_estimators` {rule.more_trees:,}, every other parameter of"
                " `BOOSTER_PARAMS` unchanged",
            ],
        ],
        "--",
    )
    lines += [
        "",
        "**The rows every figure is over** are the validation rows that neither the"
        " record-quality term nor the plan-shift rule covers, both fitted on each split's"
        " training rows as `taxiout.tails` fits them, because the booster predicts no other"
        " row of what ships. The floors were measured over the rows that exclude the term"
        " alone, and are kept as printed. Beside every figure, in its own columns, the same"
        " reading over those rows whose recorded taxi-out is four hours or less: a"
        " diagnostic, which decides nothing.",
        "",
        f"**How a condition is read.** Every contrast is the candidate's RMSE minus the"
        f" reference's over the same rows of the same split, both at the same seed, so"
        f" negative is better. It is read at {d} decimals, the precision at which"
        " `reports/baseline.md` prints every contrast it reads against a floor and every"
        ' floor: "better by more than the floor" asks the printed gain to exceed the'
        ' printed floor, and "not worse by more than the floor" asks the printed loss not'
        " to. The interval beside each is the day-clustered percentile interval of"
        f" `taxiout.validation.cluster_bootstrap`, {BOOTSTRAP_DRAWS:,} draws, one draw of"
        " days for both sides; it is printed and decides nothing.",
        "",
    ]
    legs = (stages.get("legs") or {}).get("counts", {})
    if legs:
        lines += ["### The rows, per leg", ""]
        lines += table(
            [
                "leg",
                "validation rows",
                "the term's",
                "the plan-shift rule's",
                "read",
                "of them 4 h or less",
            ],
            [
                [
                    leg.name.capitalize(),
                    f"{legs[leg.name]['valid']:,}",
                    f"{legs[leg.name]['term']:,}",
                    f"{legs[leg.name]['plan_shift']:,}",
                    f"{legs[leg.name]['booster']:,}",
                    f"{legs[leg.name]['four_hours_or_less']:,}",
                ]
                for leg in leg_plan()
                if leg.name in legs
            ],
            "-rrrrr",
        )
        lines.append("")
    lines += ["## Before anything was read", ""]
    lines += [
        "Each check with what it was read against. The floors and October's figure are read"
        " from where they are printed; the year's floor from the five-seed report of the"
        " held-out months, a working file not yet in this repository, whose sha256 was"
        f" `{floors['seeds_report_sha256']}` when it was read. A disagreement is a stop.",
        "",
    ]
    lines += table(
        ["check", "expected", "got", ""],
        [
            [r["what"], r["expected"], r["got"], "ok" if r["ok"] else "**stop**"]
            for r in checks.rows
        ],
        "---r",
    )
    lines.append("")
    if not stopped:
        lines += section_readings_tables(measured, decision, rule)
    lines += ["## What the rule adopts", "", verdict, ""]
    if not stopped:
        lines += section_readings(measured, decision, rule)
    lines += section_fits(records, stages)
    lines += section_fingerprint(stages)
    lines += [
        f"Run under {env['python']}, {env['lightgbm']}, pandas {env['pandas']},"
        f" numpy {env['numpy']}.",
        "",
    ]
    lines += report_inputs_footer(REPO, report_inputs(records))
    lines += ["", ATTRIBUTION, ""]
    return "\n".join(lines)


def section_readings_tables(measured: dict, decision: dict, rule: Rule) -> list[str]:
    """The four conditions, the fifth, the combination and the seed-1 check."""
    d = rule.contrast_decimals
    lines = ["## The four conditions of 28 September", ""]
    lines += [
        "Each candidate at the ladder's seed against the reference at the same seed. The"
        " three July airports are read from the December fit as well, beside the rule and"
        " not in it.",
        "",
    ]
    rows = []
    cond_by = {(c.split, c.rows): c for c in conditions(rule)}
    for arm in candidate_arms(rule):
        got = measured.get(arm.key, {}).get("seed0", {})
        for leg in leg_plan():
            for which in ("all", "three"):
                r = got.get(f"{leg.name}:{which}")
                if r is None or (leg.name == "october" and which == "three"):
                    continue
                c = cond_by.get((leg.name, which))
                label = f"{leg.name.capitalize()}, {'all ten' if which == 'all' else 'the three'}"
                verdict = (
                    "beside the rule"
                    if c is None
                    else mark(holds(c.kind, r["rows"]["contrast"], c.floor, d))
                )
                rows.append(
                    [
                        LABELS[arm.key],
                        label,
                        fmt(r["rows"]["reference"], d),
                        fmt(r["rows"]["arm"], d),
                        signed(r["rows"]["contrast"], d),
                        interval(r["rows"], d),
                        "" if c is None else f"{c.kind}, {fmt(c.floor, d)}",
                        verdict,
                        f"{fmt(r['short']['reference'], d)} / {fmt(r['short']['arm'], d)} /"
                        f" {signed(r['short']['contrast'], d)}",
                    ]
                )
    lines += table(
        [
            "candidate",
            "read on",
            "reference (s)",
            "candidate (s)",
            "contrast (s)",
            "interval (s)",
            "condition, floor (s)",
            "verdict",
            "4 h or less: reference / candidate / contrast (s), diagnostic",
        ],
        rows,
        "--rrrr--r",
    )
    lines.append("")
    four = decision.get("clears_four")
    if four is not None:
        lines += [
            "**Holding all four:** "
            + (", ".join(LABELS[k] for k in four) if four else "none of the three")
            + ".",
            "",
        ]

    lines += ["## The fifth condition, the twelve held-out months", ""]
    year_rows = {
        k: v["seed0"]["year:all"]
        for k, v in measured.items()
        if "year:all" in v.get("seed0", {})
    }
    if not year_rows:
        four = decision.get("clears_four")
        if four == []:
            why = (
                "Not reached: the rule fits the held-out months only for a candidate that"
                " holds the four conditions, and none did, so none was fitted."
            )
        elif four is None:
            why = "Not reached yet: the four conditions are not all measured."
        else:
            why = (
                "Not reached yet: "
                + ", ".join(LABELS[k] for k in four)
                + " held the four conditions, and the twelve held-out fits are not all"
                " on disk."
            )
        lines += [why, ""]
    else:
        lines += [
            "Each held-out-month split of 2025 (one month held out, the other eleven"
            " training) fitted once at the ladder's seed, and pooled over the year, against"
            " the base rung's own seed-0 fits of the same splits, read from the stored files"
            " under `data/processed/monthly/` and not refitted. Before they were read, the"
            f" reference was refitted on held-out {check_month()} and compared with the"
            " stored file, every column.",
            "",
        ]
        for key, year in year_rows.items():
            mrows = []
            for split in month_plan():
                r = year["months"][split.valid_month]
                mrows.append(
                    [
                        split.valid_month,
                        f"{r['rows']['rows']:,}",
                        fmt(r["rows"]["reference"], d),
                        fmt(r["rows"]["arm"], d),
                        signed(r["rows"]["contrast"], d),
                        f"{fmt(r['short']['reference'], d)} / {fmt(r['short']['arm'], d)} /"
                        f" {signed(r['short']['contrast'], d)}",
                    ]
                )
            v = holds("not worse", year["rows"]["contrast"], rule.floor_year, d)
            mrows.append(
                [
                    "**the year**",
                    f"{year['rows']['rows']:,}",
                    fmt(year["rows"]["reference"], d),
                    fmt(year["rows"]["arm"], d),
                    f"{signed(year['rows']['contrast'], d)} {interval(year['rows'], d)}",
                    f"{fmt(year['short']['reference'], d)} / {fmt(year['short']['arm'], d)} /"
                    f" {signed(year['short']['contrast'], d)}",
                ]
            )
            lines += [f"### {LABELS[key].capitalize()}", ""]
            lines += table(
                [
                    "held out",
                    "rows",
                    "the base rung (s)",
                    "candidate (s)",
                    "contrast (s)",
                    "4 h or less: base / candidate / contrast (s), diagnostic",
                ],
                mrows,
                "-rrrrr",
            )
            lines += [
                "",
                f"Not worse by more than the year's floor, {fmt(rule.floor_year, d)} s:"
                f" {mark(v)}.",
                "",
            ]

    if "combination" in measured or decision.get("next") == "combination":
        lines += ["## The combination", ""]
        combo = decision.get("combination") or {}
        crow = []
        got = measured.get("combination", {}).get("seed0", {})
        for c in conditions(rule):
            r = got.get(f"{c.split}:{c.rows}")
            crow.append(
                [c.label]
                + (
                    ["n/a"] * 4 + [fmt(c.floor, d), mark(combo.get(c.key)), "n/a"]
                    if r is None
                    else [
                        fmt(r["rows"]["reference"], d),
                        fmt(r["rows"]["arm"], d),
                        signed(r["rows"]["contrast"], d),
                        interval(r["rows"], d),
                        fmt(c.floor, d),
                        mark(combo.get(c.key)),
                        f"{fmt(r['short']['reference'], d)} /"
                        f" {fmt(r['short']['arm'], d)} /"
                        f" {signed(r['short']['contrast'], d)}",
                    ]
                )
            )
        lines += table(
            [
                "condition",
                "reference (s)",
                "combination (s)",
                "contrast (s)",
                "interval (s)",
                "floor (s)",
                "verdict",
                "4 h or less: reference / combination / contrast (s), diagnostic",
            ],
            crow,
            "-rrrrr-r",
        )
        lines.append("")

    winner = decision.get("winner")
    if winner and "seed1" in measured.get(winner, {}):
        lines += ["## Seed 1, the sign check", ""]
        srow = []
        for leg in ("december", "july"):
            for which in ("all", "three"):
                r0 = measured[winner]["seed0"][f"{leg}:{which}"]
                r1 = measured[winner]["seed1"].get(f"{leg}:{which}")
                if r1 is None:
                    continue
                s0, s1 = r0["rows"], r1["rows"]
                srow.append(
                    [
                        f"{leg.capitalize()}, {'all ten' if which == 'all' else 'the three'}",
                        fmt(s1["reference"], d),
                        fmt(s1["arm"], d),
                        signed(s0["contrast"], d),
                        signed(s1["contrast"], d),
                        interval(s1, d),
                        (
                            "same sign"
                            if same_sign(s0["contrast"], s1["contrast"], d)
                            else "**not the same sign**"
                        )
                        if which == "all"
                        else "beside the rule",
                        f"{signed(r0['short']['contrast'], d)} /"
                        f" {signed(r1['short']['contrast'], d)}",
                    ]
                )
        lines += table(
            [
                "leg",
                "reference at seed 1 (s)",
                "arm at seed 1 (s)",
                "contrast at seed 0 (s)",
                "contrast at seed 1 (s)",
                "interval at seed 1 (s)",
                "the check",
                "4 h or less: contrast at seed 0 / at seed 1 (s), diagnostic",
            ],
            srow,
            "-rrrrr-r",
        )
        lines.append("")
    return lines


def section_readings(measured: dict, decision: dict, rule: Rule) -> list[str]:
    """Whether any verdict turns on the precision it was read at.

    Each condition, and the seed-1 sign check where it was made, read three
    other ways beside the rule's; then, under each reading, which candidates
    would hold the four conditions, and the fifth where it was measured. The
    rule's reading is the only one that decides anything.
    """
    d = rule.contrast_decimals
    differs: dict[str, list[str]] = {k: [] for k in OTHER_READINGS}
    holders: dict[str, list[str]] = {k: [] for k in ("rule", *OTHER_READINGS)}
    for arm in candidate_arms(rule):
        m = measured.get(arm.key, {}).get("seed0", {})
        held = {k: True for k in holders}
        for c in conditions(rule):
            r = m.get(f"{c.split}:{c.rows}")
            if r is None:
                if c.key in FOUR:
                    held = {k: False for k in holders}
                continue
            rule_says = holds_read("rule", c.kind, r["rows"], c.floor, d)
            held["rule"] &= rule_says
            for k in OTHER_READINGS:
                got = holds_read(k, c.kind, r["rows"], c.floor, d)
                held[k] &= got
                if got != rule_says:
                    differs[k].append(f"{LABELS[arm.key]}, {c.label}")
        for k, v in held.items():
            if v:
                holders[k].append(LABELS[arm.key])
    winner = decision.get("winner")
    for leg, sign in (decision.get("signs") or {}).items():
        r0 = measured[winner]["seed0"][f"{leg}:all"]["rows"]
        r1 = measured[winner]["seed1"][f"{leg}:all"]["rows"]
        t0 = Decimal(f"{r0['arm']:.1f}") - Decimal(f"{r0['reference']:.1f}")
        t1 = Decimal(f"{r1['arm']:.1f}") - Decimal(f"{r1['reference']:.1f}")
        other = {
            "unrounded": (r0["contrast"] < 0 and r1["contrast"] < 0)
            or (r0["contrast"] > 0 and r1["contrast"] > 0),
            "figures": (t0 < 0 and t1 < 0) or (t0 > 0 and t1 > 0),
            "contrast": same_sign(r0["contrast"], r1["contrast"], 1),
        }
        for k, v in other.items():
            if v != sign:
                differs[k].append(f"the seed-1 sign check, {leg.capitalize()}")
    lines = ["## Whether a verdict turns on the precision", ""]
    lines.append(
        "Every condition, and the seed-1 sign check where it was made, read three other"
        " ways beside the rule's: "
        + "; ".join(OTHER_READINGS.values())
        + ". They decide nothing."
    )
    lines.append("")
    for k, words in OTHER_READINGS.items():
        lines.append(
            f"- {words.capitalize()}: "
            + (
                "every one reads as it does at the rule's precision."
                if not differs[k]
                else "these read differently, " + "; ".join(differs[k]) + "."
            )
        )
    lines.append("")
    same = all(holders[k] == holders["rule"] for k in OTHER_READINGS)
    named = ", ".join(holders["rule"]) or "no candidate"
    every = {"rule": "at the rule's precision", **OTHER_READINGS}
    if same:
        lines.append(
            f"Under every reading the same candidates hold the conditions read: {named}."
            " The verdict does not turn on the precision."
        )
    else:
        lines.append(
            "The candidates holding the conditions read differ between readings: "
            + "; ".join(
                f"{every[k]}, {', '.join(v) or 'none'}" for k, v in holders.items()
            )
            + ". The rule's reading is the one that decides."
        )
    lines.append("")
    return lines


def section_fits(records: dict, stages: dict) -> list[str]:
    lines = ["## Every fit", ""]
    lines += [
        "Each fit's own wall time on the monotonic clock, which a suspended machine does"
        " not advance, and each stage's whole, with the time the machine spent suspended"
        " inside it.",
        "",
    ]
    rows = []
    for name, r in sorted(records.items(), key=lambda kv: kv[1]["written_utc"]):
        rows.append(
            [
                f"`{name}`",
                f"{r['n_estimators']:,}",
                f"{r['columns']}",
                "yes" if r["weight_sum"] is not None else "no",
                f"{r['fit_rows']:,}",
                f"{r['fit_s']:,.0f}",
                f"{r['predict_s']:,.0f}",
                f"{r['peak_rss_gib']:,.1f}",
            ]
        )
    lines += table(
        [
            "fit",
            "trees",
            "columns",
            "weighted",
            "fit rows",
            "fit (s)",
            "predict (s)",
            "peak (GiB)",
        ],
        rows,
        "-rr-rrrr",
    )
    lines.append("")
    srows = []
    for name, s in stages.items():
        for run in runs(s):
            srows.append(
                [
                    f"{name}, {run['started_utc']}",
                    f"{len(run['fits'])}",
                    f"{run['wall_s']:,.0f}",
                    f"{run['suspended_s']:,.0f}",
                    f"{run['peak_rss_gib']:,.1f}",
                ]
            )
    lines += table(
        ["stage", "fits", "wall (s)", "suspended (s)", "peak (GiB)"], srows, "-rrrr"
    )
    lines.append("")
    return lines


def runs(stage: dict | None) -> list[dict]:
    """Every run of a stage, the earlier ones first."""
    if stage is None:
        return []
    latest = {k: v for k, v in stage.items() if k != "previous"}
    return list(stage.get("previous", [])) + [latest]


def section_fingerprint(stages: dict) -> list[str]:
    every = [(name, run) for name, s in stages.items() for run in runs(s)]
    moved = [name for name, run in every if run["fingerprint"]["touched"]]
    lines = ["## What each run was allowed to write", ""]
    lines += [
        "Every file under `reports/`, the stored residuals and records of the ladder, the"
        " held-out months and the earlier experiments, and the submissions, by size,"
        " modification time and sha256, before each stage and after it. "
        + (
            "No such file changed in any run; each wrote only under"
            " `data/processed/candidates/`, which git ignores."
            if not moved
            else "Files changed while these ran: " + ", ".join(sorted(set(moved))) + "."
        ),
        "",
    ]
    rows = [
        [
            f"{name}, {run['started_utc']}",
            f"{run['fingerprint']['files']:,}",
            ", ".join(run["fingerprint"]["touched"]) or "none",
        ]
        for name, run in every
    ]
    lines += table(["stage", "files checked", "touched"], rows, "-r-")
    lines.append("")
    return lines


def report_inputs(records: dict) -> list[Path]:
    """Every file the report reads, for the declared-inputs footer."""
    paths = [REPO / BASELINE, REPO / CANDIDATES_TEXT, REPO / RULE_TEXT]
    paths += [
        REPO / LADDER_RESIDUALS / f"{leg}_{SUBMISSION_KEY}.parquet"
        for leg in ("december", "july")
    ]
    if any(r["split"].startswith("heldout-") for r in records.values()):
        paths += [
            REPO / MONTHLY / f"heldout-{m}_{SUBMISSION_KEY}.parquet" for m in MONTHS
        ]
    seeds = REPO / MONTHLY_SEEDS / f"{max(MONTHS)}_seed1.parquet"
    if record_name("december", "reference", 1) in records and seeds.exists():
        paths.append(seeds)
    paths += [WORK / "stands.json"]
    paths += [p for p in sorted(WORK.glob("stage_*.json"))]
    for name in sorted(records):
        paths += [record_path(name), kept_path(name)]
    return paths


def write_report(rule: Rule, seeds_report: Path, generated: str) -> int:
    records = read_records()
    checks = Checks()
    floors = preflight(checks, rule, seeds_report)
    reproduction_checks(checks, records, rule, floors["october_printed"])
    month_check(checks, records)
    seed1_check(checks, records)
    if not (WORK / "stands.json").exists():
        raise SystemExit(
            "no legs stage has passed its checks; there is nothing to report"
        )
    stands = set(json.loads((WORK / "stands.json").read_text(encoding="utf-8")))
    movements: set[str] = set()
    for name in records:
        ids = pd.read_parquet(kept_path(name), columns=["mvt_id"])["mvt_id"].dropna()
        movements |= {f"{int(v)}" for v in ids.unique()}
    before = fingerprint(exempt=REPORT)
    text = build_report(records, rule, checks, floors, generated)
    guard(text, stands, movements)
    REPORT.write_text(text, encoding="utf-8")
    moved = touched(before, fingerprint(exempt=REPORT))
    if moved:
        raise SystemExit(
            "writing the report touched a file it must not: " + ", ".join(moved)
        )
    print(f"wrote {REPORT.relative_to(REPO)}")
    print(
        stop_line(checks.failed)
        if checks.failed
        else verdict_line(decide(measure(records, rule), rule, latest_written(records)))
    )
    return 1 if checks.failed else 0


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--seeds-report",
        type=Path,
        required=True,
        help="the five-seed report of the held-out months, where the year's floor is printed",
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="the legs, the floors, the fit list"
    )
    mode.add_argument("--fit", choices=("legs", "months", "combination", "seed1"))
    mode.add_argument(
        "--report", action="store_true", help="the report, from the records"
    )
    ap.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = ap.parse_args(argv)
    rule = the_rule()
    if args.dry_run:
        return dry_run(rule, args.seeds_report)
    if args.report:
        return write_report(rule, args.seeds_report, args.date)
    run = {
        "legs": run_legs,
        "months": run_months,
        "combination": run_combination,
        "seed1": run_seed1,
    }
    return run[args.fit](rule, args.seeds_report)


if __name__ == "__main__":
    raise SystemExit(main())
