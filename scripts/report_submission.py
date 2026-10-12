#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""The scored file's own record, and write ``reports/submission-<date>-v<n>.md``.

Two sections, each computed here and neither changing anything it counts.

**The truncated first hours.** ``taxiout.windows.coverage_start`` reads the
blocks of covered months from the frame itself, so the training frame is one
block, January to December 2025, and the ranking frame two, January 2026 and
July 2026. A feature whose window reaches outside its row's block carries a
flag saying so, and four blocks carry one: ``win_truncated``,
``sched_truncated``, ``lag_truncated`` and ``occ_truncated``. This counts them
per flag and per month on the ranking frame, built exactly as
``scripts/make_submission.py`` builds it, and on the training frame's first
and last months, the two edges of its one block. It changes no feature and
decides nothing: it records how many of the rows the file predicts sit at an
edge, and at which edge.

**The file.** For a version already written: its size and sha256, read from
the file and set beside the manifest ``make_submission.py`` wrote with it; the
target's type as the file stores it, and how many of its values exceed
``taxiout.submission.INT32_SQUARE_LIMIT_S``, past which a scorer that squares
residuals in that type wraps; the
manifest's own counts, with the record-quality term's rows recounted here from
the same two frames; when ``--first-build`` names the directory an earlier
build of the same version was copied to, whether the two files are
byte-identical and the two manifests differ in the time alone; and, once the
one upload has been scored, the two values the upload rule allows to be read
back.

Each version gets its own record, named by its date and its version, so two
uploads on one day do not share a file. The record of 4 October 2026,
``reports/submission-2026-10-04.md``, was written before the version was in the
name and is frozen under the date alone; nothing here rewrites it. Where the
manifest records whether the tail rules were substituted, the record says so
beside the term's count; a manifest written before that field existed renders
as it always did. Where it records the fusion (G1n), the record says which
airports its gate keeps, how many rows it took and the model's sha256. Where it
records group B's new base, the record says so, gives each airport's menu item
and the new base's sha256, and names the fusion's base.

Every number is computed; nothing is hand-typed. The report carries
aggregates only, and the generator refuses to write a movement identifier or a
literal stand identifier.

Needs what ``scripts/make_submission.py`` needs: the challenge data under
``data/challenge/``, the distance matrices and edge sets, the METAR cache and
the OurAirports table. A few minutes; it builds both frames.

**The final file.** With ``--final`` the file read is ``<team>_final.parquet``,
the final phase's one file, and the record carries what its manifest adds: the
flight-plan window and the rows it moved, the parts file, and the days the
fusion was fitted on. ``--upload DIR`` names the directory the one upload wrote
its start and end times, the bucket's listing and the object's metadata to;
the record gives the times, and whether the object's ETag and size equal the
file's MD5 and size. Until the organizers publish a figure for it, the record
says so. Every row this adds reads a manifest field the earlier files lack, so
their records regenerate unchanged but for the date.

Usage
-----
    python scripts/report_submission.py --team funny-firefly --version 4
    python scripts/report_submission.py --team funny-firefly --version 4 \\
        --first-build /path/to/the/first/build --date 2026-09-19
    python scripts/report_submission.py --team funny-firefly --version 12 \\
        --final --upload /path/to/the/upload/records --date 2026-10-11
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# The submission's own loaders and constants, so that the frames counted here
# are the frames it builds rather than a second construction that could drift.
from make_submission import (  # noqa: E402
    DATA,
    ID,
    METAR,
    OURAIRPORTS,
    SUBMISSIONS,
    TAILS_PREDICT_COLUMNS,
    TAILS_TRAIN_COLUMNS,
    load_edge_sets,
    load_matrix,
    sha256,
)
from run_baselines import num, pct, table  # noqa: E402
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.challenge import (  # noqa: E402
    SUBMITTING,
    load_movements,
    load_ranking,
    to_pandas,
)
from taxiout.congestion import DEMAND_HALF_MIN, WINDOWS_MIN  # noqa: E402
from taxiout.destination import load_airports  # noqa: E402
from taxiout.features import (  # noqa: E402
    MOVEMENT_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET,
    airport_fallback,
    build_features,
)
from taxiout.metar import load_metar, regime_by_hour  # noqa: E402
from taxiout.occupancy import COVERAGE_MARGIN_S  # noqa: E402
from taxiout.planlags import HALF_WINDOW_MIN  # noqa: E402
from taxiout.submission import INT32_SQUARE_LIMIT_S, above_square_limit  # noqa: E402
from taxiout.tails import TailModel  # noqa: E402
from taxiout.windows import coverage_end, coverage_start, seconds  # noqa: E402

TIME = "MVT_TIME_UTC_mvt"
FLAGS: tuple[str, ...] = (
    "win_truncated",
    "sched_truncated",
    "lag_truncated",
    "occ_truncated",
)
ANY = "any of the four"
DAY_S = 86400

# What each flag says, read off the constants that set its window.
MEANS = {
    "win_truncated": (
        f"the trailing {max(WINDOWS_MIN)} min before take-off reaches before the"
        " block's start"
    ),
    "sched_truncated": (
        f"{max(DEMAND_HALF_MIN)} min either side of the scheduled time reaches"
        " outside the block"
    ),
    "lag_truncated": (
        f"{HALF_WINDOW_MIN} min either side of take-off reaches outside the block"
    ),
    "occ_truncated": (
        "no earlier arrival at the stand was found, and take-off is within"
        f" {COVERAGE_MARGIN_S // 3600} h of the block's start"
    ),
}


def frames() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Both sides, departures and flags, as ``make_submission.main`` builds them."""
    matrix = load_matrix()
    fallback = airport_fallback(matrix)
    weather = regime_by_hour(load_metar(METAR))
    airports = load_airports(OURAIRPORTS)
    edges = load_edge_sets()
    train_columns = sorted(set(REQUIRED_COLUMNS) | set(TAILS_TRAIN_COLUMNS) | {TARGET})
    rank_columns = sorted(set(REQUIRED_COLUMNS) | set(TAILS_PREDICT_COLUMNS) | {ID})
    out = {}
    for side, load, columns in (
        ("ranking", load_ranking, rank_columns),
        ("training", load_movements, train_columns),
    ):
        dep = to_pandas(load(DATA, columns, phase="DEP")).reset_index(drop=True)
        moves = to_pandas(load(DATA, list(MOVEMENT_COLUMNS))).reset_index(drop=True)
        X = build_features(
            dep,
            matrix,
            fallback=fallback,
            movements=moves,
            edges=edges,
            weather=weather,
            airports=airports,
        )
        missing = [c for c in FLAGS if X[c].isna().any()]
        if missing:
            raise SystemExit(
                f"the {side} frame arrived without {missing}; it is not the frame the "
                "submission is built from"
            )
        out[side] = (dep, X[list(FLAGS)].astype(bool))
        print(f"  {side}: {len(dep):,} departures")
        del moves, X
    return out


def counts(dep: pd.DataFrame, flags: pd.DataFrame, months: list[str]) -> dict:
    """Per month: departures, and per flag the rows set, split by the edge."""
    t = dep[TIME]
    month = pd.to_datetime(t, utc=True).dt.strftime("%Y-%m").to_numpy()
    tt = seconds(t)
    first = tt < coverage_start(t) + DAY_S
    last = tt >= coverage_end(t) - DAY_S
    flags = flags.copy()
    flags[ANY] = flags.any(axis=1)
    out = {}
    for m in months:
        rows = month == m
        got = {"departures": int(rows.sum())}
        for c in flags.columns:
            f = flags[c].to_numpy() & rows
            got[c] = {
                "n": int(f.sum()),
                "first": int((f & first).sum()),
                "last": int((f & last).sum()),
                "elsewhere": int((f & ~first & ~last).sum()),
            }
        out[m] = got
    return out


def label(month: str) -> str:
    return pd.Period(month, freq="M").strftime("%B %Y")


WORDS = ("no", "one", "two", "three", "four", "five", "six")


def blocks(months: list[str]) -> str:
    """The covered blocks of a frame's months, worded: runs of consecutive months."""
    runs: list[list[pd.Period]] = []
    for p in sorted(pd.Period(m, freq="M") for m in months):
        if runs and p == runs[-1][-1] + 1:
            runs[-1].append(p)
        else:
            runs.append([p])

    def one(run: list[pd.Period]) -> str:
        if len(run) == 1:
            return run[0].strftime("%B %Y")
        if run[0].year == run[-1].year:
            return f"{run[0].strftime('%B')} to {run[-1].strftime('%B %Y')}"
        return f"{run[0].strftime('%B %Y')} to {run[-1].strftime('%B %Y')}"

    return f"{WORDS[len(runs)]}, " + " and ".join(one(r) for r in runs)


def section_truncation(tables: dict, rank_months: list[str]) -> list[str]:
    columns = list(FLAGS) + [ANY]
    share_rows, edge_rows = [], []
    for side, by_month in tables.items():
        for m, got in by_month.items():
            n = got["departures"]
            share_rows.append(
                [f"{label(m)} ({side})", num(n)]
                + [f"{num(got[c]['n'])} ({pct(got[c]['n'] / n, 2)})" for c in columns]
            )
            for c in columns:
                edge_rows.append(
                    [
                        f"{label(m)} ({side})",
                        f"`{c}`" if c in FLAGS else c,
                        num(got[c]["first"]),
                        num(got[c]["last"]),
                        num(got[c]["elsewhere"]),
                    ]
                )
    return [
        "## The truncated first hours",
        "",
        "`taxiout.windows.coverage_start` reads the blocks of covered months from"
        " the frame itself. The training frame is one block, January to December"
        f" 2025; the ranking frame is {blocks(rank_months)}. A flag is set"
        " where its feature's window reaches outside the row's block:",
        "",
        *(f"- `{c}`: {MEANS[c]}." for c in FLAGS),
        "",
        "Counted on the ranking frame as `scripts/make_submission.py` builds it, and"
        " on the training frame's first and last months, the two edges of its one"
        " block. A month is the calendar month of the take-off time in UTC, as"
        " `coverage_start` reads it; the share is of that month's departures. No"
        " feature is changed by this, and nothing is decided on it.",
        "",
        *table(
            ["month", "departures"] + [f"`{c}`" for c in FLAGS] + [ANY],
            share_rows,
            "-rrrrrr",
        ),
        "",
        "Where the flagged rows sit: in the first 24 hours of the row's block, in"
        " its last 24 hours, or elsewhere (a scheduled time far from the take-off"
        " can put a `sched_truncated` row there).",
        "",
        *table(
            ["month", "flag", "first 24 h of the block", "last 24 h", "elsewhere"],
            edge_rows,
            "--rrr",
        ),
        "",
    ]


def section_file(
    team: str,
    version: int,
    first_build: Path | None,
    term: dict[str, int],
    final: bool = False,
    upload: Path | None = None,
) -> list[str]:
    name = f"{team}_final" if final else f"{team}_v{version}"
    path = SUBMISSIONS / f"{name}.parquet"
    manifest = SUBMISSIONS / f"{name}.manifest.json"
    head = ["## The file", ""]
    if not path.exists() or not manifest.exists():
        return head + [f"No `{name}.parquet` with its manifest has been written yet.", ""]
    m = json.loads(manifest.read_text(encoding="utf-8"))
    digest = sha256(path)
    template = sha256(DATA / SUBMITTING)
    # The target's type and its exposure to a scorer that squares residuals in
    # that type, read from the parquet itself rather than from the manifest.
    target = pq.read_table(path, columns=[TARGET]).column(TARGET)
    above = above_square_limit(target.to_numpy())

    def agrees(ok: bool) -> str:
        return "yes" if ok else "**no**"

    by_month = m.get("record_quality_rows_by_month") or {}
    bare = m.get("ranking_bare_prior") or {}
    rows = [
        ["file", f"`{path.name}`"],
        ["size (bytes)", num(path.stat().st_size)],
        ["sha256, read from the file", f"`{digest}`"],
        ["the manifest's sha256 agrees", agrees(digest == m.get("file_sha256"))],
        [f"`{TARGET}` type, read from the file", f"`{target.type}`"],
        [
            f"values above {num(INT32_SQUARE_LIMIT_S)} s, the largest residual whose"
            " square fits a signed 32-bit integer",
            num(above),
        ],
        ["template sha256, as read at the build", f"`{m.get('template_sha256')}`"],
        ["the template on disk now agrees", agrees(template == m.get("template_sha256"))],
        ["git revision at the build", f"`{m.get('git_revision')}`"],
        ["tree clean at the build", str(m.get("git_tree_clean")).lower()],
        ["booster", f"{m.get('booster_backend')} {m.get('booster_version')}"],
        ["seeds averaged", num(m.get("seeds", float("nan")))],
        ["rows fitted on", num(m.get("fit_rows", float("nan")))],
        ["record-quality term: covered cells", num(m["record_quality_covered_cells"])],
        ["record-quality term: ranking rows", num(m["record_quality_rows"])],
        *(
            [
                [
                    "the two tail rules' predictions substituted",
                    "yes" if m["tail_rules_substituted"] else "no",
                ],
                [
                    "rows each rule would have taken",
                    ", ".join(
                        f"{k.replace('_', ' ')} {num(v)}"
                        for k, v in m["rows_each_rule_would_take"].items()
                    ),
                ],
            ]
            if "tail_rules_substituted" in m
            else []
        ),
        *(
            [
                [
                    "the term's rows predicted by",
                    "C6b" if m["record_quality_from_c6b"] else "the term",
                ],
                [
                    "rows each rule took",
                    ", ".join(
                        f"{k.replace('_', ' ')} {num(v)}"
                        for k, v in m["rows_each_rule_took"].items()
                    ),
                ],
                [
                    "configuration",
                    "the rung alone"
                    if m["submission_configuration"] is None
                    else ", ".join(m["submission_configuration"]["candidates"])
                    + f"; {m['seeds']} seeds",
                ],
                ["booster file sha256", f"`{m['booster_file_sha256']}`"],
            ]
            if "record_quality_from_c6b" in m
            else []
        ),
        *(
            [
                ["the fusion applied (G1n)", "yes" if m["fusion_g1n_applied"] else "no"],
                [
                    "airports the fusion's gate keeps",
                    ", ".join(a for a, g in m["fusion"]["gate"].items() if g["keeps"]),
                ],
                ["rows the fusion took", num(m["fusion"]["rows_taken"])],
                ["fusion model sha256", f"`{m['fusion']['model_sha256']}`"],
            ]
            if m.get("fusion")
            else []
        ),
        *(
            [
                ["group B's new base applied", "yes" if m["group_b_applied"] else "no"],
                [
                    "the new base per airport",
                    "; ".join(
                        f"{it} at "
                        + ", ".join(
                            a
                            for a, x in m["group_b"]["base_per_airport"].items()
                            if x == it
                        )
                        for it in sorted(set(m["group_b"]["base_per_airport"].values()))
                    ),
                ],
                ["new base sha256", f"`{m['group_b']['new_base_sha256']}`"],
                ["the fusion's base", m["fusion"]["base"]],
            ]
            if m.get("group_b")
            else []
        ),
        *(
            [["days the fusion was fitted on", num(m["fusion"]["fit_days"])]]
            if (m.get("fusion") or {}).get("fit_days")
            else []
        ),
        *(
            [
                [
                    "the flight-plan window applied (W)",
                    "yes" if m["window_applied"] else "no",
                ],
                [
                    "the window's edge, and the clip applied",
                    f"{num(m['window']['w_star_s'])} s; {num(m['window']['w_prime_s'])} s",
                ],
                ["ranking rows with a flight plan", num(m["window"]["rows_with_lobt"])],
                ["rows the window moved", num(m["window"]["rows_moved"])],
                *(
                    [f"rows the window moved in {label(k)}", num(v)]
                    for k, v in sorted(
                        _by_month(m["window"]["rows_moved_by_airport_month"]).items()
                    )
                ),
            ]
            if m.get("window")
            else []
        ),
        *(
            [["parts file sha256", f"`{m['parts_file_sha256']}`"]]
            if m.get("parts_file_sha256")
            else []
        ),
        *(
            [f"record-quality term: ranking rows in {label(k)}", num(v)]
            for k, v in sorted(by_month.items())
        ),
        [
            "the term's rows, recounted here",
            ", ".join(f"{label(k)} {num(v)}" for k, v in sorted(term.items())),
        ],
        ["the recount agrees", agrees({k: v for k, v in by_month.items() if v} == term)],
        ["ranking pairs encoded to the bare prior", num(bare.get("pairs", float("nan")))],
        ["ranking rows on those pairs", num(bare.get("rows", float("nan")))],
        *(
            [
                f"within-variance length ratio, {side} encoding, `{level}`",
                f"{num(got['ratio'], 2) if got['ratio'] is not None else 'n/a'}"
                f" ({'material: k moves with the route' if got['material'] else 'not material'})",
            ]
            for side, levels in (m.get("encoding_length_dependence") or {}).items()
            for level, got in levels.items()
        ),
        ["written (UTC)", str(m.get("written_utc"))],
    ]
    out = head + [
        f"Read from `data/submissions/{path.name}` and the manifest written beside"
        " it, both gitignored. The manifest's counts are the ones"
        " `scripts/make_submission.py` computed while building the file; the"
        " record-quality term's rows are recounted here from the same two frames.",
        "",
        *table(["", ""], rows, "--"),
        "",
        "The parameters the booster was fitted with, as the manifest records them:",
        "",
        *table(
            ["parameter", "value"],
            [[f"`{k}`", f"`{v}`"] for k, v in m.get("booster_params", {}).items()],
            "--",
        ),
        "",
    ]
    if first_build is not None:
        other = first_build / path.name
        other_m = first_build / manifest.name
        if other.exists() and other_m.exists():
            a = json.loads(other_m.read_text(encoding="utf-8"))
            differ = sorted(k for k in set(a) | set(m) if a.get(k) != m.get(k))
            other_digest = sha256(other)
            out += [
                "**Built twice.** The same version was built a second time with"
                " `--overwrite`, the first build having been copied aside.",
                "",
                *table(
                    ["", "first build", "second build"],
                    [
                        ["sha256", f"`{other_digest}`", f"`{digest}`"],
                        [
                            "size (bytes)",
                            num(other.stat().st_size),
                            num(path.stat().st_size),
                        ],
                        [
                            "written (UTC)",
                            str(a.get("written_utc")),
                            str(m.get("written_utc")),
                        ],
                    ],
                    "---",
                ),
                "",
                f"Byte-identical: {agrees(other_digest == digest)}. Manifest fields"
                " that differ: "
                + (", ".join(f"`{k}`" for k in differ) if differ else "none")
                + ".",
                "",
            ]
        else:
            out += [f"No earlier build of `{path.name}` under the directory given.", ""]
    used, score = m.get("scorer_used_pairs"), m.get("scorer_score")
    if final:
        out += section_upload(path, upload, used, score)
    elif used is None and score is None:
        out += [
            "**The scorer.** Not uploaded. The manifest's `scorer_used_pairs` and"
            " `scorer_score` are empty.",
            "",
        ]
    else:
        out += [
            "**The scorer.** The file was uploaded once. What was read back is what"
            " the upload rule allows, and nothing was chosen on it:",
            "",
            *table(
                ["", ""],
                [
                    ["file name the scorer reports", f"`{m.get('scorer_file_name')}`"],
                    ["processed at", str(m.get("scorer_processed_at"))],
                    ["`usedPairs`", num(float(used)) if used is not None else "n/a"],
                    ["score", str(score)],
                ],
                "--",
            ),
            "",
        ]
    return out


def _by_month(by_airport_month: dict[str, int]) -> dict[str, int]:
    """Sum an ``"<ICAO> <YYYY-MM>"`` count over the airports."""
    out: dict[str, int] = {}
    for k, v in by_airport_month.items():
        month = k.split()[-1]
        out[month] = out.get(month, 0) + int(v)
    return out


def section_upload(path: Path, upload: Path | None, used, score) -> list[str]:
    """The one upload of the final file, from the records the upload wrote."""
    head = ["**The upload.**"]
    if upload is None or not (upload / "upload.end").exists():
        return head[:0] + ["**The upload.** Not uploaded.", ""]
    stat = next(
        json.loads(line)
        for line in (upload / "stat.json").read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("{")
    )
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    etag = str(stat.get("etag", "")).strip('"')
    size = int(stat.get("size", -1))
    started = (upload / "upload.start").read_text(encoding="utf-8").strip()
    ended = (upload / "upload.end").read_text(encoding="utf-8").strip()
    rows = [
        ["object", f"`{stat.get('key') or stat.get('name') or path.name}`"],
        ["started (UTC)", started],
        ["ended (UTC)", ended],
        ["the object's ETag is the file's MD5", "yes" if etag == md5 else "**no**"],
        [
            "the object's size is the file's",
            "yes" if size == path.stat().st_size else "**no**",
        ],
    ]
    if used is None and score is None:
        rows.append(["the organizers' figure", "not yet published; recorded when it is"])
    else:
        rows += [
            ["`usedPairs`", num(float(used)) if used is not None else "n/a"],
            ["score", str(score)],
        ]
    return [
        "**The upload.** The file was uploaded once, to the destination and in"
        " the form the organizers set for the final phase. What is read back is"
        " what the upload rule allows, and nothing was chosen on it:",
        "",
        *table(["", ""], rows, "--"),
        "",
    ]


def check_no_identifiers(text: str, stands: set[str], movements: set[str]) -> None:
    """Refuse to write a movement identifier or a literal stand identifier."""
    tokens = set(re.findall(r"(?<![\w.])\d+(?![\w])", text))
    if tokens & movements:
        raise SystemExit(
            f"refusing to write the report: {len(tokens & movements)} movement "
            "identifier(s) would be published"
        )
    exempt = {a.icao for a in AIRPORTS}
    literal = {
        v.strip()
        for v in stands
        if len(v.strip()) >= 3 and re.search(r"[A-Za-z]", v) and re.search(r"[0-9]", v)
    } - exempt
    leaked = [
        v for v in literal if re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", text)
    ]
    if leaked:
        raise SystemExit(
            f"refusing to write the report: {len(leaked)} literal stand "
            "identifier(s) would be published"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument(
        "--first-build",
        type=Path,
        default=None,
        help="directory an earlier build of the same version was copied to",
    )
    ap.add_argument(
        "--final",
        action="store_true",
        help="read <team>_final.parquet, the final phase's one file",
    )
    ap.add_argument(
        "--upload",
        type=Path,
        default=None,
        help="with --final: the directory the one upload wrote its records to",
    )
    ap.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="the date in the report's name; defaults to today, UTC",
    )
    args = ap.parse_args()
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; nothing to report", file=sys.stderr)
        return 1
    out_path = REPO / "reports" / f"submission-{args.date}-v{args.version}.md"

    print("building both frames ...")
    got = frames()
    rank, rank_flags = got["ranking"]
    train, train_flags = got["training"]
    rank_months = sorted(
        pd.to_datetime(rank[TIME], utc=True).dt.strftime("%Y-%m").unique()
    )
    train_months = sorted(
        pd.to_datetime(train[TIME], utc=True).dt.strftime("%Y-%m").unique()
    )
    tables = {
        "ranking": counts(rank, rank_flags, rank_months),
        "training": counts(train, train_flags, [train_months[0], train_months[-1]]),
    }

    # The record-quality term, fitted and applied as the submission fits and
    # applies it, so the manifest's count is checked rather than repeated.
    tail = TailModel().fit(train).predict(rank)
    month = pd.to_datetime(rank[TIME], utc=True).dt.strftime("%Y-%m").to_numpy()
    term = {k: int(v) for k, v in tail.notna().groupby(month).sum().items() if int(v) > 0}

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        f"# The final file, `{args.team}_final`"
        if args.final
        else f"# The scored file, `{args.team}_v{args.version}`",
        "",
        f"Generated by `scripts/report_submission.py` on {generated} from the data"
        " under `data/challenge/` and the file and manifest under"
        " `data/submissions/`. Every figure below is computed there, and the"
        " generator refuses to write a movement or a stand identifier.",
        "",
        *section_truncation(tables, rank_months),
        *section_file(
            args.team, args.version, args.first_build, term, args.final, args.upload
        ),
        # The frames read the distance matrices, so the report carries their
        # attribution, as every report the licence gate reads must.
        "Distance matrices derived from OpenStreetMap: data (c) "
        "OpenStreetMap contributors, ODbL 1.0. Movement and flight data",
        "supplied by EUROCONTROL PRC and the OpenSky Network under the terms",
        "of the PRC Data Challenge 2026.",
    ]
    text = "\n".join(lines).rstrip() + "\n"
    stands = set(rank["STAND_mvt"].dropna().astype(str)) | set(
        train["STAND_mvt"].dropna().astype(str)
    )
    movements = {
        str(int(v)) for v in pd.to_numeric(rank[ID], errors="coerce").dropna().to_numpy()
    }
    check_no_identifiers(text, stands, movements)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
