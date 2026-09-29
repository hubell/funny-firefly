#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit: the ranking files the organizers re-issued on 4 September 2026
# replace those of 2 September, every report that reads them is regenerated,
# and the submission is built on the template the scorer counts.
#
# WHAT CHANGED IN THE DATA. The template grew from 215,876 rows to 344,841:
# July 2026 gained its departures at EDDM, LEBL, LEMD, LFPG, LIRF, LSZH and
# LTFM, 128,965 rows, and no row was removed. In the 431,843 ranking rows both
# versions hold, 405,623 changed in their movement, block or taxi times, each
# time by at most 12 s. The columns and the twelve training files did not
# change. reports/ranking-refresh-2026-09-28.md is the dated record.
#
# WHAT MOVES, IN ONE PLACE:
#   * reports/ranking-refresh-2026-09-28.md, new, frozen under its digest.
#   * the nine reports that declare the two files, regenerated in the order
#     their footers dictate: data_review, tails, geometry_gaps, baseline (the
#     ladder), congestion, occupancy, residual, headroom, transfer. What moves
#     in them is what is computed on the ranking set.
#   * scripts/run_baselines.py: RANKING_MIX to the re-issued set's bucket
#     counts, as reports/geometry_gaps.md prints them; the register's entry
#     with it, its basis now naming that report (it named data_review.md,
#     which does not print them); tests/test_strata_pooling.py's own copy of
#     the same counts.
#   * scripts/report_tails.py, review_challenge_data.py, report_geometry_gaps.py
#     and report_occupancy.py: sentences that described the set of 2 September
#     as fixed text (three July airports, a term touching January alone) are
#     computed from the data. Two register citations into
#     review_challenge_data.py move with the lines.
#   * src/taxiout/provenance.py and docs/provenance.md, for all of the above.
#   * tests/test_check_paper.py: the test that held the freeze condition from
#     the figure pass of 25 September now pins what the paper owes the
#     re-issued set, exactly: --strict fails, 20 figures unmatched, one figure
#     matching only in a frozen record without being listed. The paper pass
#     brings both to nothing and flips it back.
#
# IF THE DOCUMENTATION CHANGE OF 27 SEPTEMBER IS STILL UNCOMMITTED (HEAD is
# the freeze commit and scripts/commit_docs.sh is in the tree), this commit
# carries its nine paths too: its edits to src/taxiout/provenance.py and
# docs/provenance.md sit under these, and its own script requires a tree that
# holds its nine paths and nothing else, which this change makes impossible.
# Its checks come with it, but for one: it required the deposit's DOI, and
# that DOI is not reserved yet. So this accepts either the DOI written, one and
# the same in the README and the paper with no mark left, or no DOI at all and
# the three **[DOI]** marks exactly where that change put them (two in the
# README, one in the paper), for the score commit to fill. Half of either
# fails. If that change is committed already, HEAD is its commit and this one
# carries its own paths alone.
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * models.py is the tag's, and no parameter, feature or rule moves.
#   * no validation figure moves: the ladder rebuilt all 52 residual files
#     byte for byte, and the checks below hold every figure of baseline.md
#     outside the ranking-mix columns to the tag's, and residual.md,
#     headroom.md and transfer.md to the tag's but for their dates.
#   * docs/paper.md is not touched. It still describes the ranking set of 2
#     September, so check_paper.py --strict fails, and the count it fails with
#     is the paper pass's work list.
#   * it makes no tag.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

PY="${PYTHON:-$([ -x .venv/bin/python ] && echo "$PWD/.venv/bin/python" || echo python3)}"
RUFF="$([ -x .venv/bin/ruff ] && echo "$PWD/.venv/bin/ruff" || echo ruff)"
case "$PY" in /*) ;; */*) PY="$PWD/$PY" ;; esac

BASE="bfe43f1"
TAG="freeze-2026-10-04"
DOCS_SUBJECT="Say what the clean checkout reproduced, name the deposit, and make the one report it cannot regenerate a dated record"
OWN="docs/provenance.md
reports/baseline.md
reports/congestion.md
reports/data_review.md
reports/geometry_gaps.md
reports/headroom.md
reports/occupancy.md
reports/ranking-refresh-2026-09-28.md
reports/residual.md
reports/tails.md
reports/transfer.md
scripts/commit_ranking_refresh.sh
scripts/report_geometry_gaps.py
scripts/report_occupancy.py
scripts/report_tails.py
scripts/review_challenge_data.py
scripts/run_baselines.py
src/taxiout/provenance.py
tests/test_check_paper.py
tests/test_strata_pooling.py"
DOCS="Makefile
README.md
docs/modelling_notes.md
docs/paper.md
docs/provenance.md
reports/reproduction-2026-09-27.md
scripts/check_penalty_sensitivity.py
scripts/commit_docs.sh
src/taxiout/provenance.py"

# The reviewed state, by digest: the files this change writes, the challenge
# files it reads, the submission it builds, and what the paper checker reports.
SUMS="27ec09048019ba534d6515973cf7827f4b7fac60d4e8263941f53fbbaa1c976e  reports/data_review.md
c1a1d4e7faa0b21e6d7d4dbce65ef2aab3f476fd6d6d3d80f0f328082059ab67  reports/tails.md
1d503b00fc15c0dce7b27aaa37dcecdfdec452e1491e15fae71baf19c9041ff6  reports/geometry_gaps.md
22c195f5ee1f7858bceedd5bce920aff089dc71b9f076558fc19bb543c9c204a  reports/baseline.md
7af0890ebe73db5f75b4996d2707a10a99859a68ad290a1de9184e2e2e8b07f9  reports/congestion.md
649bf0ac04e9465bdb6e552599b94cfac9c3b41b9e78d2aeb08ca057c8ad3951  reports/occupancy.md
59185ee5b0cf4b2ce1dcff2e3117b97af89e0b1055c91c2c0082fc388a1f6114  reports/residual.md
a7a958622c035b351adbb2e5aea988e67d38d668ce6e288297f88bec4ce951bd  reports/headroom.md
8d4bbf9f62ba0f9ef438812c68c10e216e100f1ab816d47c0ec2d3e36c0846ae  reports/transfer.md
03e2d6ca4e42c7f8443044dca37804a6711ed140f16428dbd39f67762d11b8b8  reports/ranking-refresh-2026-09-28.md
a090dd4cf76cc4ffabfaae2f77f32aeb292bef54a5802297f8f002cdf030f9f5  scripts/run_baselines.py
1080c3a60bc8bbf877cddfa4518b7c346f9cd8e83b80621076f648f5e2104396  scripts/report_tails.py
fc15f65e67e071af28abd93e54df59ac733ee9291bd649c54e66530446f9bb16  scripts/review_challenge_data.py
654a2b5456df844d3c93adf2ec2098efd78b044e96ccd11c47dbbda9f5f5ae03  scripts/report_geometry_gaps.py
ca9e672adde37a5c743a105bf8d88be1e4c21ae756984e1027867cde4176c904  scripts/report_occupancy.py
b78d5a1fe4bf1bd5de1be0b9c93af2c84a856645b37cbd298dc7d88357c8fe78  tests/test_strata_pooling.py
2cf8134d664b9d7711a8cdde59116fc23e1189a6d79fe39e3612e3a4c7345f76  tests/test_check_paper.py
b6f6836f45d8409a6e055a0f44a54889a06d1f8cf5fae3111289704b5c9025e5  src/taxiout/provenance.py
0dd941fe696559eeb194cbc822a7f24ecef95b01b26ea987209e405c62e161e8  docs/provenance.md
9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86  data/challenge/ranking.parquet
0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b  data/challenge/submitting.parquet
f8a54d0954d9b9af8aeca471bffb6f495417457a03b9f646dcc88c22521a2841  data/submissions/funny-firefly_v6.parquet
d530ae0ca9eec2cab196cd07c097087520070300efc48a7c37b1c1fff351587f  data/submissions/funny-firefly_v6.manifest.json"
V6_ROWS=344841
SUITE_PASSED=759
# What the paper owes, as check_paper.py --strict reports it and as
# tests/test_check_paper.py pins it.
PAPER_UNMATCHED=20
PAPER_UNLISTED=1
PAPER_UNLISTED_AT="§7.4 line 899: 0.19"

# check_paper.py --strict's output must fail by exactly the reviewed debt.
paper_owes() {
    local out="$1" summary hist only listed
    summary="$(grep -m1 -E '^      [0-9]+ figures in the ' "$out" | sed 's/^ *//')"
    case "$summary" in
        *" $PAPER_UNMATCHED unmatched"*) ;;
        *) echo "the paper checker's summary is not the reviewed one: $summary"; return 1 ;;
    esac
    hist="$(grep -m1 -E '^      [0-9]+ figure\(s\) match only in a frozen dated record, [0-9]+ of them on PAPER_HISTORICAL_FIGURES' "$out" | sed 's/^ *//')"
    only="$(sed -E 's/^([0-9]+) figure.*/\1/' <<< "$hist")"
    listed="$(sed -E 's/.* record, ([0-9]+) of them.*/\1/' <<< "$hist")"
    if [ -z "$hist" ] || [ "$((only - listed))" != "$PAPER_UNLISTED" ]; then
        echo "the unlisted historical figures are not the reviewed $PAPER_UNLISTED: $hist"
        return 1
    fi
    if ! grep -qF "$PAPER_UNLISTED_AT matches only in" "$out"; then
        echo "the unlisted historical figure is not the reviewed one, $PAPER_UNLISTED_AT"
        return 1
    fi
    echo "$summary; $((only - listed)) unlisted historical ($PAPER_UNLISTED_AT)"
}

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say "where this sits"
if [ "$(git rev-parse HEAD)" = "$(git rev-parse "$BASE^{commit}")" ]; then
    STATE="carrying the documentation change of 27 September"
    PATHS="$(printf '%s\n%s\n' "$OWN" "$DOCS" | sort -u)"
    if [ "$(git rev-parse "$TAG^{commit}")" != "$(git rev-parse HEAD)" ]; then
        echo "the tag $TAG does not point at HEAD"
        exit 1
    fi
elif [ "$(git rev-parse HEAD^)" = "$(git rev-parse "$BASE^{commit}")" ] \
     && [ "$(git log -1 --format=%s HEAD)" = "$DOCS_SUBJECT" ]; then
    STATE="on the documentation commit"
    PATHS="$(printf '%s\n' "$OWN" | sort -u)"
else
    echo "HEAD is neither $BASE nor the documentation commit on top of it"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' HEAD): $STATE"

say "the working tree holds these paths and nothing else"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$PATHS" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$PATHS") <(echo "$actual") || true
    exit 1
fi
echo "ok  $(printf '%s\n' "$PATHS" | wc -l) paths"

if [ "$STATE" != "on the documentation commit" ]; then
    say "0/9  the documentation change of 27 September, as its own script checks it"
    # "|| true": under pipefail a grep that finds no DOI would end the script
    # here without a word, before the check below can say what is missing.
    doi_readme="$(grep -oE '10\.5281/zenodo\.[0-9]+' README.md | sort -u || true)"
    doi_paper="$(grep -oE '10\.5281/zenodo\.[0-9]+' docs/paper.md | sort -u || true)"
    marks_readme="$(grep -cF '**[DOI]**' README.md || true)"
    marks_paper="$(grep -cF '**[DOI]**' docs/paper.md || true)"
    if [ -z "$doi_readme" ] && [ -z "$doi_paper" ]; then
        # Not reserved yet: the marks stay, exactly as they were written.
        if [ "$marks_readme" != "2" ] || [ "$marks_paper" != "1" ]; then
            echo "no DOI is written, so the three [DOI] marks must stand as written:"
            echo "expected 2 in README.md and 1 in docs/paper.md, found $marks_readme and $marks_paper"
            exit 1
        fi
        DOI_STATE="marks"
    else
        if [ -z "$doi_readme" ] || [ -z "$doi_paper" ]; then
            echo "the DOI is written in one file and not the other: README '$doi_readme', paper '$doi_paper'"
            exit 1
        fi
        if [ "$(printf '%s\n' "$doi_readme" | wc -l)" != "1" ] || [ "$doi_readme" != "$doi_paper" ]; then
            echo "the DOI is not one and the same in both files: README '$doi_readme', paper '$doi_paper'"
            exit 1
        fi
        if grep -qF '[DOI]' README.md docs/paper.md; then
            echo "the DOI is written but a [DOI] mark is still in place"
            exit 1
        fi
        DOI_STATE="written"
    fi
    if grep -qE '\[settled|\[pending|Working draft' docs/paper.md; then
        echo "docs/paper.md still carries drafting apparatus"
        exit 1
    fi
    n="$(grep -cF '**[score]**' docs/paper.md || true)"
    if [ "$n" != "2" ]; then
        echo "expected the two [score] marks for the upload's commit and found $n"
        exit 1
    fi
    for f in docs/paper.md README.md reports/reproduction-2026-09-27.md Makefile \
             scripts/check_penalty_sensitivity.py; do
        if grep -q -- "$(printf '\xe2\x80\x94')" "$f"; then
            echo "$f carries an em dash"
            exit 1
        fi
    done
    if sed -n '/^\*\*The committed matrices were built from an unpinned fetch/,/^$/p' docs/modelling_notes.md \
       | grep -q -- "$(printf '\xe2\x80\x94')"; then
        echo "the rewritten paragraph of docs/modelling_notes.md carries an em dash"
        exit 1
    fi
    git show "HEAD:reports/determinism.md" > "$work/determinism.head.md"
    if ! cmp -s "$work/determinism.head.md" reports/determinism.md; then
        echo "reports/determinism.md differs from the tag's; a dated record is not edited"
        exit 1
    fi
    for pair in "docs/paper.md|A reader brings four things to the repository at the tag" \
                "docs/paper.md|reports/reproduction-2026-09-27.md" \
                "docs/paper.md|the Zenodo deposit" \
                "README.md|**The deposit.**" \
                "Makefile|reports/reproduction-2026-09-27.md" \
                "scripts/check_penalty_sensitivity.py|reports/reproduction-2026-09-27.md" \
                "docs/modelling_notes.md|reports/reproduction-2026-09-27.md"; do
        if ! grep -qF -- "${pair#*|}" "${pair%%|*}"; then
            echo "missing from ${pair%%|*}: ${pair#*|}"
            exit 1
        fi
    done
    if [ "$DOI_STATE" = "written" ]; then
        doi_said="DOI $doi_readme"
    else
        doi_said="no DOI yet, and the three [DOI] marks as written"
    fi
    echo "ok  $doi_said; no drafting apparatus; 2 [score] marks; no em dash; determinism.md the tag's"
fi

say "1/9  the reviewed state, by digest"
if ! printf '%s\n' "$SUMS" | sha256sum -c --quiet -; then
    echo "a file is not at the digest this change was reviewed at"
    exit 1
fi
for f in data/challenge/ranking.parquet data/challenge/submitting.parquet \
         data/submissions/funny-firefly_v6.parquet data/submissions/funny-firefly_v6.manifest.json; do
    if ! git check-ignore -q "$f"; then
        echo "$f is not ignored by git"
        exit 1
    fi
done
echo "ok  $(printf '%s\n' "$SUMS" | wc -l) files at their digests; the data and the submission ignored"

say "2/9  reports/ moves in the nine regenerated reports and the new record alone"
want_reports="$(printf '%s\n' "$PATHS" | grep '^reports/')"
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- reports/ \
         | sed 's/^...//' | sort)"
if [ "$moved" != "$want_reports" ]; then
    echo "reports/ moves in: $moved"
    exit 1
fi
echo "ok  $(printf '%s\n' "$moved" | wc -l) paths under reports/"

say "3/9  the frozen documents, the new record included, are at their digests"
"$PY" - <<'PY'
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "src")
from taxiout.provenance import CONSTANTS, FROZEN_DOCUMENTS

paths = [f.path for f in FROZEN_DOCUMENTS]
if "reports/ranking-refresh-2026-09-28.md" not in paths:
    raise SystemExit("reports/ranking-refresh-2026-09-28.md is not in FROZEN_DOCUMENTS")
for f in FROZEN_DOCUMENTS:
    if hashlib.sha256(Path(f.path).read_bytes()).hexdigest() != f.sha256:
        raise SystemExit(f"{f.path} is not at its digest")
mix = next(c for c in CONSTANTS if c.name == "RANKING_MIX")
if tuple(mix.literals) != (2496.0, 4126.0, 31299.0, 306920.0):
    raise SystemExit(f"RANKING_MIX in the register reads {tuple(mix.literals)}")
if "geometry_gaps.md" not in mix.basis or "344,841" not in mix.basis:
    raise SystemExit("RANKING_MIX's basis does not name geometry_gaps.md and 344,841")
print(f"  ok  {len(FROZEN_DOCUMENTS)} dated records at their digests; RANKING_MIX sums to 344,841")
PY

say "4/9  docs/provenance.md is what the register generates"
"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import check_provenance as cp

if cp.document() != Path("docs/provenance.md").read_text(encoding="utf-8"):
    raise SystemExit("docs/provenance.md is not what the register generates")
print("  ok")
PY

say "5/9  nothing else under src/, and models.py is the tag's"
if [ -n "$(git -c core.fileMode=false status --porcelain -- src/ | grep -v 'src/taxiout/provenance.py')" ]; then
    echo "src/ moves outside the register"
    exit 1
fi
git show "HEAD:src/taxiout/models.py" > "$work/models.head.py"
if ! cmp -s "$work/models.head.py" src/taxiout/models.py; then
    echo "src/taxiout/models.py differs from HEAD's"
    exit 1
fi
echo "ok  models.py byte-identical to HEAD's"

say "6/9  the 52 residual files at their digests"
if ! ( cd data/processed/residuals && sha256sum -c --quiet - ) <<'RESIDUALS'
10b563b062148276c4b7cbe3ee7cd5113abf88e99428bc26cd0aac8d97514a9e  december_arrivals-nolag.parquet
223676181e6d27be5688a7ef99ee826ba8be07978b3a281494fc89f717ec1b99  december_arrivals.parquet
2aa98fc74f4d4e1b377abf241c4ea44a0f4a09e123cb2ee1757809c81c0814a3  december_callsign.parquet
d999452a1c15ed7b07323ac205734f5ad2721c8952210d2a3cf01c4d7f81d752  december_configuration-nolag.parquet
4803f7ccbba35bc8867a9b1cbd54808ca85a3e23c37f92797f70f024405f1ef9  december_configuration.parquet
b2a721eda78bcdcd20d6855f82f6605852095d9ce9ce09129385a9a9c1c75368  december_corridor-nolag.parquet
92a051c0d61202e8ce52f714c391529cbd6bc11ba540f58703172ce90e184ef3  december_corridor.parquet
a8586a19e28d72a858a80d21ddcdd2997849310a9d609533e1e3340e7fbfec75  december_demand-nolag.parquet
e043c97a74152729aae4fb91f2eb87d45108ba07cea9a6a1eec838bce1fc714b  december_demand.parquet
dc349f589df1a9395b8f8cf26034f33a072bea4ee9d76b7b71150afc1fb1bbab  december_destination.parquet
b2b7fd8597a2d46b7815c29ff1da047744ad85befc29c0aeb37317c905e948c5  december_encoding-added.parquet
0411e7f70d40212c220ba74183bbff39512ade83040c16232f1d7fa979c05219  december_encoding-for-identifiers.parquet
8fa4476fe1c5a960de26fa47d401bdcd93b3f4f0a0b3f66f5bd945b158838c6c  december_encoding-for-stand.parquet
5ff4f0a52461d8ab67854411607d77a1c2f800acc57317fecfdf55989a9420e6  december_identifiers-geometry-aircraft.parquet
5f7ac328bb4104f5079774b705c2b7fb85cc45ba2ae214491f561130e35c3a17  december_identifiers-geometry.parquet
26c1a361ebf19fa7cc63b7819db7a139fbb056465b11a630daa9c7ddf007b8d4  december_identifiers.parquet
2dcea50617df642cf72181a0fc7aca3138f492c417e8421ae1f16688b0d8a16d  december_occupancy-tfree.parquet
30342f76a8f7c6b8a5b80118e04ec270c90f849368f9d04949045a4423b4a27e  december_occupancy.parquet
24bb2ced438d707200df77f6636053247a41c492c0d44f892c599204717dd8ea  december_plan-diffs.parquet
bcffcb58f5548254a35e0ad2c3595358e57646a2c01d5f7a0e9f8d4148a86c74  december_plan-lags.parquet
055e16c734e10bf70b0c13eb5be4de44f5f64bc494402fdcbb37397974d1a597  december_saturation-nolag.parquet
b75a0bf2e45a6e836452d07d5f5c7f807103abf7ed2843496843ec44d5874946  december_saturation.parquet
5f805a514ca717a9d79db860ed994f4cb4650870d61bb158b5a7e2dd8d5f6265  december_throughput-nolag.parquet
a79ea77f7183fb788a91c50404f4e81435d1ecf2e3ed152ec90fb138c3e42fbe  december_throughput.parquet
bde5606d50b3f11439c76538bd99f716128e32115084c21eb820b10c18012c68  december_weather-nolag.parquet
409872f116abcb7e6492393fc557df424eee7d2527a9e0d36adfd5260502ae0c  december_weather.parquet
f83fb10edd12613918297829f5342f3583d494daa3ef48b23f389491b5cfaf08  july_arrivals-nolag.parquet
834e43bdd754a0822640b5f0eaad1a39c65e63a612b2e1cc5cd55290ebc8c54d  july_arrivals.parquet
4953d342b3e3d2b16f62d1e59683c9821edd25f38ac7a2ec27042b278e722b1c  july_callsign.parquet
90c14cc7e1a01650ba5d823dff1a3ebbaf988c8808093e96e9bbafccd3ce34e7  july_configuration-nolag.parquet
a536cbb3d501e53a291a0821cbf711856ffce0aee3c85d74bb6bb8d8cf87cc4f  july_configuration.parquet
63782a3467a8aadcb58817b4f19c8a617f5275b912ec89cf54d50e0e85553318  july_corridor-nolag.parquet
e5fd971db009e71f1b0eb78279fc3c46609ec1ec543cd630b6a73ae74ea88dd4  july_corridor.parquet
97f96b86e4e36f22cf162ffe836b86515bd3b9a9a4b994cce00efce393fc393d  july_demand-nolag.parquet
dc9ff41919a4a65c3f6d73b18b520df77bc78375326df53087dea6441dbe19f3  july_demand.parquet
960d5a135c86606d67210ecc43560c45b9bfdcf901f5b936c6131a62fe995244  july_destination.parquet
8c0381efef1297bbf4e41905f1d8faab1a8c184d3f0c8906817b153185a6f8e8  july_encoding-added.parquet
be1f9aa632b349dcfe3c748bcf15e41c8568226bc2b43262b5546e0fc0fcfd6c  july_encoding-for-identifiers.parquet
af5d05aaaa34129a1c8ad59f64db050cbe502b386488bddb05dc9b9edcf792eb  july_encoding-for-stand.parquet
158b4a5a67eb1807d0355034fb7b595e220dcb3ae9e7e35eb1e7719e9e3ef649  july_identifiers-geometry-aircraft.parquet
ddc7adb78dacd01d1e939d75f60a93800a7f813581c1b9c1accdc2929b9defa3  july_identifiers-geometry.parquet
c684212c083cc7d1a871358287be95e62f93dfeffad905f5e9e1c8bd2cf083e6  july_identifiers.parquet
1fe2078b05c752215d9c967e3cddfd7ceda709274dae08b7b2df4338417a9434  july_occupancy-tfree.parquet
ee7cf852f06fcbf5f62c8eaae06d6605ba77396dffedc88b7669568713184962  july_occupancy.parquet
2696f14eb59d771831f5aa01be2a52eea19acfb262e82116f33676b179ee9fb4  july_plan-diffs.parquet
e60ccaa4c8b71478524c914400817bf5f0135a1dbd36d8e22393a2d0c37c2857  july_plan-lags.parquet
c150d3522db296b77f69c0064cec35ba6d40797b689b377997729f1c2560a67f  july_saturation-nolag.parquet
ae8331a8869a6966f458bcd4db588090570643d1b73bb927dba23f31657ecbe0  july_saturation.parquet
40f88810e15f858022d751554be1f421356aa0500f8c3eeda8fd081b8b2b0279  july_throughput-nolag.parquet
530d241869ff00688cbe16bb35c95455e87c1ba147d12dfa7ec008c39208c463  july_throughput.parquet
7dd63f8f1d09ee5104b0aa24986f9d09309fa18ff18f11bec3f9bb134cb691d0  july_weather-nolag.parquet
c29f8c8588e9212dc19a05c33fc8004e7f6ee7886b629ee59c844b928680e81f  july_weather.parquet
RESIDUALS
then
    echo "a residual file is not at the digest the tag's ladder wrote"
    exit 1
fi
echo "ok  52 residual files, as the tag's ladder wrote them"

say "7/9  no validation figure moved"
for f in baseline residual headroom transfer; do
    git show "HEAD:reports/$f.md" > "$work/$f.head.md"
done
"$PY" - "$work" <<'PY'
import difflib
import re
import sys
from pathlib import Path

work = Path(sys.argv[1])
FOOT = "## What this report was generated from"
DATE = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC|\d{4}-\d{2}-\d{2}|\d{1,2} September 2026"
)
CHALLENGE = re.compile(r"^\| `data/challenge/(ranking|submitting)\.parquet` \| `[0-9a-f]{64}` \|$")


def plain(text):
    return [DATE.sub("DATE", s) for s in text.splitlines() if not CHALLENGE.match(s)]


# residual.md, headroom.md and transfer.md read the residuals and the training
# year, not the ranking set: nothing in them may move but a date.
for name in ("residual", "headroom", "transfer"):
    old = plain((work / f"{name}.head.md").read_text(encoding="utf-8"))
    new = plain(Path(f"reports/{name}.md").read_text(encoding="utf-8"))
    if old != new:
        diff = "\n".join(list(difflib.unified_diff(old, new, lineterm="", n=0))[:20])
        raise SystemExit(f"reports/{name}.md moved beyond its dates:\n{diff}")

# baseline.md: every table cell the tag printed, but the ranking mix and what is
# read at it; every prose figure, but the three sentences quoting the mix.
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SEP = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?%?")
ALLOWED_COLUMNS = {
    "Error by pair frequency": {6},
    "The three arms": {5, 6},
    "What the chosen arm is, in its own terms": {1},
    "The Gate 2c blocks": {5, 6},
}
ALLOWED_PROSE = (
    "**The ranking mix** is the same arithmetic",
    "It is not the best of them pooled",
    "of the gain comes from the `>100` bucket",
)


def parse(text):
    lines = text.split(FOOT, 1)[0].splitlines()
    path, tables, prose, i = [], [], [], 0
    while i < len(lines):
        m = HEAD_RE.match(lines[i])
        if m:
            level = len(m.group(1))
            path = [p for p in path if p[0] < level] + [(level, m.group(2))]
            i += 1
            continue
        if lines[i].startswith("|") and i + 1 < len(lines) and SEP.match(lines[i + 1]):
            j, rows = i + 2, []
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            tables.append((path[-1][1] if path else "", lines[i], rows))
            i = j
            continue
        prose.append(lines[i])
        i += 1
    return tables, prose


ot, op = parse((work / "baseline.head.md").read_text(encoding="utf-8"))
nt, np_ = parse(Path("reports/baseline.md").read_text(encoding="utf-8"))
if [(s, h, len(r)) for s, h, r in ot] != [(s, h, len(r)) for s, h, r in nt]:
    raise SystemExit("baseline.md's tables differ in number, heading, header or rows")
moved = 0
for (sec, _h, orows), (_s, _h2, nrows) in zip(ot, nt):
    for orow, nrow in zip(orows, nrows):
        for k, (a, b) in enumerate(zip(orow, nrow)):
            if a != b:
                if k not in ALLOWED_COLUMNS.get(sec, set()):
                    raise SystemExit(f"baseline.md, {sec}, row {orow[0]!r}, column {k}: {a} -> {b}")
                moved += 1
if len(op) != len(np_):
    raise SystemExit("baseline.md's prose changed in length")
for a, b in zip(op, np_):
    if NUM.findall(DATE.sub("DATE", a)) != NUM.findall(DATE.sub("DATE", b)):
        if not any(k in a and k in b for k in ALLOWED_PROSE):
            raise SystemExit(f"baseline.md prose moved: {a[:120]!r} -> {b[:120]!r}")
        moved += 1
print(f"  ok  baseline.md: {moved} cells and sentences moved, all at the ranking mix;"
      " residual.md, headroom.md, transfer.md moved in their dates alone")
PY

say "8/9  the submission is built on the re-issued template"
"$PY" - "$V6_ROWS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

rows = int(sys.argv[1])
template = Path("data/challenge/submitting.parquet")
sub = pq.read_table("data/submissions/funny-firefly_v6.parquet")
tpl = pq.read_table(template, columns=["MVT_ID_mvt"])
manifest = json.loads(Path("data/submissions/funny-firefly_v6.manifest.json").read_text())
if sub.num_rows != rows or tpl.num_rows != rows:
    raise SystemExit(f"rows: file {sub.num_rows}, template {tpl.num_rows}, want {rows}")
if set(sub.column("MVT_ID_mvt").to_pylist()) != set(tpl.column("MVT_ID_mvt").to_pylist()):
    raise SystemExit("the file's identifiers are not the template's")
if manifest["template_sha256"] != hashlib.sha256(template.read_bytes()).hexdigest():
    raise SystemExit("the manifest was not written against this template")
print(f"  ok  v6: {sub.num_rows:,} rows, the template's identifiers, built at"
      f" {manifest['git_revision'][:7]}, tree clean {manifest['git_tree_clean']}")
PY

say "9/9  the gates; the paper checker fails, and says by how much"
"$RUFF" check .
"$RUFF" format --check .
"$PY" -m pytest -q > "$work/pytest.txt" 2>&1 || {
    tail -30 "$work/pytest.txt"
    echo "the suite fails"
    exit 1
}
if ! tail -1 "$work/pytest.txt" | grep -qE "^$SUITE_PASSED passed(,| in )"; then
    tail -1 "$work/pytest.txt"
    echo "expected the full suite at $SUITE_PASSED passed"
    exit 1
fi
echo "ok  $(tail -1 "$work/pytest.txt")"
"$PY" scripts/check_licence_hygiene.py
"$PY" scripts/check_provenance.py > "$work/provenance.txt" || {
    cat "$work/provenance.txt"
    echo "the provenance gate fails; it cannot be committed failing"
    exit 1
}
n="$(grep -cE '^  ok  ' "$work/provenance.txt" || true)"
if [ "$n" != "14" ]; then
    echo "expected 14 verdicts and found $n; a check that is not called cannot fail"
    cat "$work/provenance.txt"
    exit 1
fi
if "$PY" scripts/check_provenance.py --strict > "$work/strict.txt"; then
    echo "check_provenance.py --strict passes, but its check 14 reads a paper that still"
    echo "describes the set of 2 September; a gate that passes here cannot see"
    exit 1
fi
failed="$(grep -E '^  FAIL ' "$work/strict.txt" | sed -E 's/^  FAIL +//' || true)"
if [ "$failed" != "14 -- every figure in the paper is in a file its section cites" ]; then
    cat "$work/strict.txt"
    echo "check_provenance.py --strict fails on more than the paper"
    exit 1
fi
echo "ok  14 provenance checks green; --strict fails on check 14 alone, the paper"
if "$PY" scripts/check_paper.py --strict > "$work/paper_strict.txt"; then
    echo "check_paper.py --strict passes on a paper that still describes the set of 2 September;"
    echo "a checker that passes here cannot see"
    exit 1
fi
owed="$(paper_owes "$work/paper_strict.txt")" || { echo "$owed"; exit 1; }
echo "ok  check_paper.py --strict fails as reviewed: $owed"
if grep -rqE -- 'Co-authored-b[y]' $PATHS; then
    echo "a committed path carries a co-authorship trailer"
    exit 1
fi
echo "ok  no trailer"

say "staging"
git add -- $PATHS
git -c core.fileMode=false status --short

docs_paragraph=""
if [ "$STATE" != "on the documentation commit" ]; then
    if [ "$DOI_STATE" = "written" ]; then
        deposit="section 2.3 and the README name the Zenodo deposit, $doi_readme."
    else
        deposit="section 2.3 and the README name the Zenodo deposit, whose DOI is
not reserved yet: its three [DOI] marks stay, for the score commit to fill."
    fi
    docs_paragraph="
This commit also carries the documentation change of 27 September, whose own
script could no longer run once this change was in the tree. On 27 September a
fresh clone at freeze-2026-10-04 rebuilt the 35 matrix files and the 52
residual files byte for byte, regenerated fourteen reports in which no figure
moved, and built the submission byte for byte;
reports/reproduction-2026-09-27.md is the dated record of that run.
reports/determinism.md, the one report a clean checkout cannot regenerate,
joins the dated records under its digest, its file unchanged. The paper's
section 10 says what a reader brings and what a clean checkout reproduces;
$deposit The README is in the house style, and three stale statements about
cost and pinning quote the record.
"
fi

git commit -q -F - <<MSG
Read the ranking set the organizers re-issued on 4 September, and build the submission on its template

On 4 September 2026 the organizers re-issued ranking.parquet and
submitting.parquet. The template grew from 215,876 rows to 344,841: July 2026
gained its departures at EDDM, LEBL, LEMD, LFPG, LIRF, LSZH and LTFM, 128,965
rows, and nothing was removed. In the 431,843 ranking rows both versions hold,
405,623 changed in their movement, block or taxi times, each time by at most
12 s. The columns and the twelve training files did not change.
reports/ranking-refresh-2026-09-28.md is the dated record of the comparison.
344,841 is the usedPairs every scored submission reported when the leaderboard
was read on 18 September.

Every report that reads the two files is regenerated, in the order their
footers dictate. The model is the tag's: models.py and every parameter,
feature and rule are unchanged, the ladder rebuilt all 52 residual files byte
for byte, and no validation figure moved. What moved is what is computed on
the ranking set: its census; the share of it on pairs seen a hundred times or
fewer, 9.83% to 11.00%; the record-quality term's ranking rows, 107 to 383,
276 of them LIRF's July departures; the regime exposure over ten July
airports; and the ranking mix the ladder pools at, whose counts RANKING_MIX
carries. The register's basis for RANKING_MIX now names geometry_gaps.md,
where the counts are printed; it named data_review.md, which does not print
them. Seven sentences of generator prose that described the set of 2
September as fixed text, three July airports and a term touching January
alone, are now computed from the data.

The submission is built on the re-issued template:
data/submissions/funny-firefly_v6.parquet, 344,841 rows, sha256
f8a54d0954d9b9af8aeca471bffb6f495417457a03b9f646dcc88c22521a2841, manifest sha256
d530ae0ca9eec2cab196cd07c097087520070300efc48a7c37b1c1fff351587f. It was built on the freeze
commit with this change uncommitted, so its manifest reads git_tree_clean
false.

The paper still describes the ranking set of 2 September, and nothing in
docs/paper.md moves here. check_paper.py --strict fails, and what it fails by
is the paper pass's debt: $PAPER_UNMATCHED figures unmatched, and $PAPER_UNLISTED
figure, 0.19 in section 7.4, matching only in a frozen record without being
listed as history. tests/test_check_paper.py, which held the freeze condition
from the figure pass of 25 September, now pins exactly that debt; the paper
pass brings both counts to nothing and flips it back.
$docs_paragraph
MSG

say "the commit"
git log -1 --stat --format='%h %s'

say "the commit's own tree, in a fresh worktree"
wt="$work/worktree"
git worktree add -q --detach "$wt" HEAD
(
    cd "$wt"
    "$PY" -m pytest -q -rs > "$work/wt_pytest.txt" 2>&1 || {
        tail -30 "$work/wt_pytest.txt"
        exit 1
    }
    tail -3 "$work/wt_pytest.txt"
    "$PY" scripts/check_provenance.py > "$work/wt_provenance.txt" 2>&1 || {
        cat "$work/wt_provenance.txt"
        exit 1
    }
    grep -E "^provenance" "$work/wt_provenance.txt"
    if "$PY" scripts/check_paper.py --strict > "$work/wt_paper.txt" 2>&1; then
        echo "check_paper.py --strict passes in the committed tree, and it should not yet"
        exit 1
    fi
    wt_owed="$(paper_owes "$work/wt_paper.txt")" || {
        echo "$wt_owed"
        echo "the committed tree's paper checker disagrees with the working copy's"
        exit 1
    }
    echo "ok  check_paper.py --strict fails in the committed tree by the same debt: $wt_owed"
) || {
    git worktree remove --force "$wt"
    echo "the committed tree does not pass its own gates."
    exit 1
}
git worktree remove --force "$wt"

say "done"
echo "Nothing is pushed and no tag is made. 'git show HEAD' to read it, 'git reset --hard HEAD^' to undo."
