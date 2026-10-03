# Licence of the files this directory holds

**The CSVs generated here are not covered by this repository's GPLv3.** They are
a Derivative Database of OpenStreetMap and carry the Open Database License
(ODbL) v1.0. Nothing in the repository can change that: the share-alike
obligation belongs to OpenStreetMap's contributors, not to this project, so it
is not ours to waive.

That is why the CSVs are gitignored rather than committed. Every file that *is*
committed to this repository is GPLv3, with no second obligation attached to
any of it. The ODbL question arises only when you run the pipeline and then
share what it wrote.

## If you redistribute these files

Offer them under ODbL 1.0, and carry this notice with them:

> Contains information from OpenStreetMap, which is made available under the
> Open Database License (ODbL). © OpenStreetMap contributors.
> <https://www.openstreetmap.org/copyright>; the licence:
> <https://opendatacommons.org/licenses/odbl/1-0/>

## What is and is not encumbered

ODbL sorts things into three categories, and the distinction is the whole of it.

| | category | obligation |
| --- | --- | --- |
| The CSVs here | Derivative Database | attribution **and** share-alike |
| `reports/*.md` | Produced Work | attribution only |
| Everything in `src/` and `scripts/` | not a database | none: GPLv3 |
| A **feature frame** joining these columns to challenge data | Derivative Database | attribution **and** share-alike |
| A model trained on those features | attribution-only object | attribution in the model's docs |
| Predictions submitted to the challenge | not implicated | none |

A **Derivative Database** is a database built from OSM. A **Produced Work** is
something made *from* a database in another medium (a report, an image, an
analysis), and ODbL §4.3 asks only for a notice that says where the data came
from and that it is available under the ODbL; §4.5 keeps share-alike from
reaching it.

The model row is not a guess. OSMF's Attribution Guidelines, adopted by the
board in June 2021, state that training sets which are substantial extractions
are Derivative Databases needing ODbL if publicly used; that models must be
attributed in documentation such as a README; and that predictions "are not
implicated by ODbL", with a caveat for models overtrained enough to recreate
their training set, which does not describe a gradient-boosted regressor on a
few dozen features.

The consequence that is easiest to forget: **a cached feature frame carrying
these columns is itself a Derivative Database**, so it stays out of the
repository exactly as the matrices do. The share-alike obligation reaches the
CSVs and any table built from them, and stops there. It does not reach the code,
the reports, the model, or the submission.

## Identifiers, and why the reports carry shapes

Stand identifiers are the one class of OSM content with no public-domain
substitute. Runway designators, lengths and threshold positions are published in
aeronautical information publications and in OurAirports, which is public domain,
so they are not OSM's to restrict. A stand identifier at these airports comes
from OSM and nowhere else.

So the coverage report describes identifier *shapes*, `@##` for a pier letter
and two digits, rather than listing examples. A shape answers the question that
matters, which is what normalisation the join will need, and it is a description
rather than an extract. `scripts/check_licence_hygiene.py` verifies that no
literal identifier has crept back in.

## Reproducing the files

```bash
make data      # fetch and cache OSM geometry (Overpass, resumable, ~10 min)
make reports   # rebuild the matrices and every report (~3 min, offline)
```

The build is deterministic: repeated runs produce byte-identical output from the
same cache. `scripts/fetch_osm.py --date` pins the fetch to a moment in OSM's
history, which is what makes a rebuild months from now reproduce the same
geometry rather than whatever the map says then.
