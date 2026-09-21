# Steam Game Recommendation System

A content-based, hardware-aware game recommendation engine built on the
Kaggle "Steam Store Games" dataset family (`steam.csv`,
`steam_description_data.csv`, `steam_requirements_data.csv`).

Tell it a game you like and what PC you have, and it recommends new games
ranked by content similarity, review quality, popularity, player
engagement, and whether your machine can actually run them — with an
explanation for every recommendation.

This is **Version 1**: content-based only. There is no user-interaction
dataset (no per-user ratings/purchases), so this is deliberately **not**
collaborative filtering. See [Limitations](#limitations) and
[Future work](#future-work-phase-2--local-profiles) below.

---

## Project objective

> "Tell me a Steam game I like and what PC I have, and give me the best
> new games I should try, ranked by how similar they are, how
> well-regarded they are, how much players engage with them, and how
> suitable they are for my hardware."

The system explicitly balances three questions for every candidate game:
1. **Would this user probably like this game?** (content similarity)
2. **Can this user's PC reasonably run this game?** (hardware compatibility)
3. **Is this a worthwhile, well-regarded game at all?** (quality/popularity)

It is not simply a "highest rated games" list.

---

## Datasets

| File | Join key | Role |
|---|---|---|
| `steam.csv` | `appid` | Core catalog: name, genres, tags, categories, developer/publisher, ratings, playtime, owners, price, release date, platforms |
| `steam_description_data.csv` | `steam_appid` | Free-text `detailed_description` / `about_the_game` / `short_description` |
| `steam_requirements_data.csv` | `steam_appid` | Semi-structured PC/Mac/Linux system requirements |

`appid` and `steam_appid` are the same integer ID and join cleanly (verified:
`steam.csv` has zero duplicate `appid`s in the 27,075-row reference copy
inspected during development).

### Data schema notes (verified, not assumed)

- `steam.csv` columns actually used: `appid, name, release_date, developer,
  publisher, platforms, categories, genres, steamspy_tags, positive_ratings,
  negative_ratings, average_playtime, median_playtime, owners, price`.
- `owners` is a string range (`"20000-50000"`); the pipeline parses the
  numeric midpoint and logs any row where that parse fails.
- `steam_requirements_data.csv` stores requirements two different ways in
  the same file: a stringified Python dict in `pc_requirements` /
  `mac_requirements` / `linux_requirements` (e.g. `"{'minimum': '...html...',
  'recommended': '...'}"`), **and** pre-split `minimum` / `recommended`
  plain-text columns for Windows. Neither is fully clean — the `minimum`
  column can still contain an embedded `"Recommended: ..."` tail, and
  `recommended` is null in a large fraction of rows (documented as ~48% by
  the dataset). The parser (`src/requirements_parsing.py`) handles both
  representations and the embedded-tail case explicitly, with unit tests
  covering it.

---

## Preprocessing pipeline

`src/dataset_builder.build_unified_dataset()` runs, in order:

1. **Load** all three CSVs (`src/data_loading.py`), verifying required
   columns exist and raising a clear `SchemaError` naming exactly what's
   missing if not — never silently assumes a column exists.
2. **Deduplicate** by ID on all three datasets (`src/data_validation.py`),
   logging exactly how many rows were dropped.
3. **Join coverage report**: how many games have a description, how many
   have requirements, how many description/requirements rows have no
   matching game — logged and returned as a `JoinCoverageReport`, not
   silently swallowed.
4. **Numeric coercion**: malformed values in rating/playtime/price columns
   become `NaN` (logged), not dropped rows.
5. **Feature engineering** (`src/feature_engineering.py`): unified TF-IDF
   text blob, Bayesian-smoothed rating ratio, log-transformed
   popularity/engagement, parsed release age.
6. **Requirements parsing** (`src/requirements_parsing.py`): regex/rule-based
   extraction of OS, CPU (GHz), RAM, GPU, GPU VRAM, storage, DirectX — every
   field tagged `parsed` / `unavailable` / `uncertain`, never guessed.

Run it directly:
```bash
python scripts/build_dataset.py
```
This caches the unified dataset + parsed requirements to `data/processed/`
so the app doesn't rebuild on every launch.

---

## Recommendation methodology

```
USER INPUT (reference game + PC specs)
    -> CONTENT-BASED CANDIDATE GENERATION   (TF-IDF cosine similarity, top ~300)
    -> FILTERS                               (genre, platform, price, min compatibility)
    -> HARDWARE COMPATIBILITY SCORE          (per candidate)
    -> QUALITY / POPULARITY / ENGAGEMENT     (normalized over candidate set)
    -> FINAL WEIGHTED RANKING
    -> TOP-N + EXPLANATIONS
```

### Content model

Unified per-game text = genres + steamspy_tags + categories + developer +
publisher + short_description, with genres/tags repeated 3x and categories
2x in the blob before TF-IDF fitting — this biases TF-IDF term weight
toward curated metadata over noisier free-text description, without a
custom similarity kernel. TF-IDF + cosine similarity
(`src/content_similarity.py`) is the baseline; the `ContentModel` interface
(`fit` / `most_similar` / `similarity`) is designed so a future embedding
model is a drop-in replacement with no changes to ranking or candidate
generation.

### Quality / popularity signals

Raw review counts and owner counts are **never used directly** for
ranking — a game with 10x more reviews isn't automatically 10x better.
- `smoothed_rating_ratio`: Bayesian-smoothed positive ratio (prior: 20
  "neutral" votes at 75% positivity), so a game with 1 review doesn't
  outrank a game with 50,000.
- `log_owners_estimate`, `log_average_playtime`, `log_median_playtime`:
  log1p-transformed so scale doesn't dominate.
- All are min-max normalized to `[0, 1]` over the current candidate set
  before ranking (`src/ranking.attach_normalized_ranking_columns`).

### Ranking weights

```python
content_similarity:      0.40
hardware_compatibility:  0.20
rating_quality:          0.20
popularity:              0.10
engagement:              0.10
```
These are starting points, not scientifically optimized values — see the
comment block above `RANKING_WEIGHTS` in `src/config.py` for the reasoning
behind each one, and they're fully configurable via
`RecommendationRequest.weights` (see `scripts/evaluate.py`'s ablation tests
for how much they change the outcome).

---

## Hardware compatibility methodology

`src/hardware_compatibility.py` compares `UserHardwareSpec` against parsed
requirements **per field**, not with one blanket `user_ram >= game_ram`
rule for everything:
- **OS**: family-matched (windows/mac/linux), not string-equality.
- **CPU / RAM / GPU VRAM / Storage**: numeric threshold comparisons, but
  only when both the requirement *and* the user's value were actually
  parsed/provided — a missing value is excluded from scoring, not treated
  as a failure.
- Score = fraction of *known* comparisons that passed. Fields that
  couldn't be parsed or weren't provided by the user don't count against
  (or for) the game.

Labels: `COMPATIBLE`, `LIKELY COMPATIBLE`, `BORDERLINE`, `NOT RECOMMENDED`,
`UNKNOWN`. **`UNKNOWN` is returned whenever there isn't enough data to say
more** — the system never claims a game runs when the data doesn't support
that claim. This is unit-tested explicitly
(`tests/test_hardware_compatibility.py::test_no_requirements_data_returns_unknown_not_compatible`).

---

## Explanations

Every recommendation shows reasons built only from data actually present
for that game vs. the reference game — shared genres/tags/categories
(computed by set intersection, not assumed), the numeric content
similarity, a rating-quality phrase derived from the smoothed ratio, and
hardware notes. If a hardware field couldn't be determined, the
explanation says so explicitly rather than omitting it silently.

---

## Evaluation (`scripts/evaluate.py`)

Because there is no user-interaction dataset, this **does not** compute
Precision@K, Recall@K, or any metric requiring historical per-user
preferences. It runs:
1. **Similarity sanity checks** — inspect top-5 matches for well-known
   games, report genre overlap as a rough sanity signal (not a ground-truth metric).
2. **Ablation tests** — content-only / content+quality / content+hardware /
   full-weight rankings for the same reference game, confirming each
   signal measurably changes the output (not dead weight).
3. **Data quality summary** — coverage stats for content text and parsed
   requirements.

Run: `python scripts/evaluate.py`

### Limitations of this evaluation
- No true recommendation-quality metric is possible without real user
  interaction data — everything here is a proxy.
- Genre overlap is a rough correctness signal, not a guarantee: a great
  recommendation can share zero genre labels (e.g. matched on tags/mood
  instead).
- Hardware compatibility is only validated against games with actual
  parsed requirements; `UNKNOWN`-labeled games are untested by design,
  never assumed correct.

---

## How to run

```bash
pip install -r requirements.txt

# 1. Place the three raw CSVs in data/raw/:
#    data/raw/steam.csv
#    data/raw/steam_description_data.csv
#    data/raw/steam_requirements_data.csv

# 2. Build the unified dataset (one-time, cached to data/processed/):
python scripts/build_dataset.py

# 3. Run the app (either):
streamlit run app/streamlit_app.py
# or, no-streamlit CLI:
python app/cli.py

# 4. Run the test suite:
pytest tests/ -v

# 5. Run the offline evaluation:
python scripts/evaluate.py
```

> **Note on dataset size**: `steam_description_data.csv` and
> `steam_requirements_data.csv` can be tens of megabytes. The pipeline
> streams them with pandas and doesn't require anything beyond standard
> memory for the ~27K-row version of this dataset family. If a file is
> missing at any of the three paths, the system degrades gracefully
> (descriptions: content text falls back to genres/tags/categories only;
> requirements: every game reports `UNKNOWN` hardware compatibility) rather
> than crashing — see `src/data_loading.py`.

---

## How to add new datasets

The join-key constants live in one place: `src/config.py`
(`STEAM_ID_COL`, `DESCRIPTION_ID_COL`, `REQUIREMENTS_ID_COL`). To add a
new source dataset:
1. Add a loader function to `src/data_loading.py` following the existing
   pattern (verify required columns, return `None` + warn if the file is
   optional/missing, never assume column names without checking).
2. Add its ID column to the join-coverage report in
   `src/data_validation.check_join_coverage`.
3. If it contributes new content signal, extend
   `src/feature_engineering.build_unified_text` or
   `build_quality_popularity_features`.
4. If it contributes a new ranking signal, add it to
   `RankingWeights` in `src/config.py` and to
   `src/ranking.score_candidates`.

---

## Project architecture

```
steam_recommender/
├── src/
│   ├── config.py                 # paths, weights, thresholds — no magic numbers elsewhere
│   ├── data_loading.py            # load + schema verification
│   ├── data_validation.py         # join coverage, dedup, numeric coercion
│   ├── text_utils.py              # HTML stripping, whitespace normalization
│   ├── feature_engineering.py     # unified text blob, quality/popularity features
│   ├── content_similarity.py      # TF-IDF + cosine (ContentModel interface)
│   ├── requirements_parsing.py    # regex-based requirements extraction
│   ├── hardware_compatibility.py  # per-field compatibility scoring engine
│   ├── ranking.py                 # multi-signal weighted final ranking
│   ├── explanations.py            # human-readable, data-grounded explanations
│   ├── dataset_builder.py         # orchestrates the full Phase-1 pipeline
│   ├── recommender.py             # SteamRecommender: UI-independent orchestrator
│   └── profile.py                 # Phase 2 stub interfaces (NOT implemented/wired)
├── app/
│   ├── streamlit_app.py           # Streamlit UI
│   └── cli.py                     # terminal UI (no streamlit dependency)
├── scripts/
│   ├── build_dataset.py           # builds + caches the unified dataset
│   └── evaluate.py                # offline evaluation / ablation tests
├── tests/                          # pytest suite (58 tests as of Phase 1)
├── data/
│   ├── raw/                       # put the 3 source CSVs here
│   └── processed/                 # cached unified dataset + requirements map
└── requirements.txt
```

The recommendation engine (`src/`) has no dependency on either UI — `app/cli.py`
and `app/streamlit_app.py` both call the exact same `SteamRecommender` class.

---

## Limitations

- **No collaborative filtering, by design.** There's no per-user
  interaction dataset in this dataset family, so "similar to what you
  like" means "content-similar to one reference game you pick," not
  "liked by users like you."
- **System requirements text is genuinely messy.** Not every field parses
  for every game — CPU model names in particular are hard to compare
  numerically (a "GHz" number is a weak proxy for real-world CPU
  performance across different architectures/generations). The system is
  built to say `UNKNOWN` rather than guess.
- **Genre/tag overlap-based explanations** only describe *what the data
  says*, not deeper thematic similarity a human might perceive.
- **TF-IDF is a baseline, not a ceiling.** It captures shared vocabulary,
  not deep semantic similarity — two games described in very different
  words but conceptually alike may score lower than expected. This is the
  intended seam for a future embeddings-based `ContentModel`.
- **Ranking weights are configured defaults, not learned or "optimal."**
  See `src/config.py` for the reasoning; they're trivially swappable.

---

## Future work (Phase 2 — local profiles)

**Not implemented in this version.** `src/profile.py` sketches the data
shape only (`UserProfile`: name, favorite/excluded games, hardware,
preferences) and local JSON save/load stubs — nothing is wired into the
recommendation flow. The intended Phase 2 design:

```
Profile -> liked games -> aggregate content preferences -> candidate games
    -> remove already-liked/owned -> hardware compatibility -> ranking
    -> personalized recommendations
```

This remains a personalized **content-based** recommender, not
collaborative filtering, unless real multi-user interaction data is
introduced later. No remote database or authentication is planned —
profiles would be saved as local JSON files only.
