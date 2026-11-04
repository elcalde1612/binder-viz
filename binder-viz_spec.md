# binder-viz

A Streamlit dashboard to load, consolidate, analyze and select protein binder
designs, comparing in-silico predictions (cofolding + Rosetta) against
experimental results.

## Goals

1. Consolidate predictions for the same binder sequence across cofolding models,
   replicates and Rosetta relaxation into one table.
2. Measure which scores (single and composite) best predict experimental binding.
3. Use the best scores to pick candidates for the next experimental round.

**Non-goals:** running the cofolding or Rosetta pipelines. This tool only
consumes their outputs.

---

## Data

### Input layout

```
<run_folder>/
  traj_stats/
    <name>_phase2.csv   # cofolding scores (multiple replicates per sequence)
    <name>_phase3.csv   # Rosetta scores (multiple replicates per sequence)
```

### Folder -> (target, model) mapping

- Each run folder is assigned a **target** and a **cofolding model** in the Setup tab
  (defaults: target = folder name, model parsed from the name if recognizable).
- **A target can have multiple folders.** Examples: one folder per model, or
  several batches of designs for the same target and model.
- Folders sharing the same (target, model) are concatenated. Duplicate
  (target, model, sequence, replicate) rows are detected and reported, with
  the choice to keep first, keep last or error.
- The same sequence may appear for several models, and the same sequence ID may
  appear in several batches.

### Replicates and aggregation

Both phase2 and phase3 contain multiple replicates per sequence (seeds/samples/
relaxed structures). Aggregation is configured **separately per stage** in Setup:

| Mode | Behavior |
|---|---|
| `none` | Keep all replicates as rows (replicate-level analysis) |
| `best` | Keep the single best replicate, chosen by a user-selected **selector score** (default: ipTM for phase2, total Rosetta score for phase3) and its direction. The whole row is kept so values stay coherent, rather than taking the best value per column. |
| `worst` | Same, but keep the worst replicate |
| `mean` / `median` | Per-column aggregate across replicates |

- Also computed alongside any aggregation: `n_replicates`, and optionally the
  std of key scores (replicate consistency can itself be predictive, so it is
  offered as a candidate score in the Analysis tab).
- Per-column overrides are allowed (e.g. `best` for ipTM, `mean` for pLDDT).

### Joining phase2 and phase3

- Keys: `pdbname` (phase2) <-> `description` (phase3). Rosetta appends a suffix
  such as `_0001`, so normalize with `re.sub(r"_\d{4}$", "", description)`.
- **Default join strategy:** aggregate each stage to sequence level first (per the
  settings above), then join on `sequence_id` (target + model + sequence).
- **Replicate-level join** (only when aggregation = `none` for both stages): join
  on the normalized name plus replicate/seed ID, if one can be parsed from the
  names. If it cannot, Setup says so and falls back to the default strategy.
- Fallback: join on sequence if names do not match.
- Join report in Setup: matched, unmatched phase2, unmatched phase3, duplicate
  keys, replicate counts per sequence (histogram). Never drop rows silently.

### Unified schema

Analysis unit = one row per `sequence_id` (or per replicate when aggregation is `none`).

| Column | Description |
|---|---|
| `sequence_id` | Stable ID, hash of target + binder sequence |
| `target` | Target name |
| `sequence` | Binder sequence |
| `n_rep_p2`, `n_rep_p3` | Replicate counts behind each row |
| `<score>__<model>` | Phase2 cofolding scores (ipTM, pLDDT, ipSAE, PAE_interaction, ...) |
| `ros_<score>__<model>` | Phase3 Rosetta scores (dG_separated, SASA, shape complementarity, ...) |

Scores are prefixed by model and stage so nothing collides. The consolidated
table is cached as Parquet, keyed on a hash of the folder contents and config.

### Score registry (known scores + inference)

The user supplies a **score registry** (YAML or CSV) listing known scores. All
other numeric columns are inferred.

```yaml
scores:
  - name: iptm
    stage: phase2
    direction: higher_better
    group: confidence       # optional; used for composites and display
  - name: dG_separated
    stage: phase3
    direction: lower_better
    group: interface
```

- **Known** scores use the supplied stage, direction and group.
- **Inferred** scores: any numeric column not in the registry. The stage comes
  from the source file. The direction is auto-detected from the sign of the
  effect against experimental data (or left as `unknown` if there are no
  labels). They get an "inferred" badge in the UI, direction is overridable, and
  the direction is excluded from the "best score" ranking until confirmed (or
  included, with a toggle).
- The registry can be edited in the Setup tab and saved back to file.
- Warn on registry entries not found in any CSV and on columns with
  excessive NaNs, constant values or non-numeric types.

### Experimental data

Internal and public data share the same class labels. Kinetics are an additional
column analyzed separately.

| Column | Source | Description |
|---|---|---|
| `binding_class` | internal + public | none/NaN, weak, medium, strong |
| `binds` | derived | `binding_class != none` |
| `binding_ord` | derived | 0-3 from the class |
| `source` | both | `internal` / `public` |
| `kd_nM`, `log10_kd`, `kd_qualifier` | internal only | Kinetics (nullable), analyzed in a separate regression analysis |
| `fsec_*` | internal only | FSEC metrics (expression/monodispersity); can be shown alongside but are not outcomes |

- Because both sources carry classes, **class-based metrics can be pooled across
  sources**. Pooled results are shown next to per-source results (toggle),
  since class calibration may differ between labs and assays.
- **Kd is never pooled into class analysis** and never mixed with public
  data. It has its own section (below).
- Censored Kd values (">10 uM", "no binding detected"): keep the qualifier in
  `kd_qualifier` and include them as censored observations. Spearman uses the
  censored value as a bound with ties, and plots mark censored points.
- Experimental rows join to predictions on `sequence_id` (or `target` + `sequence`).
  Report unmatched experimental rows and sequences with multiple measurements
  (aggregate or flag, configurable).

---

## App structure

Streamlit tabs, with shared state in `st.session_state` and cached loaders
(`st.cache_data`).

### 1. Setup tab

- **Folders:** add/remove run folders; assign target and model per folder;
  see the detected `traj_stats` files; see a (target x model) grid with
  folder counts and row counts.
- **Replicate handling:** phase2 and phase3 aggregation mode, selector score and
  direction, per-column overrides (see above).
- **Phase toggles:** phase2 only, or phase2 + phase3.
- **Score registry:** load, edit and save; list of inferred scores with direction
  overrides.
- **Experimental data:** load internal and/or public files, column mapping, class
  label mapping (e.g. "Nan"/"no" -> none).
- **Validation summary:** join report, duplicate handling, missing columns, NaN-heavy
  scores, experimental match rate.
- **Global filters:** target, model, min pLDDT, etc.
- "Load" builds and caches the consolidated dataset used by the other tabs.

### 2. Analysis tab

Requires experimental data. Choose the outcome:

| Outcome | Analysis |
|---|---|
| `binds` | Classification |
| `binding_ord` | Ordinal |
| `log10_kd` | Regression (internal only, separate section) |

**Metrics**

| Outcome | Metrics |
|---|---|
| Binary | AUROC, AUPRC, hit-rate/enrichment in top-k% |
| Ordinal | Spearman, Kendall tau |
| Kd | Spearman against `log10_kd`, censoring-aware |

Every metric has a bootstrap 95% CI and N (positives/negatives), with a warning
when N is small. The bootstrap resamples **sequences**, not replicates, to
avoid treating replicates as independent.

**Sections**

1. **Raw scores:** ranked table and bar chart of all scores by metric (known
   and inferred, filterable by group/stage/model); ROC/PR curves; distributions
   split by outcome; correlation heatmap. Replicate-consistency scores
   (std across replicates) are included as candidates.
2. **Composite scores:**
   - Built-in composites (e.g. ipTM + ipSAE, ipTM - dG).
   - User-defined composites: pick scores, weights and normalization (z-score
     or rank).
   - Fitted composites (logistic/ridge) use cross-validation **grouped by
     target**; coefficients are shown.
3. **Multi-model** (when more than one model is loaded for a target):
   - **Consensus:** mean rank, mean z-score, min (conservative) and number of
     models passing a threshold.
   - **Composite A vs B** (user-selectable: per-model composites or consensus):
     side-by-side metrics, overlaid ROC curves, a paired bootstrap or DeLong
     test of the AUROC difference, and an A-vs-B scatter colored by outcome.
4. **Kinetics (internal only):** the same raw/composite machinery against
   `log10_kd`, with its own plots (score vs log Kd scatter, censored points
   marked).
5. **Stratification:** split any result by target, by source (internal/public)
   and by model; pooled vs per-source toggle for class metrics. Within-target
   and across-target ranking performance are reported separately.
6. **Aggregation sensitivity (optional):** re-run a chosen metric under
   different replicate aggregations (best/mean/worst) to see how robust a score
   is to that choice.

### 3. Candidate selection tab

Available when run folders are loaded. Experimental data is optional and
used only to exclude already-tested designs and to show validated scores.

- Choose the ranking score: any raw, composite or consensus score, with a
  default from the Analysis tab (top AUROC).
- Hard filters: min/max thresholds per score, per target, and agreement
  across N models.
- Diversity: cluster by sequence identity (user threshold) and cap picks per
  cluster and per target.
- Select top N, then manually add/remove in an editable table.
- Per-candidate view: scores, rank in each model, replicate count and spread, and
  flags (models disagree, few replicates, high replicate variance).
- Export: CSV (all scores + selection rationale) and FASTA of selected
  sequences.

---

## Non-functional requirements

- **Reproducibility:** save/load the session config as YAML (folder -> target/model
  mapping, aggregation settings, score registry path, label mapping,
  thresholds, composites); embed it in exports.
- **Performance:** ~10^5-10^6 replicate rows should stay interactive (Parquet cache,
  Polars or pandas, cached computations).
- **Robustness:** tolerate missing columns and NaNs per model; never fail the whole
  load for one bad file; surface warnings in the UI.
- **Testing:** unit tests for name normalization, joins, replicate aggregation
  (best/worst keeps coherent rows), concatenation of multi-folder targets,
  label mapping, score direction inference and metric/CI functions, using small
  fixture CSVs.
- **Docs:** README with the folder layout, an example dataset, the score
  registry format, and a glossary.

## Suggested stack

Streamlit, pandas/polars, scikit-learn, scipy, plotly, pyarrow, pyyaml, pytest.

## Milestones

1. Loader: multi-folder targets, replicate aggregation, join, unified schema, score
   registry, Setup tab with validation.
2. Experimental data loader and label harmonization.
3. Analysis: raw scores, composites, multi-model/consensus, then kinetics.
4. Candidate selection and export.
