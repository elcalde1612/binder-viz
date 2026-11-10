# binder-viz

Streamlit dashboard for consolidating protein-binder predictions, comparing
scores with experimental binding, and selecting candidates for experimental
evaluation. Prediction and Rosetta pipelines are not run by this app.

## Run

```bash
conda activate proteinforge
pip install -e '.[test]'
streamlit run app.py
```

Use **Load example** for a small synthetic dataset, or start with **Experimental
Data** at the top of Setup. Enter experimental file/folder paths and click
**Load experimental data** to discover targets before configuring prediction
runs. Experiments are optional: use **New target** for a prediction-only run.
After configuring runs, click **Load**. The app has Setup, Analysis, and
Candidate Selection tabs.

## Appearance

Use the top-right menu's **Theme** group to choose **Light**, **Dark**, or
**System** appearance. Tables, inputs, and charts follow the active theme.
Streamlit 1.63 or newer is required for the paired native palettes.

## Local Inputs

Paths refer to files on the machine running Streamlit. Home-directory paths
such as `~/data` are supported.

- **Run folders:** enter a run, a `traj_stats` folder, a parent folder, or an
	individual phase CSV. Phase CSVs are discovered recursively. **Discover Local
	Runs** adds separate runs from a parent directory so target/model assignments
	remain editable. A parent path entered directly in the table applies that
	row's target/model assignment to all discovered files.
- **Experimental data:** enter a CSV file or a folder. All CSVs in the folder
	and subfolders are loaded with the same source, target, and column mapping.
	Bad files produce warnings without stopping other files from loading.
- **Score registry:** enter a YAML/CSV file or a folder. A folder lists matching
	files; choose one and click **Open registry**.
- **Session:** enter a YAML file or folder in the sidebar. Choose a discovered
	file when needed, then click **Open**. Saving to a folder writes `session.yaml`.
- **Aggregation overrides, class labels, column maps:** accept inline mappings
	or a local YAML/JSON file or folder containing one matching file. Ambiguous
	folders report an error rather than choosing an arbitrary mapping.

Expected prediction layout:

```text
run_folder/
	traj_stats/
		designs_phase2.csv
		designs_phase3.csv
```

Names ending in `_phase2_stats.csv` / `_phase3_stats.csv` also work. Phase2 uses
`sequence`, `pdbname`, and numeric score columns; phase3 uses `description`,
optionally `sequence`, and numeric Rosetta scores. Multiple folders can share a
target/model. Duplicate replicates, missing scores, and unmatched stage rows
are reported in Setup. Select best/worst/mean/median/none separately per stage;
best/worst retain coherent rows unless per-column overrides are specified.

Experimental CSVs use `target`, `sequence` (or `sequence_id`), and
`binding_class` (`none`, `weak`, `medium`, `strong`). Internal data may also
contain `kd_nM`, `kd_qualifier`, and `fsec_*` columns. Configure alternate column
names and class labels in Setup. Kd analysis is separate from class analysis.

Example registry:

```yaml
scores:
	- name: iptm
		stage: phase2
		direction: higher_better
		group: confidence
	- name: dG_separated
		stage: phase3
		direction: lower_better
		group: interface
```

## Experimental Targets and Bulk Runs

Experimental targets are available before any prediction runs are loaded.
In **Discover Local Runs**, choose an experimental target, **New target** with
a custom name, or **Infer from folder**. The run table's target cells offer
experimental targets and configured target names for reassignment.

**Infer from folder** first matches normalized names, then accepts a conservative,
unambiguous fuzzy match to an experimental target. Matches use the experimental
target's spelling; ambiguous or unrecognized names keep the folder name. Explicit
target assignments are preserved.

Run-table changes stay as drafts while you edit multiple cells. Click
**Apply run-folder edits** to commit them together, then **Load** to reload
predictions using the applied configuration.

**Local run-data paths** accepts one path per line, a comma-separated list,
or a JSON/YAML list. Overlapping paths do not add runs twice. The selected
target/model assignment applies to all discovered runs in that batch.

```text
/data/target_a/batch1_af2
/data/target_a/batch2_af2
/data/target_a/opendde
```

Datasets with `binding_strength`, `binding`, `data_origin`, and `KD (M)` are
recognized automatically. With source set to `auto`, `proteinbase` rows are
public and `internal` rows remain internal. Molar Kd values are converted to nM;
public kinetics never enter Kd analysis. Blank strengths retain binary binding
labels without inventing an ordinal strength. Source, column mapping, and Kd
input unit can be overridden in **Experimental Files and Labels**.

## Default Composite and Replicates

For composites spanning phase2 and phase3, enable **Replicate Handling >
Aggregate joined replicates**. Choose **best**, **worst**, or **mean**, and a
composite selector for each model, then click **Load**. This mode replaces the
independent stage aggregation settings and their per-column overrides.

```yaml
joined_replicates:
	mode: best
	selectors:
		esmfold2: composite__esmfold2
		opendde: composite_interface__opendde
```

Models without an explicit selector use `composite__<model>`. Matching uses
target, model, sequence, and replicate ID; missing or ambiguous IDs are errors.
Stages are outer-joined, so unmatched rows remain available and are reported.
Missing ingredients produce missing composite values, sorted last for both
best and worst; groups with no usable selector retain their first row with a
warning. A mean uses the available values for each column.

Composite percentiles are calculated over raw joined replicate rows within each
target/model. Best/worst retain the entire joined row selected by the composite
(higher is better). Mean averages the per-replicate composite values and numeric
measurements, rather than computing a new composite from averaged measurements.
These composite values are preserved in Analysis and Candidate Selection.
Standard deviations and phase-specific counts describe all available replicates.
Model-local built-in and custom weighted composites are supported; cross-model
consensus and fitted scores remain downstream analyses, not replicate selectors.

**Replicate Handling > Selector score** lists numeric columns detected separately
for each stage, plus **Composite (consolidation)** when recognized ingredients
are present. Best/worst select a coherent row using that stage's selector;
its stage-local value is exported as `replicate_composite__<model>` (or
`ros_replicate_composite__<model>` for phase3).

The final `composite__<model>` follows the original consolidation formula:
direction-adjusted percentile ranks on a 0-100 scale, mean ranks within each
group, and group weights of **50% refolding**, **30% electrostatics**, and
**20% Rosetta**. Refolding uses ipTM/ipSAE; electrostatics uses APBS
complementarity/energy and PRODIGY binding energy/log-Kd; Rosetta uses interface
dG, shape complementarity, packstat, and normalized cross-interface energy.
Shared electrostatic metrics are averaged across the two phases before ranking.
Entirely absent groups are omitted and the remaining weights are normalized;
a row missing every value of a present group retains a missing composite.

Final percentiles use unique sequences within each loaded target/model, after
the configured stage aggregation. Retained replicate rows are averaged per
sequence for this calculation. The stage-local selector ranks the available
raw stage rows instead. Neither uses the old dashboard's fixed historical
cohort, and global display filters do not recompute the final percentiles.

## Exploratory Charts

Analysis includes distributions against all loaded sequences, target/source/
binding-class and available FSEC groupings, cross-model score agreement,
highlighted sequence evidence blocks, internal observed Kd, candidate ranking,
and selected cluster counts. These views are available before clicking
**Analyze**, including prediction-only datasets. Highlighting does not change
the nomination shortlist.

**Model Agreement** has independent **X score** and **Y score** selectors and
can color by **Binding strength** or binary **Binding**, with unmeasured designs
kept separate. Analysis labels and highlighted-sequence choices prefer the
experimental CSV's `name` (or `design`/binder-name equivalent), then its `id` or
supplied `sequence_id`. Identifiers retain leading zeros. Unmatched predictions
use their sequence as the display label. An internal target/sequence key remains
separate for joins and selections; CSV identifiers are not used as unique keys.
Reload predictions with **Load** to refresh names in an already-open session.

Distributions default to sequence-averaged observations. **Retained replicate
rows** shows rows retained by Setup aggregation, not discarded raw replicates.
Agreement plots distinguish unmeasured designs; only uncensored internal Kd
controls affinity bubble sizes. The Kd view excludes public kinetics and marks
censored bounds. Missing FSEC, affinity, or clustering data is not fabricated.

## Negative Selection

In **Candidate Selection > Target Specificity / Negative Selection**, choose
**A** as a positive target and **B** as a negative target. Leaving positive
targets empty includes all targets except the negative targets.

Set an off-target score, binding direction, and cutoff for each negative target.
For example, `iptm__af2`, `higher_better`, `0.7` excludes a candidate if its
prediction against B has ipTM **at least 0.7**. A lower-better score excludes
values **at or below** its cutoff. For multiple retained replicates, any
threshold-passing replicate excludes the sequence. Scores use the aggregation
configured in Setup.

Cross-target matching uses the binder **sequence**, not target-specific IDs.
Any known positive experimental measurement against B also excludes the binder.
Turn off **Apply predicted off-target binding thresholds** for experimental-only
negative selection. Each negative target must pass independently; negative
targets themselves are not candidates.

**Exclude sequences without off-target evidence** is on by default. Turning it
off retains unassessed candidates with a missing-evidence flag; missing data is
not evidence of specificity. Global target/pLDDT filters do not remove evidence
used for off-target checks. Prediction thresholds are a screening heuristic,
not proof that a binder is experimentally specific.

The candidate table and CSV include off-target values and evidence status.
CSV/FASTA/ZIP exports embed session configuration, including negative-selection
rules and the missing-evidence policy.

## Tests

```bash
conda run -n proteinforge python -m pytest -q
```

## Glossary

- **Sequence ID:** stable hash of target and normalized binder sequence.
- **Replicate:** a seed, sample, or relaxed structure of the same binder.
- **ipTM:** predicted interface confidence; larger is generally better.
- **dG:** Rosetta interface energy; more negative is generally better.
- **AUROC/AUPRC:** binary binding discrimination metrics.
- **Composite:** weighted, fitted, or multi-model combination of scores.
- **Negative target:** off-target a selected binder should not bind.
