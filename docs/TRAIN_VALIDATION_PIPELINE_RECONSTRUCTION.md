# CRESCENT training and validation pipeline reconstruction

Date: 2026-09-19

Scope: reconstruction of the most recent usable training/validation pipeline in
this source tree. The repository contains one initial Git commit and many copied
experimental versions, so file modification time, imports, data-format
compatibility, directory names, and produced artifacts were considered together.
No source code was changed as part of this analysis.

## 1. Main conclusion

There is no single fully wired script in the repository that represents the
complete latest release pipeline. The latest implementation is distributed over
several modules, and some newest defaults were repurposed for simulation or quick
debugging.

The best reconstruction of the intended release pipeline is:

```text
merged CNA segment tables
    |
    +-- amp matrix construction
    |     preprocess/Version0209/sort.py
    |     preprocess/Version0209/gen_bin_0306_cpp.cpp
    |
    +-- del matrix construction
    |     preprocess/Version0209/gen_bin_del.cpp
    |
    v
40-channel matrix reduction
    preprocess/compress.py
    |
    v
curated positive/negative centers
    center_position.yaml / aaa.py / dated type_centers_index_*.yaml
    + optional coordinate-to-index conversion
      gen_dataset/centers_coord_to_index.py or gen_dataset/TC_convert.py
    |
    v
multi-scale labeled sample generation
    amp: gen_dataset/gen_sample_amp.py + config.yaml
    del: gen_dataset/gen_sample_del.py + config_del.yaml
    |
    v
dataset parsing
    Model/utils/Dataset.py::TSVDataset
    |
    v
model and training helpers
    Model/Models.py::ComplexMultiStreamCNN
    Model/utils/train.py::train_one_epoch
    Model/utils/evaluate.py::evaluate_detailed
    |
    v
leave-one-project-out training/validation
    Model/auto_cross_ct_val.py::auto_cross_val
```

For a release, I would use the modules above as the canonical component set, but
I would not run their current hard-coded `main` blocks unchanged. The current
`auto_cross_ct_val.py` main is configured as a one-epoch, LIHC-only amp run, and
the default center YAML files currently describe `RUBIC5` simulation data rather
than the 20 real cancer projects.

## 2. Recommended canonical component set

| Pipeline role | Recommended latest module | Confidence | Reason |
|---|---|---:|---|
| Raw focal-CNA classification | `preprocess/Version0209/sort.py` | Medium | Latest arm-range classifier; labels arm-level segments and splits files by chromosome. Its subprocess target is not sufficient to choose amp versus del by itself. |
| Amp bin matrix | `preprocess/Version0209/gen_bin_0306_cpp.cpp` | High | Latest compiled general/focal matrix builder; consumes sorted per-chromosome data and emits `start,end,chr,<samples>,sum`. |
| Del bin matrix | `preprocess/Version0209/gen_bin_del.cpp` | High | Explicit deletion-only builder; CN<2 boundaries, diploid baseline 2, ignores amplification records. |
| 40-channel reduction | `preprocess/compress.py` | High | Latest shared reduction script, selected by `mut`; produces the matrix schema expected by current generators. |
| Coordinate center source | `gen_dataset/center_position.yaml` or curated Python source such as `gen_dataset/aaa.py` | Medium | These contain real-project manual/tool-curated center coordinates, but the repository does not identify one final authoritative source. |
| Coordinate-to-index converter | `gen_dataset/centers_coord_to_index.py` | High for reusable release | Clean CLI, explicit input/output options, and debug metadata support. |
| Newer one-off converter | `gen_dataset/TC_convert.py` | Medium | Newer by timestamp and adds deduplication/reporting, but paths and output names are hard-coded for a specific experiment. |
| Amp sample generator | `gen_dataset/gen_sample_amp.py` + `gen_dataset/config.yaml` | High | Newest modular amp generator with external YAML configuration and indexed centers. |
| Del sample generator | `gen_dataset/gen_sample_del.py` + `gen_dataset/config_del.yaml` | High | Newest modular del generator; supports index or Mb centers and external configuration. |
| PyTorch dataset | `Model/utils/Dataset.py` | High | Latest parser imported by current LOO and k-fold scripts. |
| Model | `Model/Models.py` | High | Current model imported by the latest LOO script. |
| One-epoch trainer | `Model/utils/train.py` | High for current GPU path | Current imported trainer with mixed precision and gradient clipping. |
| Validation metrics | `Model/utils/evaluate.py` | High | Current imported evaluator using sigmoid, BCE loss, PR-derived threshold, F1, AUC, and confusion matrix. |
| LOO orchestrator | `Model/auto_cross_ct_val.py::auto_cross_val` | High for latest features; medium as checked-in runnable main | Newest leave-one-project-out function, latest resource logging and multi-checkpoint export, F1-first model selection. Checked-in run settings are debug/experiment defaults. |

## 3. Complete reconstructed flow

### 3.1 Input data

The common starting point is:

```text
Data/merged_dataframe/merged_dataframe_<CANCER>.tsv
```

The expected columns include at least aliquot/sample ID, chromosome, segment
start/end, and total copy number. The amp preprocessing path additionally uses
chromosome-arm ranges from:

```text
Data/ChromosomeData2025/GRCh38_Chromosome_Arm_Ranges.tsv
```

### 3.2 Chromosome matrix preprocessing

#### Amp path

The most coherent amp chain is:

1. `preprocess/Version0209/sort.py`
   - reads the merged cancer table;
   - calculates segment length and chromosome-arm overlap;
   - marks a segment arm-level when it covers at least 30% of its arm;
   - marks cross-arm/centromere cases specially;
   - writes per-chromosome sorted files.

2. `preprocess/Version0209/gen_bin_0306_cpp.cpp`
   - removes records marked `is_arm_level`;
   - builds bins from the retained segment endpoints;
   - assigns each covered sample its copy number;
   - substitutes diploid value 2 where no value was assigned;
   - emits `start`, `end`, `chr`, sample columns, and a final `sum` column.

The Python equivalent `gen_dataset/gen_bin_0306.py` implements the same broad
idea but is older and slower. The C++ implementation is the preferred release
candidate.

Important wiring detail: `sort.py` currently launches an executable named
`./gen_bin_del`. That makes the `sort.py` main block a mixed amp/del orchestration
artifact, not a reliable amp entrypoint. For amp, the output of `sort.py` should
feed the executable compiled from `gen_bin_0306_cpp.cpp`.

#### Del path

Use `preprocess/Version0209/gen_bin_del.cpp`:

1. Read the merged real-cohort CNA table.
2. Group records by chromosome.
3. Use only CN<2 records to define bin endpoints.
4. Initialize every sample/bin to diploid value 2.
5. For a covering deletion, subtract `2 - Copy_Number`.
6. Write one `cnv_<chromosome>.tsv` matrix per chromosome.

This module is a genuinely different preprocessing method, not just a version of
the amp builder.

### 3.3 Reduce the patient axis to 40 channels

Use `preprocess/compress.py` after selecting `mut="amp"` or `mut="del"` and the
appropriate input/output roots.

The script:

- preserves three metadata columns;
- treats the final column as a suffix;
- pads with diploid value 2 if fewer than 40 feature columns exist;
- selects the 40 highest-mean columns as anchors;
- assigns remaining sample columns to anchors round-robin and averages them;
- writes tab-separated matrices.

This is the direct upstream format expected by `gen_sample_amp.py` and
`gen_sample_del.py`.

The suffix convention deserves explicit release documentation. It naturally
matches amp matrices because they end with `sum`. The inspected del C++ writer
does not add a named suffix column, so the exact del compression input schema
must be fixed or documented when packaging the release.

### 3.4 Dataset selection and labeling

The release-era training labels are represented as center lists rather than a
label column in the raw chromosome matrix:

```yaml
TYPE_CENTERS:
  <cancer_type>:
    <chromosome>:
      pos: [center1, center2, ...]
      neg: [center1, center2, ...]
```

The center list is the dataset selector:

- `pos` creates a recurrent-CNA sample with label 1;
- `neg` creates a background/non-recurrent sample with label 0;
- centers not listed are not part of the supervised training set.

#### Recommended labeling route for a release

1. Curate centers in genomic coordinates (Mb), using GISTIC, RUBIC and manual
   heatmap review.
2. Store the reviewed centers in one clearly named source YAML.
3. Convert coordinates to row indices against the exact compressed matrix using
   `gen_dataset/centers_coord_to_index.py`.
4. Preserve the converter's debug metadata so each index remains traceable to
   its original coordinate and matched bin.
5. Pass the resulting index YAML explicitly to the generator through
   `GENSAMPLES_TYPE_CENTERS`.

#### Real-cohort center files found

The current repository contains several incompatible real-cohort candidates:

| File | Cancer types | Positive centers | Negative centers | Interpretation |
|---|---:|---:|---:|---|
| `center_position.yaml` | 20 | 1,074 | 877 | Coordinate-based, broad real-cohort source; best candidate for re-indexing and auditability. |
| `type_centers_index_0908.yaml` | 20 | 1,235 | 1,242 | Largest indexed real-cohort set; directory names indicate it generated the `baseline_improved_0908` training data used by several LOO scripts. |
| `type_centers_index_0911improved.yaml` | 20 | 699 | 816 | Later, stricter/improved 20-project selection. A plausible final curated alternative. |
| `type_centers_index_0912.yaml` | 15 | 901 | 829 | Later but incomplete for a 20-project LOO experiment; likely a subset/experiment. |
| `type_centers_index_L.yaml` | 20 | 930 | 928 | Another complete 20-project alternative; its intended experiment is not documented in the filename. |

The counts above were read directly from the YAML structures. None of these
files declares itself canonical, so the scientific label set cannot be selected
from code chronology alone.

#### Current default center files are simulation data

The newest default files are not the real pan-cancer training set:

- `type_centers_index.yaml`: one type named `RUBIC5`, 45 positive and 45 negative
  centers;
- `type_centers_del.yaml`: one type named `RUBIC5`, 54 positive and 46
  negative centers;
- `rubic_sim_a.yaml`: one type named `RUBIC1`, 45 positive and 38 negative
  centers.

Therefore, invoking the newest generators with their default center files builds
a simulated RUBIC dataset, not the 20-project LOO dataset.

### 3.5 Alternative labeling method: automatic GISTIC/RUBIC intersection

`preprocess/Version0209/label.py` is a genuinely different labeling method:

- it loads intervals produced by RUBIC and GISTIC;
- marks each matrix row as inside RUBIC, inside GISTIC, or inside both;
- defines the intersection as `in_B_and_C`;
- is hard-coded for BRCA amp and writes labeled matrices.

This module does not directly produce the `TYPE_CENTERS` format consumed by the
current sample generators. It is best treated as an upstream automatic candidate
selector, followed by manual review and conversion to center lists. It should be
listed as an alternative, not mixed silently with the curated-center method.

Supporting selection/inspection modules include:

- `preprocess/Version0209/gistic_after.py`: converts GISTIC lesion output to
  chromosome/start/end intervals;
- `preprocess/Version0209/vis_2025.py`: overlays GISTIC/RUBIC intervals on the
  chromosome matrix for visual review;
- `gen_dataset/TC_convert.py`: loads a Python `TYPE_CENTERS` object, deduplicates
  approximate coordinates, converts them to indices, and reports counts.

### 3.6 Generate labeled multi-scale TSV samples

#### Amp

Use:

```text
gen_dataset/gen_sample_amp.py
gen_dataset/config.yaml
explicit real-cohort TYPE_CENTERS YAML
```

The latest amp generator:

- loads matrix and center configuration externally;
- requires index-form centers;
- performs bounds checks;
- applies full-matrix column z-normalization and clips to [-10, 10];
- extracts one centered matrix for each configured scale;
- sorts/fits patient columns;
- writes one TSV per center with label 1 or 0.

Current `config.yaml` scales are 50x40, 250x40 and 2000x40. The article and older
stable scripts use 50x40, 100x40 and 2000x40. This is a configuration choice that
must be resolved for the release.

#### Del

Use:

```text
gen_dataset/gen_sample_del.py
gen_dataset/config_del.yaml
explicit real-cohort deletion TYPE_CENTERS YAML
```

The latest del generator:

- supports index or Mb center values;
- defaults to column-wise full-matrix z-normalization;
- clips values to [-10, 10];
- extracts the configured scale triplet;
- emits the same labeled TSV structure as amp.

Current `config_del.yaml` scales are 20x40, 50x40 and 400x40. Historical
`gen_sample_del_old.py` uses 50x40, 100x40 and 500x40 and contains embedded labels
for UCEC, KICH, KIRC, KIRP and COAD. Other older deletion generators contain
embedded labels for different cancer subsets. These embedded label sets are not
a complete, clean substitute for one external deletion label YAML.

### 3.7 Load samples with the current Dataset class

Use `Model/utils/Dataset.py::TSVDataset`.

It:

- discovers flat `*.tsv` sample files;
- optionally filters by chromosome;
- optionally excludes filenames from a text list;
- reads `label:` and any number of `scale:` blocks;
- converts each scale to a float tensor shaped `[1, H, W]`;
- returns `(matrices, label, filename)`;
- exposes cancer type as the filename prefix before the first underscore.

That filename convention is essential to LOO splitting. A file named
`BRCA_chr1_..._pos.tsv` belongs to BRCA; a simulated file named
`RUBIC5_chr1_...` is treated as a separate project called RUBIC5.

### 3.8 Model and training helpers

Use:

```text
Model/Models.py
Model/utils/train.py
Model/utils/evaluate.py
```

`ComplexMultiStreamCNN` creates one branch per parsed scale. Each branch contains
a separable convolution, batch normalization, max pooling, three residual
blocks, ECA channel attention, adaptive pooling, and a projection to a common
feature vector. Four-head self-attention and a feed-forward residual block fuse
the scale vectors, after which mean pooling and an MLP emit one logit.

`train_one_epoch` uses mixed precision, BCE-with-logits loss supplied by the
caller, and gradient clipping at 0.9.

`evaluate_detailed`:

- applies sigmoid to logits;
- selects a decision threshold that maximizes F1 on the evaluated loader;
- returns loss, accuracy, precision, recall, F1, AUC, confusion matrix, labels,
  probabilities and selected threshold.

The validation threshold is therefore fitted on each held-out validation set.
The latest training script also selects the checkpoint using held-out F1. For
strict external-test interpretation, the held-out project is serving as both
model-selection validation data and reported evaluation data. This is the actual
pipeline behavior and should be described explicitly in the release protocol.

### 3.9 Leave-one-project-out training and validation

The latest LOO function is `Model/auto_cross_ct_val.py::auto_cross_val`.

For each cancer project:

1. Load all generated samples from one directory.
2. Infer scale shapes from the first sample.
3. Define the current cancer type as validation.
4. Define all other cancer types as training.
5. Train a new `ComplexMultiStreamCNN` from scratch.
6. Use Adam with learning rate 1e-5 and weight decay 3e-2.
7. Use `ReduceLROnPlateau` on validation loss.
8. Evaluate train and validation metrics after each epoch.
9. Select the checkpoint with maximum validation F1; break F1 ties with AUC.
10. Save per-epoch metrics, validation filenames, timing/resource data,
    checkpoints, ROC data and plots.
11. Write one summary row per held-out cancer type.

This is true leave-one-project-out validation: all samples whose filename starts
with the held-out project prefix are excluded from training together.

## 4. The latest LOO script versus runnable historical alternatives

### Option A — latest feature set: `Model/auto_cross_ct_val.py`

Recommended as the release code base because it has:

- F1-first checkpoint selection with AUC tie-break;
- CPU/GPU resource and timing logs;
- last-K checkpoint retention;
- export of best checkpoints by F1, AUC, validation loss and average score;
- optional multi-model t-SNE output.

Current checked-in defaults that are experiment/debug state rather than a full
release run:

- `num_epochs = 1`;
- `cancer_list = ['LIHC']`;
- `mut_type = 'amp'`;
- one fixed seed;
- amp data points to `GeneratedSamples_amp_compressed_baseline_improved_0908`;
- the post-training t-SNE points to a different `_5120` sample directory;
- five-point smoothed plots assume a multi-epoch history.

The `auto_cross_val` function itself supports all cancer types when
`target_cancer_types=None`; the limitations above are in current defaults and
the main block.

### Option B — last clearly operational 50-epoch LOO: `Model/rec/auto_cross_ct_val_03.py`

This is the strongest historical fallback if the goal is to reproduce a prior
full run without adopting the newer exporter:

- 50 epochs, batch size 8, learning rate 1e-5;
- same LOO split and core model;
- validation F1 checkpoint selection;
- last-30 checkpoint analysis and t-SNE;
- GPU telemetry through optional NVML;
- points to the 0908 amp dataset.

It predates the current resource logger and multi-criterion best-model exporter.

### Option C — older AUC-selected baseline

`Model/auto_cross_ct_val_before0913.py` and
`Model/rec/auto_cross_ct_val_baseline.py` use the same LOO organization but select
the best epoch by validation AUC rather than validation F1. They are useful if
published or archived results were selected by AUC.

### Option D — domain/label-balanced sampling

`Model/legacy/auto_cross_ct_val_shin0905.py` with
`Model/legacy/Dataset_neo0905.py` is a genuinely different training method. It
uses a `DomainLabelBatchSampler` to balance project/domain and class composition,
and the dataset class can downsample tall matrices and return metadata. This is
not the current release path, but it should be preserved as a named experimental
alternative rather than discarded as a duplicate.

### Not LOO: current pan-cancer k-fold scripts

`Model/theta.py` and `Model/kFold_panCancer.py` perform stratified five-fold
splitting over the combined sample pool. They do not hold out an entire cancer
project, so they are not substitutes for the requested LOO validation pipeline.

## 5. Recommended real-cohort release assemblies

Because labeling provenance is ambiguous, two defensible assemblies exist.

### Assembly 1 — reproduce the known 0908 amp experiment

```text
amp matrices
  -> preprocess/compress.py (amp)
  -> type_centers_index_0908.yaml
  -> gen_sample_amp.py with an amp config matching the archived experiment
  -> GeneratedSamples_amp_compressed_baseline_improved_0908
  -> Model/utils/Dataset.py
  -> Model/Models.py + Model/utils/{train,evaluate}.py
  -> Model/rec/auto_cross_ct_val_03.py for historical reproduction
     OR auto_cross_ct_val.py for the latest F1/AUC-tiebreak implementation
```

Why this is defensible: the 0908 dataset directory is referenced consistently by
the LOO training, plotting and investigation scripts.

### Assembly 2 — clean modular release from curated coordinates

```text
raw merged matrices
  -> amp or del bin construction
  -> compress.py
  -> reviewed center_position.yaml (separate amp and del label files)
  -> centers_coord_to_index.py with debug metadata
  -> gen_sample_amp.py or gen_sample_del.py with explicit environment paths
  -> TSVDataset
  -> Models.py + train.py + evaluate.py
  -> auto_cross_ct_val.py::auto_cross_val(target_cancer_types=None)
```

Why this is preferable for a public release: every stage has an external input,
center coordinates remain auditable, and experiment-specific embedded label
dictionaries are removed from the conceptual pipeline.

### Deletion-specific limitation

The external `type_centers_del.yaml` is simulation-only (`RUBIC5`).
The repository does not contain an obviously final, external, 20-project real
deletion center YAML. Real deletion labels are distributed among older embedded
generator dictionaries and generated artifacts. Therefore, a scientifically
complete del release requires you to identify or reconstruct the intended real
deletion center list. This is the main unresolved provenance decision.

## 6. Minimal release manifest to preserve

For every training run, preserve these items together:

```text
mutation type: amp or del
raw matrix builder and its commit/hash
compress.py settings and K
center source file
coordinate-to-index output plus debug mapping
generator config YAML
generated sample directory
observed scale shapes
Dataset.py and Models.py versions
train/evaluate helper versions
LOO script and checkpoint-selection metric
seed, epochs, batch size, optimizer and scheduler
held-out project list
per-fold validation filenames
best-model checkpoint and selected threshold
```

Without the center file, generator configuration and sample directory, a model
checkpoint cannot unambiguously identify which historical pipeline created it.

## 7. Verification performed during this reconstruction

- Both C++ bin builders pass `g++ -std=c++17 -fsyntax-only`.
- All recommended Python files parse successfully as Python syntax.
- The YAML center-set counts above were read structurally rather than estimated
  from text.
- A direct diff shows `Model/theta.py` and `Model/kFold_panCancer.py` differ in
  their amp/del main configuration, while their k-fold training method is the
  same.
- The workspace contains 83 flat deletion TSVs, all for UCEC (60 positive, 23
  negative), and 83 RUBIC1 amp simulation TSVs (45 positive, 38 negative).
- The named top-level real amp sample directories are present but contain no flat
  TSV files in this checkout, so a full LOO training run cannot be reproduced
  from the local artifacts alone.
- Runtime parsing/training was not executed because this environment lacks the
  project's Python dependencies (`pandas` was the first missing import). No
  dependencies were installed because the request was analysis/reporting only.

## 8. Important decisions for the project owner

These cannot be resolved reliably from the repository alone:

1. **Canonical amp center set** — 0908 (largest/known-used), 0911improved
   (later/stricter), `L`, or regenerated from `center_position.yaml`.
2. **Canonical real del center set** — the current external del YAML is
   simulation-only; an intended real-cohort source must be selected.
3. **Published scale definitions** — amp 50/100/2000 versus current 50/250/2000;
   del 20/50/400 versus historical generated 50/100/500.
4. **Checkpoint criterion** — newer validation F1 with AUC tie-break versus older
   validation AUC.
5. **Validation interpretation** — whether held-out-project threshold/model
   selection is acceptable, or whether threshold selection needs a separate
   inner validation split.
6. **Arm-level scope** — the inspected amp preprocessing removes arm-level
   segments, although the research framing includes focal and broad events.

## 9. Code evidence index

### Preprocessing

- `preprocess/Version0209/sort.py:5-119` — arm classification, per-chromosome
  output and current subprocess wiring.
- `preprocess/Version0209/gen_bin_0306_cpp.cpp:90-177` — amp/general focal bin
  construction.
- `preprocess/Version0209/gen_bin_0306_cpp.cpp:180-238` — amp matrix output
  schema and executable entrypoint.
- `preprocess/Version0209/gen_bin_del.cpp:7-22` — deletion design rationale.
- `preprocess/Version0209/gen_bin_del.cpp:104-150` — deletion endpoints,
  baseline, subtraction and output.
- `preprocess/compress.py:6-25` — mutation-specific roots and K=40.
- `preprocess/compress.py:42-92` — shared 40-channel reduction.

### Label selection

- `preprocess/Version0209/label.py:7-57` — GISTIC/RUBIC containment and
  intersection labeling.
- `preprocess/Version0209/label.py:60-86` — hard-coded BRCA amp entrypoint.
- `preprocess/Version0209/gistic_after.py:1-39` — GISTIC interval extraction.
- `gen_dataset/centers_coord_to_index.py:1-24` — converter contract.
- `gen_dataset/centers_coord_to_index.py:162-228` — coordinate-to-index logic.
- `gen_dataset/centers_coord_to_index.py:235-269` — reusable CLI.
- `gen_dataset/TC_convert.py:39-141` — newer deduplication and Python center
  loading.

### Sample generation

- `gen_dataset/config.yaml:4-27` — latest amp generator configuration.
- `gen_dataset/gen_sample_amp.py:28-101` — modular config loading.
- `gen_dataset/gen_sample_amp.py:313-516` — indexed center selection and sample
  generation.
- `gen_dataset/config_del.yaml:4-25` — latest del generator configuration.
- `gen_dataset/gen_sample_del.py:36-103` — modular del configuration.
- `gen_dataset/gen_sample_del.py:272-314` — Mb/index center resolution.
- `gen_dataset/gen_sample_del.py:316-416` — del sample generation.

### Dataset, model, training and validation

- `Model/utils/Dataset.py:8-104` — discovery, filtering and item return.
- `Model/utils/Dataset.py:106-163` — labeled multi-scale parser.
- `Model/Models.py:106-191` — current multi-stream model and attention fusion.
- `Model/utils/train.py:10-43` — mixed-precision epoch training and gradient
  clipping.
- `Model/utils/evaluate.py:27-86` — probability, threshold and validation metrics.
- `Model/auto_cross_ct_val.py:363-392` — dataset discovery and held-out project
  list.
- `Model/auto_cross_ct_val.py:437-463` — project-level LOO split and optimizer.
- `Model/auto_cross_ct_val.py:469-575` — F1/AUC checkpoint selection.
- `Model/auto_cross_ct_val.py:597-675` — metrics, best checkpoint, ROC and summary.
- `Model/auto_cross_ct_val.py:785-835` — current experiment/debug main settings.
- `Model/rec/auto_cross_ct_val_03.py:167-417` — operational 50-epoch historical
  LOO alternative.

## 10. Bottom line

The latest coherent release architecture is modular and consists of the current
amp/del bin builders, `compress.py`, an explicitly chosen real-cohort center set,
the current amp/del sample generators, `TSVDataset`, `ComplexMultiStreamCNN`, the
current train/evaluate helpers, and `auto_cross_val` for project-level LOO.

The code implementation is recoverable. The uncertain part is scientific data
provenance: the canonical real label files and scale configurations are not
unambiguously marked, and the newest defaults now point to simulations. Once you
choose the intended amp and del center sets and scale triplets, the rest of the
release pipeline can be pinned with high confidence.
