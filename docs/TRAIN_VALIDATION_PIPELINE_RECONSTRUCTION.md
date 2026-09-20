# CRESCENT training and validation pipeline

This document defines the single development pipeline packaged in
`CRESCENT-dev`. The training and validation entry point is exactly:

```text
Model/auto_cross_ct_val.py
```

The pipeline supports amplification (`amp`) and deletion (`del`). They use
different matrix preprocessing, labels, and sampling scales, then share the
same dataset class, neural network, training helpers, and leave-one-project-out
validation procedure.

## 1. Canonical pipeline

```text
Data/merged_dataframe/merged_dataframe_<CANCER>.tsv
    |
    +-- amp
    |     preprocess/Version0209/sort.py
    |     preprocess/Version0209/gen_bin_0306_cpp.cpp
    |     preprocess/compress.py
    |     gen_dataset/config.yaml
    |     gen_dataset/type_centers_index.yaml
    |     gen_dataset/gen_sample_amp.py
    |
    +-- del
          preprocess/Version0209/gen_bin_del.cpp
          preprocess/compress.py
          gen_dataset/config_del.yaml
          gen_dataset/type_centers_del.yaml
          gen_dataset/gen_sample_del.py
    |
    v
GeneratedSamples_<mutation>/
    |
    v
Model/utils/Dataset.py
    |
    v
Model/Models.py
Model/utils/train.py
Model/utils/evaluate.py
    |
    v
Model/auto_cross_ct_val.py
    |
    v
experimental/<mutation>/<session>/<held-out-project>/
```

## 2. Input data

The starting data is in `Data/merged_dataframe/`. There is one TSV for each of
20 TCGA cancer projects. Required columns are:

- `GDC_Aliquot`
- `Chromosome`
- `Start`
- `End`
- `Copy_Number`
- `Major_Copy_Number`
- `Minor_Copy_Number`

The amplification arm-classification step also uses
`Data/ChromosomeData2025/GRCh38_Chromosome_Arm_Ranges.tsv`.

## 3. Mutation-specific preprocessing

### 3.1 Amplification

1. `preprocess/Version0209/sort.py` calculates segment length relative to the
   chromosome arm, adds `is_arm_level`, and writes per-chromosome focal inputs.
2. Compile `preprocess/Version0209/gen_bin_0306_cpp.cpp` with C++17.
3. Run the compiled program for each cancer project. It builds adjacent genomic
   bins from retained segment endpoints and writes sample copy-number values.
4. Run `preprocess/compress.py` with `mut="amp"` to reduce or pad the patient
   axis to 40 columns.

The C++ tool accepts:

```text
gen_bin_amp <cancer_type> [project_root]
```

If `project_root` is omitted, it uses the current working directory.

### 3.2 Deletion

1. Compile `preprocess/Version0209/gen_bin_del.cpp` with C++17.
2. Run it for each cancer project.
3. The program uses only records with `Copy_Number < 2` to define bins.
4. Every sample starts at diploid value 2; a covering deletion subtracts
   `2 - Copy_Number`.
5. Run `preprocess/compress.py` with `mut="del"` to reduce or pad the patient
   axis to 40 columns.

The deletion tool accepts:

```text
gen_bin_del <cancer_type> [project_root]
```

## 4. Labels and sample generation

The pipeline uses two label manifests:

- Amp: `gen_dataset/type_centers_index.yaml`
- Del: `gen_dataset/type_centers_del.yaml`

Each file maps cancer type and chromosome to positive and negative center
indices. The generators convert those centers into labeled multi-scale TSV
samples:

- Amp: `gen_dataset/gen_sample_amp.py` with `gen_dataset/config.yaml`
- Del: `gen_dataset/gen_sample_del.py` with `gen_dataset/config_del.yaml`

Configured shapes are:

| Mutation | Scale 1 | Scale 2 | Scale 3 |
|---|---:|---:|---:|
| Amp | 50 x 40 | 250 x 40 | 2000 x 40 |
| Del | 20 x 40 | 50 x 40 | 400 x 40 |

Run from `gen_dataset/` so the default relative paths resolve correctly:

```bash
cd gen_dataset

# Amplification
GENSAMPLES_CONFIG=./config.yaml \
GENSAMPLES_TYPE_CENTERS=./type_centers_index.yaml \
python gen_sample_amp.py

# Deletion
GENSAMPLES_CONFIG=./config_del.yaml \
GENSAMPLES_TYPE_CENTERS=./type_centers_del.yaml \
python gen_sample_del.py
```

Each generated file contains a binary label followed by three matrix blocks.
The filename starts with the cancer-project name, which is how leave-one-out
validation assigns samples to projects.

## 5. Dataset and model

`Model/utils/Dataset.py` provides `TSVDataset`. It discovers generated TSV
samples, reads their label and scale blocks, converts matrices to tensors, and
returns:

```text
([scale_tensor_1, scale_tensor_2, scale_tensor_3], label, filename)
```

`Model/Models.py` provides `ComplexMultiStreamCNN`:

- one convolutional/residual/ECA branch per scale;
- adaptive pooling and feature projection per branch;
- multi-head attention across scale features;
- residual feed-forward refinement;
- mean fusion and a one-logit binary classifier.

Both mutation types use this same model class. Their distinction is encoded in
the preprocessed matrices, center labels, scale shapes, sample directory, and
trained checkpoint.

## 6. Training and leave-one-project-out validation

`Model/auto_cross_ct_val.py` is the only documented training and validation
script. Its `auto_cross_val()` function performs the following for every target
cancer project:

1. Load every generated sample with `TSVDataset`.
2. Infer branch shapes from the first sample.
3. Hold out all samples whose filename starts with the target project.
4. Train on samples from all other projects.
5. Optimize `ComplexMultiStreamCNN` with `BCEWithLogitsLoss`, Adam, and
   `ReduceLROnPlateau`.
6. Evaluate training and held-out data after each epoch.
7. Select the checkpoint with the highest validation F1; use AUC to break ties.
8. Save metrics, validation filenames, resource timings, checkpoints, ROC data,
   plots, and one summary row for the held-out project.

`Model/utils/train.py` implements one training epoch. `Model/utils/evaluate.py`
computes loss, accuracy, precision, recall, F1, AUC, confusion matrix, and the
validation F1 threshold.

### Configuration before a run

Edit the bottom block of `Model/auto_cross_ct_val.py` and confirm:

- `mut_type` is `amp` or `del`;
- `data_dir` points to the generated TSV directory;
- `cancer_list` contains the projects to hold out;
- `seeds` contains the required runs;
- `num_epochs`, `batch_size`, and `learning_rate` inside `auto_cross_val()` are
  set for the experiment;
- the t-SNE `data_dir` matches the training sample directory.

Run:

```bash
cd Model
python auto_cross_ct_val.py
```

Outputs are written below `experimental/<mutation>/`.

## 7. Expected output structure

```text
experimental/<mutation>/<timestamp>_<seed>/
    hyperparameters.txt
    summary_metrics.tsv
    <held-out-project>/
        epoch_metrics.tsv
        timings.tsv
        best_model.pth
        best_model_selection.json
        roc.tsv
        roc_val.png
        loss_curve_raw.png
        accuracy_curve_raw.png
        val_filenames_epoch_<N>.txt
```

## 8. Validation checks

Before a full run:

- confirm generated samples exist for every requested cancer project;
- inspect one amp and one del TSV with `TSVDataset.parse_file()`;
- confirm the three parsed shapes match the selected configuration;
- confirm labels contain both 0 and 1;
- ensure every held-out project leaves non-empty training and validation sets;
- run a short one-project job before launching the full project list.

## 9. Other development versions

The original development tree contained additional experiments. They are not
part of the canonical pipeline above:

- `Model/rec/auto_cross_ct_val_03.py`: an earlier 50-epoch leave-one-project-out
  run configuration.
- `Model/auto_cross_ct_val_before0913.py` and
  `Model/legacy/auto_cross_ct_val_baseline.py`: older checkpoint selection based
  primarily on validation AUC.
- `Model/legacy/auto_cross_ct_val_shin0905.py` with
  `Model/legacy/Dataset_neo0905.py`: domain/label-balanced batch sampling.
- `Model/theta.py` and `Model/kFold_panCancer.py`: stratified pan-cancer k-fold
  experiments, not leave-one-project-out validation.
- `gen_dataset/gen_sample_del_old.py`: deletion samples with the historical
  50/100/500 scale family.
- Dated `type_centers_index_*` files: historical label snapshots removed from
  this archive after consolidating labels into the two canonical manifests.
- `gen_dataset/center_position.yaml` with `centers_coord_to_index.py`:
  coordinate-based label conversion retained for provenance.
- `gen_dataset/rubic_sim_a.yaml`: auxiliary RUBIC simulation labels.

Use these names only when tracing old experiment outputs. New development runs
should follow Sections 1-8 and use `Model/auto_cross_ct_val.py`.
