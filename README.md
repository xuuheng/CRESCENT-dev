# CRESCENT-dev

Development and training archive for `CRESCENT`, a deep-learning framework for detecting recurrent copy-number amplifications (`amp`) and deletions (`del`). This repository collects the latest coherent preprocessing, labeled-sample generation, dataset loading, model, and leave-one-project-out validation code found in the original development tree.

If you are primarily interested in using **CRESCENT** for inference, please refer to [**CRESCENT**](https://github.com/BioThinkLab/CRESCENT), which provides a streamlined and performance-optimized implementation with substantially lower RAM and storage requirements, making it more suitable for use on personal computers.

This repository contains the complete research codebase, including the training, validation, and evaluation pipelines. In addition to the final implementation, some modules also retain experimental variants explored during development. These are included for completeness and reproducibility of the research process, and may not represent the recommended or best-performing configurations.

## Included

- `Data/merged_dataframe/`: merged CNA segment tables for 20 TCGA cancer projects.
- `Data/ChromosomeData2025/`: GRCh38 chromosome-arm ranges required by the amplification preprocessing path.
- `preprocess/`: focal amplification/deletion bin construction, 40-column compression, and the alternative automatic label selector.
- `gen_dataset/`: `config.yaml`, `config_del.yaml`, current amp/del generators, coordinate conversion, and all plausible center-label manifests retained for provenance.
- `Model/`: the shared multi-scale CNN/attention model, TSV dataset class, training/evaluation helpers, and leave-one-project-out validation driver.
- `docs/`: detailed pipeline reconstruction and amp/del comparison reports.

Generated bin matrices, generated training samples, checkpoints, plots, and experiment outputs are intentionally excluded.

## Pipeline

```text
Data/merged_dataframe
  -> mutation-specific chromosome/bin preprocessing
  -> 40-column compression
  -> center labels + multi-scale sample generation
  -> Model/utils/Dataset.py
  -> Model/Models.py
  -> Model/auto_cross_ct_val.py (leave one cancer project out)
```

Amp sampling currently uses `50x40`, `250x40`, and `2000x40` windows. Del sampling uses `20x40`, `50x40`, and `400x40` windows.

## Label provenance

There is no single unambiguous final label file in the original tree, so candidates are preserved instead of discarded:

- `type_centers_index_0911improved.yaml`: best-supported late real-cohort amplification candidate.
- `type_centers_index_0908.yaml`, `type_centers_index_0912.yaml`, and `type_centers_index_L.yaml`: alternative real-cohort snapshots.
- `center_position.yaml`: coordinate-based 20-project labels; convert with `centers_coord_to_index.py` when needed.
- `type_centers_index.yaml` and `type_centers_index_del.yaml`: current generator defaults, but both contain `RUBIC5` simulation labels.
- `rubic_sim_a.yaml`: for stimulation dataset provided by `RUBIC`.

For a chosen label file, set `GENSAMPLES_TYPE_CENTERS` before running a generator. Set `GENSAMPLES_CONFIG` to select the amp or del configuration.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd gen_dataset
GENSAMPLES_CONFIG=./config.yaml \
GENSAMPLES_TYPE_CENTERS=./type_centers_index_0911improved.yaml \
python gen_sample_amp.py
```

Deletion generation is analogous with `config_del.yaml` and an explicitly selected deletion label manifest.

## Training and validation

Run training from `Model/` after generating the flat labeled TSV sample directory:

```bash
cd Model
python auto_cross_ct_val.py
```

Please note that `auto_cross_ct_val.py` runs with a target TCGA-project as the held-out dataset ,it contains configuration in the bottom block. If you want to run on a specific project or try different configuration, change configuration accordingly.


