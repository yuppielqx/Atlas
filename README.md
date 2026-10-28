# Atlas experiment code

This repository contains the Atlas federated forecasting implementation and the data preparation scripts used by its experiments. Datasets and trained checkpoints are not included.

## Environment

Use Python with PyTorch and install the listed dependencies:

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended for the full-client experiments. The examples below use one GPU; increase `--n_gpus` when multiple CUDA devices are available. The main entry point is `main.py`; run `python main.py --help` for all options.

## Data layout

Pass the parent data directory with `--data_dir` (default: `./datasets`). Each sensor or station becomes one client. The loader expects chronologically ordered BasicTS-style files:

```text
datasets/
  METR-LA/
    data.dat
    desc.json
  PEMS-BAY/
    data.dat
    desc.json
  PEMS03/
    data.dat
    desc.json
  PEMS04/
    data.dat
    desc.json
  KnowAir-PM25/
    data.dat
    desc.json
  KnowAir-Full/
    data.dat
    desc.json
```

For METR-LA, PEMS-BAY, PEMS03, PEMS04, and KnowAir-PM25, `data.dat` is a `float32` memmap with shape `[time, node, 3]`: target value, time of day, and day of week. KnowAir-Full uses `[time, node, 18]` (17 weather channels plus PM2.5). Each `desc.json` must contain `shape` and `regular_settings.TRAIN_VAL_TEST_RATIO`; the loader uses those ratios for chronological train/validation/test splits. An optional `adj_mx.pkl` is read only for graph-context encoder runs.

If you have the original `KnowAir.npy.zip`, create either KnowAir format with:

```bash
python scripts/prepare_knowair_pm25.py --source datasets/KnowAir.npy.zip
python scripts/prepare_knowair_full.py --source datasets/KnowAir.npy.zip
```

The traffic datasets must be supplied in the format above. The repository does not contain a traffic-data downloader or conversion script.

## Atlas configuration

The following are the **current defaults in `main.py`**, not a claim that every reported experiment used identical settings:

| Setting | Default |
| --- | --- |
| Federated method | `--fl_mode causal_proto` |
| Prototype alignment | `--proto_align hungarian` |
| Local encoder | `--encoder_type stid`, `--stid_num_layers 3` |
| Prototypes | `--K 10`, `--tau 0.1` |
| Representation | `--hid_dim 64` |
| Input / forecast window | `--window_size 12`, `--horizon 12` |
| Training | `--n_rounds 100`, `--local_epochs 1`, `--batch_size 256`, `--lr 1e-4` |
| Selection and stopping | `--fraction 1.0`, `--patience 8` |
| Reproducibility | `--seed 42` |
| Residual path | `--use_residual true`, `--residual_mode fixed`, `--residual_scale 1.0` |
| Local training implementation | `--vectorized true` |

The default learning-rate scheduler is `plateau` with factor `0.5`, patience `3`, and minimum learning rate `1e-6`. `--mi_weight` and `--aux_inv_weight` default to zero.

## Run experiments

Run Atlas with all clients on one GPU:

```bash
python main.py --dataset metr-la --data_dir ./datasets \
  --fl_mode causal_proto --proto_align hungarian \
  --device cuda:0 --n_gpus 1 --seed 42 --run_name metr_la_atlas
```

Use `--dataset pems-bay`, `pems03`, `pems04`, `knowair-pm25`, or `knowair-full` for the other datasets. For example:

```bash
python main.py --dataset pems-bay --data_dir ./datasets \
  --device cuda:0 --n_gpus 1 --seed 42 --run_name pems_bay_atlas
```

For multiple GPUs, set `--n_gpus` to the number of visible CUDA devices. For partial client participation, disable the vectorized path; it trains every client each round and ignores `--fraction`:

```bash
python main.py --dataset metr-la --data_dir ./datasets \
  --fraction 0.25 --vectorized false \
  --device cuda:0 --n_gpus 1 --seed 42 --run_name metr_la_atlas_frac25
```

`--n_clients` limits the fixed client pool for the whole run; `--fraction` samples from that pool independently each round. For an exact subset, use `--client_ids_file` with comma-, space-, or newline-separated integer node IDs.

## Outputs

Every run writes to `<output_dir>/<run_name>/` (`--output_dir` defaults to `./output`). The main program evaluates the test set after training and saves `args.json` (the full command and configuration), `metrics.csv`, `results.json` (with `metrics.MAE` and `metrics.RMSE`), and `checkpoint.pt`. It also saves a training-curve figure. Use each run's `args.json` and checkpoint when reproducing a specific reported result.

For new experiments, read test metrics directly from `results.json`. The older `test.py` and `reeval.py` utilities are not required for the training workflow documented here.
