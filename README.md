# ghost

Multi-view person segmentation, 3D body estimation, and temporal synchronization for the [EgoExo4D](https://ego-exo4d-data.org/) dataset.

## Overview

This project processes multi-camera scene recordings in three stages:

1. **Person segmentation** — Grounding DINO detects people in keyframes; SAM2 propagates masks across all frames within each video.
2. **Body parameter estimation** — SAM3D Body estimates per-frame SMPL body parameters (pose, shape, keypoints) for each tracked person.
3. **Video synchronization** — Pairwise DTW on 3D joint sequences recovers temporal offsets between cameras, solved globally via least squares.

## Project structure

```
ghost/
├── main.py                     # End-to-end pipeline entry point
├── data/
│   ├── video_dataset.py        # Lazy video / scene dataset (EgoExoSceneDataset)
│   ├── segmentation.py         # PersonSegmenter (GDINO + SAM2)
│   └── parameters_extraction.py # BodyParameterEstimator (SAM3D Body)
├── synchronize_videos/
│   └── synchronizer.py         # Temporal alignment via weighted DTW
├── utilities/                  # Offline helper scripts
├── bash_jobs/                  # SLURM job scripts
├── test/                       # Unit and integration tests
├── Grounded-SAM-2/             # Grounded SAM2 submodule
├── sam-3d-body/                # SAM3D Body submodule
├── MHR/                        # MHR / SMPL conversion tools
├── checkpoints/                # Model weights (not tracked)
└── body_models/                # SMPL body model files (not tracked)
```

## Installation

This project uses [pixi](https://pixi.sh) for environment management. The setup requires two steps because the login node has no GPU driver.

```bash
# 1. Install conda dependencies (works without a GPU)
CONDA_OVERRIDE_CUDA=12.6 pixi install

# 2. Replace the CPU-only PyTorch with a CUDA build
CONDA_OVERRIDE_CUDA=12.6 pixi run setup-cuda
```

On **GPU compute nodes** `CONDA_OVERRIDE_CUDA` is not needed:

```bash
pixi install
pixi run setup-cuda
```

> **Note:** always use `python -m pip` (not bare `pip`) inside pixi tasks to avoid the system pip shadowing the environment.

## Usage

```bash
pixi run python main.py \
    --data_root /path/to/egoexo/takes \
    --output_dir /path/to/output \
    [--slice N]              # process only the first N scenes \
    [--detection_step 50]    # run GDINO every N frames \
    [--sam3d_step 1]         # run SAM3D every N frames \
    [--smooth]               # temporal smoothing for body params \
    [--vis]                  # save annotated segmentation videos \
    [--device cuda]
```

### Expected input layout

```
data_root/
    scene_001/
        cam01.mp4
        cam02.mp4
        ...
    scene_002/
        ...
```

### Output layout

```
output_dir/
    <scene_id>/
        <video_id>/
            frames/                  # extracted JPEGs (can be deleted)
            mask_data.npz            # compressed per-frame masks (uint16)
            json_data/               # per-frame instance metadata
            body_data/
                person_<id>.npz      # per-person body parameters
                body_params_summary.json
            segmentation.mp4         # (optional) visualisation video
        cross_video_id_mapping.json
```

## Running on the cluster (SLURM)

A reference SLURM script is provided:

```bash
sbatch bash_jobs/test_run_sam3d.sh
```

Logs are written to `logs/<job_name>_<job_id>.{out,err}`.

## Key design decisions

- **Multi-GPU parallelism**: segmentation distributes videos across all available GPUs using `torch.multiprocessing.Pool`; each worker loads its own model instances.
- **Incremental processing**: already-segmented videos are skipped automatically (detected by the presence of `mask_data.npz`).
- **Mask storage**: per-frame `.npy` files are merged into a single `.npz` after segmentation (typically 20–50× compression).
- **Synchronization**: pairwise DTW offsets between all camera pairs are combined in a global least-squares solve, giving robust start times even with missing pairs.

## Checkpoints

Download model weights and place them in `checkpoints/`:

| Model | Source |
|---|---|
| SAM 2.1 Hiera-L | [Meta / HuggingFace](https://huggingface.co/facebook/sam2.1-hiera-large) |
| Grounding DINO | `IDEA-Research/grounding-dino-tiny` (auto-downloaded via HuggingFace) |
| SAM3D Body | `facebook/sam-3d-body-dinov3` (auto-downloaded via HuggingFace) |

SMPL body model files should be placed in `body_models/`.


Removed the dependency <2 in sam3, also added a pyproject.toml to sam3d


Run the download_model task in pixi with pixi run download-model