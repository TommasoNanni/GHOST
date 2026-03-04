# GHOST - Geometric Human Out-of-sync Spatio Temporal reconstruction

## Overview

This repository contains the code for GHOST, a novel approach that leverages state-of-the-art models, time warping algorithms and a transformer based geometric fusion in order to reconstruct human meshes from casual, unsynchronized and multi-view videos. 

GHOST receives as input videos containing humans and employs [SAM3](https://ai.meta.com/research/sam3/) and [SAM 3D Body](https://ai.meta.com/research/sam3d/) to detect people and extract per-view [SMPL-X](https://smpl-x.is.tue.mpg.de/) parameters. Such human parameters are then used to temporally align videos and extract initial relative camera poses. 

Such initial parameters are then processed by the fusion module, which aligns shape, pose and camera parameters into a unique world representation. Such a representation is obtained exclusively through geometric constraints and confidence masks.
## Quick start

### Installation
This code leverages several external repositories. For [SAM3](https://huggingface.co/facebook/sam3) and [SAM 3D Body](https://huggingface.co/facebook/sam-3d-body-dinov3) (make sure to download the DINOv3 version).

```bash
# From inside /ghost:
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/facebookresearch/sam-3d-body.git
git clone git@github.com:facebookresearch/MHR.git
cd MHR
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip

git clone https://github.com/nghorbani/human_body_prior.git
```
Then make sure you have [pixi](https://github.com/prefix-dev/pixi) installed, since the code uses pixi for installation.
```bash
pixi install
pixi run setup-cuda
pixi run download-model # download sam3
pip install --no-deps -e ./human_body_prior
```
In case you are on a devicee without GPU available (or on a login node on a cluster)
```bash
CONDA_OVERRIDE_CUDA=12.6 pixi install
CONDA_OVERRIDE_CUDA=12.6 pixi run setup-cuda
CONDA_OVERRIDE_CUDA=12.6 pixi run download-model
pip install --no-deps -e ./human_body_prior
```
We built this project using python 3.12 and torch 2.7.1 with cuda 12.6 support.

Moreover, make sure to have SMPLX and SMPL body models installed in `ghost/body_models/`. Download the `SMPLX_NEUTRAL.pkl` and `SMPL_NEUTRAL.pkl` body models from [SMPL-X](https://smpl-x.is.tue.mpg.de/).

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
├── sam-3d-body/                # SAM3D Body submodule
├── MHR/                        # MHR / SMPL conversion tools
├── checkpoints/                # Model weights (not tracked)
└── body_models/                # SMPL body model files (not tracked)
```
### Repo modifications
Two key modifications have to be done to the external repositories:

1. In `sam3/pyproject.toml`, make sure to remove the `numpy<2` dependency. This causes conflicts with our new versions of pytorch and doesn't cause issues in SAM3 usage
2. SAM 3D Body doesn't have a `pyproject.toml` file. We created a minimal one that contains the dependencies needed to install it in our repo. Run:
    ```bash
    cd sam-3d-body
    cat <<EOF > pyproject.toml
    # PlaceHolder
    EOF
    ```

### Dataset
This repo has been trained using the [RICH dataset](https://rich.is.tue.mpg.de/). To replicate training using rich, follow the instructions on their website for downloading the dataset. The dataset downloads images in `.bmp` format, taking approximately 30 Mb each. To overcome the problem of such an extreme space usage, we provide a script that converts the images to `.jpg` format.

To run it, go in `bash_jobs/convert_rich_bmp_to_jpeg.sh` and change the `--root` argument to your rich directory. Then launch
```bash
bash_jobs/convert_rich_bmp_to_jpeg.sh
```
Notably, our pipeline resize images to approximately 1000 x 1000 resolution and saves them inside a `/frames` folder in the data directory. If higher resolution images are not needed, they can be deleted and this will save even more space.

Also make sure to go to `configuration/config.py` and update your data directories.

The expected dataset layouts are the ones that the scripts automatically download. 
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
