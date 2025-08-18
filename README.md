# vulture

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-checkpoints-orange)](https://huggingface.co/rmdocherty/vulture)

Convolutional upsampling of DINOv2 [1] features for weakly supervised segmentation [2].
In short, we train a bisected U-net to upsample low-resolution features by targeting high-resolution ground truths generated from other methods (_i.e,_ FeatUp [3], LoftUp [4]) which may not scale as nicely in time / memory / generalisability as CNNs.
They can they be used in Weka-style [5] feature-based / interactive / weakly supervised segmentation.
Check out the [examples](examples/) to get started!

## TODO:

- zenodo for fig data

## Contents

- [Installation](#installation)
- [Checkpoints](#checkpoints)
- [Project Structure](#projectstructure)
- [Citation](#citation)
- [Contact](#contact)
- [References](#references)

## Installation

Note: you'll need `nvcc` installed to install flash-attn. See [`install/INSTALL_NVCC.md`](install/INSTALL_NVCC.md).

### Ubuntu / Mac

Either

```bash
conda env create -f install/conda.yml
conda activate vulture
pip install . --no-deps
# Force MAX_JOBS to avoid FA hogging all the cores; --no-build-isolation s.t it can find CUDA & nvcc
MAX_JOBS=4 pip install --no-build-isolation flash-attn
```

or

```bash
python -m venv .venv
source .venv/bin/activate
pip install .
MAX_JOBS=4 pip install --no-build-isolation flash-attn
python apply.py
```

or

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv sync
# update .env if we need to change CUDA_HOME / LD_LIBRARY_PATH later
uv run --env-file install/.env -- pip install --no-build-isolation flash-attn
uv run apply.py
```

The conda path comes with all the 'dev' dependencies (needed to reproduce the figures), if you want those with pip/uv/etc, run

```bash
pip install '.[dev]'
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

### Windows

In an 'Anaconda Powershell Prompt' (search in start menu)

```powershell
conda env create -f install\conda.yml
conda activate vulture
```

Note: flash-attn doesn't build/[requires extra steps](https://github.com/Dao-AILab/flash-attention/issues/595) to build on windows.

## Checkpoints

Checkpoints are available from [huggingface](https://huggingface.co/rmdocherty/vulture/tree/main), either download them into the `trained_models/` directory or run

```bash
chmod +x install/download_chkpoints.sh
./install/download_chkpoints.sh
```

Windows:

```powershell
.\install\download_chkpoints.ps1
```

## Project structure

```bash
examples/ # example notebooks for usage
│  └─ ...
paper_figures/ # notebooks to generate the paper figures
│  └─ fig_data/ # data needed for the notebooks, downloaded from zenodo
│  └─ ...
trained_models/ # model checkpoints (weights and model configs inside)
│  └─ fit_reg_f32.pth # downloaded with `install/download_chkpoints.sh`
│  └─ ...
install/
│  └─ conda.yml # conda env file
│  └─ download_chkpoints.sh # get checkpoints from gdrive
vulture/
├─ comparisons/ # wrapper code for other upsamplers / segmentation models
│  └─ ...
├─ datasets/
│  └─ lr_hr_embedding_dataset.py
├─ models/
│  ├─ configs/ # JSONs for training run parameters
│  ├─ external/ # external models used
│  │  ├─ autoencoder.py # compresses low-res DINOv2 features
│  │  ├─ online_denoiser.py # 'denoises' low-res ViT features - from [6]
│  │  └─ vit_wrapper.py # wrapper around DINOv2 for low-res features
│  ├─ layers.py # (u-net) layer components for our down-/upsampler
│  └─ model.py # down-/upsampler architecture
├─ train/ # training script
│  └─ train_upsampler.py
├─ main.py # E2E 'CompleteUpsampler' class + helper functions
├─ feature_prep.py # FeatUp style feature preprocessing (PCA)
└─ utils.py # plotting etc
```

## Citation

## Contact

## References

- [1] M. Oquab _et al._, "DINOv2: Learning Robust Visual Features without Supervision" (2023), ICLR, https://arxiv.org/abs/2304.07193
- [2] R. Docherty _et al._, "Upsampling DINOv2 features for unsupervised vision tasks and weakly supervised materials segmentation" (2024), Neurips AI4Mat workshop, https://arxiv.org/abs/2410.19836
- [3] S. Fu _et al._, "FeatUp: A Model-Agnostic Framework for Features at Any Resolution" (2024), ICLR, https://arxiv.org/abs/2403.10516
- [4] H. Huang _et al._, "LoftUp: A Coordinate-Based Feature Upsampler for Vision Foundation Models", ICCV, https://arxiv.org/abs/2504.14032
- [5] I. Arganda-Carreras _et al._, " Trainable Weka Segmentation: a machine learning tool for microscopy pixel classification", Bioinformatics (2017), https://academic.oup.com/bioinformatics/article/33/15/2424/3092362
- [6] J. Yang _et al._, "Denoising Vision Transformers" (2024), ECCV, https://arxiv.org/abs/2401.02957
