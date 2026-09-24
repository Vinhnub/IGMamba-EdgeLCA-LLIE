# IGMamba-EdgeLCA-LLIE: Illumination-Guided Mamba with Edge-Modulated Lightweight Cross-Attention for Low-Light Image Enhancement

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced framework for **Low-Light Image Enhancement (LLIE)** that integrates **Illumination-Guided State Space Models (IG_Mamba)** with **Edge-Modulated Lightweight Cross-Attention (Edge_LCA)** under a color-intensity decoupled learning scheme.

The core primary architecture of this repository is: **Separable Learning with Edge Guidance (`net/CIDNet_Mamba_separable_learning_edge.py`)**.

---

## 📌 Table of Contents
- [1. Architecture & Key Contributions](#1-architecture--key-contributions)
- [2. Project Directory Structure](#2-project-directory-structure)
- [3. Environment Setup & Installation](#3-environment-setup--installation)
- [4. Dataset Preparation](#4-dataset-preparation)
- [5. Training Guide](#5-training-guide)
- [6. Testing & Evaluation Guide](#6-testing--evaluation-guide)
- [7. Analysis & Utility Tools](#7-analysis--utility-tools)
- [8. Citation](#8-citation)

---

## 1. Architecture & Key Contributions

Low-light images typically suffer from low visibility, non-uniform illumination, noise amplification in underexposed regions, and color degradation. `IGMamba-EdgeLCA-LLIE` tackles these challenges through an end-to-end decoupled network:

```
                              ┌───────────────────────┐
                              │  Input Image (RGB)    │
                              └──────────┬────────────┘
                                         │ HVI Transform
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
          ┌─────────────────────────┐         ┌─────────────────────────┐
          │  Intensity Channel (I)  │         │ Chrominance (H, V)      │
          └────────────┬────────────┘         └────────────┬────────────┘
                       │                                   │
             ┌─────────▼─────────┐                         │
             │   EdgeExtractor   │                         │
             │ (Sobel+Laplacian) │                         │
             └─────────┬─────────┘                         │
                       │ Edge Map                          │
                       ▼                                   │
          ┌─────────────────────────┐                      │
          │    Edge_Guided_I_LCA    │                      │
          │ (Cross-Attn + Gating)   ├──────────────┐       │
          └────────────┬────────────┘              │ Guide │
                       │ Refined I                 ▼       ▼
                       │                    ┌─────────────────────────┐
                       │                    │   IG_Mamba Modulation   │
                       │                    │  (State Space Model)    │
                       │                    └──────────────┬──────────┘
                       │                                   │
                       └─────────────────┬─────────────────┘
                                         ▼ Inverse HVI (PHVIT)
                              ┌───────────────────────┐
                              │ Enhanced Image (RGB)  │
                              └───────────────────────┘
```

### Key Technical Highlights:
1. **Color-Intensity Decoupling (HVI Space):** Transforms raw RGB input into the HVI (Hue, Saturation-Value, Intensity) color space to decouple illuminance adjustment from chrominance restoration.
2. **Structural Edge Extractor (`EdgeExtractor`):** Integrates Gaussian pre-filtering (noise suppression) with multi-directional Sobel ($X, Y$) and Laplacian operators, fused via a learnable $1 \times 1$ convolution and converted into a continuous priority map via Sigmoid mapping.
3. **Edge-Guided Cross Attention (`Edge_Guided_I_LCA`):**
   - Transposed cross-attention between Intensity (Query) and Chrominance (Key, Value) features.
   - Spatial feature modulation driven by the edge map ($1.0 + \text{edge\_scale} \cdot \text{edge}$), sharpening object boundaries and fine textures.
   - Spatial Edge Gating at the output layer to suppress noisy artifacts in flat, dark background areas.
4. **Illumination-Guided Mamba (`IG_Mamba`):**
   - Uses the refined Intensity representation to dynamically modulate SSM parameters ($\Delta$, $B$, $C$).
   - Achieves global receptive field modeling with linear $\mathcal{O}(N)$ computational complexity.
5. **Dark Focus Mechanism & Multi-Objective Loss:**
   - Dedicated `--dark_focus True` flag prioritizing optimization on severely underexposed regions.
   - Combined objective: $\mathcal{L}_1$ + $\mathcal{L}_{SSIM}$ + $\mathcal{L}_{VGG}$ (Perceptual) + $\mathcal{L}_{Edge}$ + $\mathcal{L}_{LSGD}$ (*Dark-Focused Spatial Gradient Loss*) + Exposure Control Loss.
6. **Dual-Engine SSM Compatibility:**
   - Automatically utilizes high-performance CUDA kernels via `mamba_ssm` on Linux / server environments.
   - Seamlessly falls back to a vectorized Pure-PyTorch implementation on platforms without compiled CUDA kernels (e.g., Windows), ensuring zero crashes.

---

## 2. Project Directory Structure

```
IGMamba-EdgeLCA-LLIE/
├── data/                       # Dataloaders, learning rate schedulers, CLI options
│   ├── data.py                 # Dataset loader implementations (LOL, SID, SICE, FiveK...)
│   ├── options.py              # Central argument parsing configuration
│   └── scheduler.py            # Cosine annealing, cyclical restart, and warmup schedulers
├── dataset/                    # Benchmark dataset storage
│   ├── LOL/                    # LOLv1, LOLv2-real, LOLv2-synthetic datasets
│   └── UnPair/                 # Unpaired real-world datasets (DICM, LIME, MEF, NPE, VV...)
├── loss/                       # Loss functions and perceptual networks
│   ├── losses.py               # L1, L2, SSIM, VGG, Edge, LSGD (Dark Focus), Exposure losses
│   └── vgg_arch.py             # Feature extractor backbone using VGG19
├── net/                        # Neural network architectures
│   ├── CIDNet_Mamba_separable_learning_edge.py  # [MAIN] Separable learning + Edge + Mamba
│   ├── CIDNet_Mamba_separable_learning.py       # Separable learning with Mamba
│   ├── CIDNet_base.py          # Baseline CIDNet model
│   ├── Edge_LCA.py             # Edge-Guided Lightweight Cross Attention module
│   ├── edge_filter.py          # EdgeExtractor module (Gaussian + Sobel + Laplacian)
│   ├── HVI_transform.py        # Color space conversion (RGB <-> HVI, RGB <-> YCbCr)
│   └── IG_Mamba.py             # Illumination-Guided Mamba (CUDA & PyTorch dual engine)
├── output/                     # Enhanced images generated by eval.py
├── results/                    # Validation sample snapshots saved during training
├── weights/                    # Checkpoint storage (.pth files)
│   └── train/                  # Saved training checkpoints (epoch_*.pth)
├── eval.py                     # Script for image generation / inference
├── measure.py                  # Full-reference evaluation (PSNR, SSIM, LPIPS)
├── measure_niqe_bris.py        # No-reference evaluation (NIQE, BRISQUE)
├── net_test.py                 # Complexity profiling (Parameters, FLOPs, Latency)
├── test_dark_focus_sync.py     # Unit test suite verifying model & loss synchronization
├── test_ig_mamba.py            # Unit test suite for IG_Mamba correctness
├── visualize_ig_mamba.py       # Visualization tool for delta modulation & feature maps
├── train.py                    # Single-GPU training script
├── train_ddp.py                # Multi-GPU Distributed Data Parallel (DDP) training script
├── requirements.txt            # Python dependencies
└── Readme.md                   # Project documentation
```

---

## 3. Environment Setup & Installation

We recommend using **Python 3.10** with **Miniconda / Anaconda**.

### Step 1: Create and Activate Conda Environment

```bash
conda create -n igmamba python=3.10 -y
conda activate igmamba
```

### Step 2: Install PyTorch and Torchvision

Select the command corresponding to your system's CUDA version:

- **For CUDA 12.x / 12.8:**
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  ```
- **For CUDA 11.8:**
  ```bash
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
  ```
- **For CUDA 13.x:**
  ```bash
  pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
  ```

### Step 3: Install Mamba SSM Kernel (Optional Acceleration)

- **On Linux / Cloud Servers (A100, H100, RTX 3090/4090):**
  Install official pre-compiled wheels or compile from source:
  ```bash
  pip install causal-conv1d>=1.2.0
  pip install mamba-ssm>=2.0.0
  ```
  *(If you have a local `.whl` wheel file, run `pip install <wheel_name>.whl`)*.

- **On Windows:**
  The `net/IG_Mamba.py` module includes an automatic **Pure-PyTorch Fallback**. You **do not need** to compile C++/CUDA kernels on Windows; the model will automatically detect the environment and run reliably out of the box.

### Step 4: Install Required Packages

Install the remaining dependencies via:
```bash
pip install opencv-python lpips thop einops scipy scikit-image matplotlib tqdm pillow huggingface_hub
```
Or directly from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

Run the built-in test suites to verify that the environment, models, and losses are working properly:
```bash
python test_dark_focus_sync.py
python test_ig_mamba.py
```
If all tests return `[PASS]`, your setup is complete and ready.

---

## 4. Dataset Preparation

Organize the datasets inside the `dataset/` directory (or specify custom paths in [`data/options.py`](data/options.py)):

### 1. LOLv1 Dataset:
```
dataset/LOL/LOLv1/
├── train/
│   ├── low/            # Low-light input images
│   └── high/           # Normal-light ground truth images
└── test/
    ├── low/            # 15 low-light test images
    └── high/           # 15 ground truth images
```

### 2. LOLv2 (Real & Synthetic) Datasets:
```
dataset/LOL/LOLv2-real/
├── Train/
│   ├── Low/
│   └── Normal/
└── Test/
    ├── Input/          # 100 test input images
    └── GT/             # 100 ground truth images
```

### 3. Unpaired Real-World Datasets (DICM, LIME, MEF, NPE, VV):
```
dataset/UnPair/
├── DICM/
├── LIME/
├── MEF/
├── NPE/
└── VV/
```

---

## 5. Training Guide

The primary model of this repository is **`CIDNet_Mamba_separable_learning_edge`** (`net/CIDNet_Mamba_separable_learning_edge.py`).

> [!TIP]
> When training this model, ensure that the model import line in [`train.py`](train.py) and [`train_ddp.py`](train_ddp.py) is:
> ```python
> from net.CIDNet_Mamba_separable_learning_edge import CIDNet
> ```

### 5.1. Single-GPU Training

Train on the **LOLv1** dataset:
```bash
python train.py --dataset lol_v1 --cropSize 256 --batchSize 4 --lr 1e-4 --nEpochs 1000 --snapshots 5
```

Train on the **LOLv2-Real** dataset with Dark Focus and Loss Warmup enabled:
```bash
python train.py --dataset lolv2_real --cropSize 256 --batchSize 4 --lr 1e-4 --nEpochs 1000 --dark_focus True --number_warmup_epoch 5 --number_transition_epoch 10
```

### 5.2. Multi-GPU Distributed Data Parallel (DDP) Training

Use PyTorch's `torchrun` to scale training across multiple GPUs (e.g., 2 GPUs):
```bash
torchrun --nproc_per_node=2 train_ddp.py --dataset lol_v1 --batchSize 4 --cropSize 256 --lr 2e-4 --nEpochs 1000
```

### 5.3. Key Training Arguments ([`data/options.py`](data/options.py))

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--dataset` | `lol_v1` | Target dataset: `lol_v1`, `lolv2_real`, `lolv2_syn`, `SID`, `fivek`, `SICE_mix` |
| `--cropSize` | `256` | Random crop patch size during training |
| `--batchSize` | `4` | Training batch size per GPU |
| `--nEpochs` | `1000` | Total training epochs |
| `--lr` | `1e-4` | Initial learning rate (Cosine Annealing Restart schedule) |
| `--dark_focus` | `True` | Prioritizes dark region optimization in IG_Mamba and LSGD loss |
| `--L1_weight` | `1.0` | Weight for L1 MAE pixel loss |
| `--D_weight` | `0.5` | Weight for SSIM structural loss |
| `--E_weight` | `50.0` | Weight for Edge consistency loss |
| `--P_weight` | `0.01` | Weight for VGG perceptual loss |
| `--gamma` | `False` | Applies random gamma augmentation curves for generalization |
| `--start_epoch` | `0` | Epoch index to resume training from (`> 0` loads corresponding checkpoint) |

Checkpoints are automatically saved after every `--snapshots` epochs into `./weights/train/epoch_{epoch}.pth`.

---

## 6. Testing & Evaluation Guide

Evaluation consists of two standard steps: **Step 1: Generate Enhanced Images** $\rightarrow$ **Step 2: Quantitative Metric Measurement**.

### Step 1: Generate Enhanced Images (`eval.py`)

Run inference using your desired dataset flag and checkpoint:

- **Evaluate on LOLv1:**
  ```bash
  python eval.py --lol
  ```
  *(Add `--perc` if evaluating a checkpoint trained with perceptual loss: `python eval.py --lol --perc`)*
  *Outputs will be saved in `./output/LOLv1/`.*

- **Evaluate on LOLv2-Real:**
  ```bash
  # Evaluate using Best PSNR checkpoint
  python eval.py --lol_v2_real --best_PSNR

  # Evaluate using Best SSIM checkpoint
  python eval.py --lol_v2_real --best_SSIM
  ```
  *Outputs will be saved in `./output/LOLv2_real/`.*

- **Evaluate on LOLv2-Synthetic:**
  ```bash
  python eval.py --lol_v2_syn --perc
  ```
  *Outputs will be saved in `./output/LOLv2_syn/`.*

- **Evaluate on Unpaired Real-World Datasets (LIME, DICM, etc.):**
  ```bash
  python eval.py --unpaired --LIME
  python eval.py --unpaired --DICM
  ```

### Step 2: Compute Quantitative Metrics (`measure.py`)

The script compares the generated images in `./output/` against ground truth references:

```bash
# Evaluate LOLv1 test set
python measure.py --lol

# Evaluate LOLv2-Real test set
python measure.py --lol_v2_real

# Evaluate LOLv2-Synthetic test set
python measure.py --lol_v2_syn
```

**Standard Terminal Output:**
```
100%|███████████████████████████████████████████| 15/15 [00:01<00:00, 10.2img/s]
===> Avg.PSNR: 24.5218 dB 
===> Avg.SSIM: 0.8542 
===> Avg.LPIPS: 0.1245 
```

> [!NOTE]
> The metric evaluation in `measure.py` calculates PSNR/SSIM across standard 8-bit RGB color channels, strictly following the benchmark conventions used in recent SOTA papers (HVI-CIDNet, Retinexformer), ensuring fair and consistent comparisons.

---

## 7. Analysis & Utility Tools

### 7.1. No-Reference Quality Metrics: NIQE & BRISQUE (`measure_niqe_bris.py`)
Used for evaluating real-world night photography without ground truth references (e.g., LIME, DICM, MEF):
```bash
python measure_niqe_bris.py
```

### 7.2. Model Complexity Profiling (`net_test.py`)
Computes the parameter count, FLOPs (on $1 \times 3 \times 256 \times 256$ inputs), and runtime latency:
```bash
python net_test.py
```

### 7.3. IG-Mamba Delta Modulation Visualization (`visualize_ig_mamba.py`)
Extracts and visualizes intermediate modulation maps ($\Delta$), attention scores, and intensity representations for qualitative ablation figures:
```bash
python visualize_ig_mamba.py --input dataset/LOL/LOLv1/test/low/1.png --output_dir ./visualize_results/
```

---

## 8. Citation

If you find this codebase or model architecture useful for your research, please consider citing:

```bibtex
@article{igmamba_edgelca_llie,
  title={Illumination-Guided Mamba with Edge-Modulated Lightweight Cross-Attention for Low-Light Image Enhancement},
  author={Vinh Nub and Collaborators},
  journal={arXiv preprint},
  year={2026}
}
```

---
*Contributions, suggestions, and feedback are welcome!*