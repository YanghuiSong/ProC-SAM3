# ProC‑SAM3: Prompt‑Calibrated SAM 3 for Open‑Vocabulary Remote Sensing Semantic Segmentation

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D2.2-orange.svg)](https://pytorch.org/)

## Overview

![Method overview](assets/main.png)

Open-vocabulary semantic segmentation (OVSS) in remote sensing images aims to segment categories beyond a fixed label space, but remains challenging due to complex geospatial scenes, large-scale variations, and dense small objects. Recent SAM 3-based methods provide a promising training-free foundation, yet three key issues remain: (1) a single class-name prompt lacks sufficient semantic coverage for complex remote sensing categories; (2) expanding each category into multiple prompts introduces redundant online text encoding; and (3) directly aggregating multiple prompt responses propagates noisy activations into the final prediction. To address these issues, we propose ProC-SAM3, a prompt-calibrated SAM 3 framework for remote sensing OVSS. First, we construct an offline prompt pool through a two-level convergence pipeline, where a Category Matcher groups MLLM-generated candidates into per-category sets, and Expansion Constraints further refine each set using category-specific prior knowledge. Second, the resulting text embeddings are cached and reused across all test images, eliminating repeated text encoding. Third, we introduce Presence-Guided Residual Fusion (PGRF) to gate unreliable instance responses by prompt presence and confidence, followed by peak-preserving class aggregation that retains fine-grained activations for small and sparse objects. Experiments on eight benchmarks show that ProC-SAM3 achieves an average mIoU of 56.1%, outperforming the previous best training-free method by 3.9 percentage points.

## Key Features

- **Multimodal prompting** – text + optional visual prompts  
- **Robust fusion** of multiple prompt expansions  
- **Training‑free** – works off‑the‑shelf with SAM 3 + Qwen‑VL  
- **High accuracy** – outperforms previous methods across 8 datasets  

## Datasets

We evaluate ProC‑SAM3 on eight diverse remote sensing benchmarks.  
All data preprocessing follows the instructions from [SegEarth‑OV](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md).

- LoveDA, Potsdam, Vaihingen, iSAID  
- OEM, UDD5, VDD, UAVid  

## Results

The table below reports **mIoU** (%) for open‑vocabulary semantic segmentation.  
Our method achieves the highest average performance across all datasets.

![Result table](assets/results.png)

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/YanghuiSong/ProC-SAM3.git
   cd ProC-SAM3
   ```

2. **Set up the environment**  
   Follow the official [SAM 3 installation guide](https://github.com/facebookresearch/sam3) to prepare the environment (our code is built on top of SAM 3).

3. **Download required models**  
   - [Qwen3‑VL‑8B‑Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)  
   - [SAM 3 checkpoint](https://huggingface.co/facebook/sam3) (from the official Meta repository)

4. **Configure paths**  
   Update the model paths in `config.py` to point to your local checkpoints.

## Run Evaluation on a Dataset

```bash
python eval.py --config configs/cfg_DATASET_pgrf_max.py
```

Replace `DATASET` with the desired dataset name (e.g., `vaihingen`, `potsdam`).

## Visualization

```bash
python visualize_segmentation.py --input-path images/ --output-dir results/
```

## Acknowledgments

This work is based on [SAM 3](https://github.com/facebookresearch/sam3) and [SegEarth‑OV3](https://github.com/earth-insights/SegEarth-OV-3). We thank the authors for their excellent open‑source contributions.
