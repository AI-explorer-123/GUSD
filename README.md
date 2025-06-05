# GUSD: Genre-aware and User-specific Spoiler Detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
Official code for [“Unveiling the Hidden: Movie Genre and User Bias in Spoiler Detection”](https://arxiv.org/abs/2504.17834).

Authors: Haokai Zhang\*, Shengtao Zhang\*, Zijian Cai, Heng Wang, Ruixuan Zhu, Zinan Zeng, Minnan Luo†  

<p align="center">
  <img src='images/ovreview.png'>
</p>

---

## Table of Contents
- [GUSD: Genre-aware and User-specific Spoiler Detection](#gusd-genre-aware-and-user-specific-spoiler-detection)
  - [Table of Contents](#table-of-contents)
  - [📂Project Structure](#project-structure)
  - [🛠 Requirements](#-requirements)
  - [🛠 Data Preparation](#-data-preparation)
  - [📊 Configuration](#-configuration)
  - [🚀 Quick Start](#-quick-start)
    - [Baseline Training](#baseline-training)
    - [Improved Variants](#improved-variants)
    - [Accelerated / Distributed Training](#accelerated--distributed-training)
    - [Heterogeneous‐Graph Variant](#heterogeneousgraph-variant)
    - [New MoE Variant](#new-moe-variant)
    - [Testing](#testing)
  - [Model Overview](#model-overview)
  - [Results \& Reproducibility](#results--reproducibility)
  - [Citation](#citation)
  - [License](#license)

---

## 📂Project Structure

```
GUSD/
├── .gitignore
├── README.md                  ← this file
├── paper.md                   ← full paper
├── config.py                  ← default hyperparams & paths
├── utils.py                   ← data loaders, batching, metrics
├── main.py                    ← baseline train/eval
├── changed_main.py            ← improved-module experiments
├── ace_changed_main.py        ← Accelerate (distributed) training
├── hetero_main.py             ← heterogeneous-graph variant
├── new_main.py                ← new GMoE variant
└── models_/
    ├── main_model.py          ← GUSD model (R2GFormer + GMoE + fusion)
    ├── meta_encoder.py        ← user/movie metadata encoder
    ├── model_utils.py         ← shared layers, loss, metrics
    ├── moe.py                 ← GMoE implementation
    ├── new_moe.py             ← alternative MoE design
    ├── test.py                ← unit tests
    └── graph_encoder/
        ├── Conv.py            ← RetGAT multi-hop graph conv
        └── models.py          ← GenreFormer & graph modules
```

---

## 🛠 Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.9  
- CUDA 11.x (for GPU)  
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
Key packages: `torch`, `torch-geometric`, `transformers`, `yacs`, `accelerate`, `numpy`, `pandas`, `scikit-learn`, `tqdm`.

---

## 🛠 Data Preparation

1. Place raw LCS/Kaggle data under `data/raw/`, the data will be available soon.  
2. Run preprocessing:
   ```bash
   python utils.py \
     --mode preprocess \
     --input_dir data/raw \
     --output_dir data/processed
   ```

---

## 📊 Configuration

Edit `config.py` or supply a YAML file via `--config` to override defaults.  
Example fields in `config.py`:

```python
DATA = {
    'LCS': 'data/processed/lcs/',
    'KAGGLE': 'data/processed/kaggle/'
}
TRAIN = {
    'batch_size': 64,
    'lr': 1e-4,
    'weight_decay': 1e-5,
    'epochs': 20,
    'num_hops': 2,
    'weight': 1.3
}
MODEL = {
    'graph_encoder': { 'layers': 2, 'hidden_channels': 1024, … },
    'meta_encoder':  { 'layers': 3, 'out_channels': 1024 },
    'moe':           { 'num_experts': 32, 'hidden_channels': 4096, … }
}
```

---

## 🚀 Quick Start

### Baseline Training

```bash
python main.py \
  --dataset LCS \
  --mode train \
  --save_dir outputs/lcs_baseline
```

### Improved Variants

```bash
python changed_main.py \
  --dataset KAGGLE \
  --mode train \
  --save_dir outputs/kaggle_changed
```

### Accelerated / Distributed Training

```bash
accelerate launch ace_changed_main.py \
  --config configs/your_config.yaml \
  --opts dataset LCS train.epochs 30
```

### Heterogeneous‐Graph Variant

```bash
python hetero_main.py \
  --dataset LCS \
  --mode train \
  --save_dir outputs/lcs_hetero
```

### New MoE Variant

```bash
python new_main.py \
  --dataset KAGGLE \
  --mode train \
  --save_dir outputs/kaggle_newmoe
```

### Testing

```bash
python -m models_.test
```

---

## Model Overview

- **Data & Batching** (`utils.py`): k-hop subgraph sampling, genre-aware batching, user-bias loading.  
- **Graph Encoder** (`graph_encoder/`): RetGAT multi-hop GAT + GenreFormer for inter-genre fusion.  
- **Meta Encoder** (`meta_encoder.py`): MLP for user/movie metadata.  
- **GMoE** (`moe.py`, `new_moe.py`): genre-aware Mixture-of-Experts routing.  
- **Main Model** (`main_model.py`): combines graph, meta, user bias, textual features & classifier.

---

## Results & Reproducibility

See [out paper](https://arxiv.org/abs/2504.17834) for detailed tables and figures:

- **LCS**: +6.1% F1, +5.5% AUC over previous SOTA.  
- **Kaggle**: similar improvements.  
- Ablation confirms RetGAT, GenreFormer, GMoE, and user bias contributions.

---

## Citation

```bibtex
@article{zhang2025unveiling,
  title={Unveiling the Hidden: Movie Genre and User Bias in Spoiler Detection},
  author={Zhang, Haokai and Zhang, Shengtao and Cai, Zijian and Wang, Heng and Zhu, Ruixuan and Zeng, Zinan and Luo, Minnan},
  journal={arXiv preprint arXiv:2504.17834},
  year={2025}
}
```

---

## License

MIT © Haokai Zhang et al.  
See [LICENSE](LICENSE) for details.