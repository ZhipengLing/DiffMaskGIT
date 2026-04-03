# DiffMaskGIT

**Enhancing Masked Generative Image Transformers via Diffusion-Guided Training and Hybrid Confidence-Aware Sampling**

CS7150 Deep Learning — Northeastern University

Authors: Zhipeng Ling, Zichen Tian

## Overview

This project addresses two fundamental limitations of [MaskGIT](https://arxiv.org/abs/2202.04200) (Masked Generative Image Transformer):

1. **VQ Codebook Bottleneck** — Vector quantization imposes a reconstruction quality ceiling (rFID ~5.87 on ImageNet-256). We replace the cross-entropy loss with a **diffusion loss module** (inspired by [MAR](https://arxiv.org/abs/2406.11838)), enabling continuous-valued latent representations via a KL-16 VAE encoder.

2. **Gumbel-Max Sampling Instability** — The standard sampling strategy causes token oscillation and spatial clustering during iterative decoding. We propose a **Hybrid Confidence-Aware Sampler** that uses greedy decoding for high-confidence tokens and stochastic sampling for uncertain positions, combined with Halton-sequence spatial diversification for re-masking.

Additionally, we integrate **2D Axial Rotary Position Embeddings (RoPE)** into the bidirectional transformer to enhance spatial coherence.

## Repository Structure

```
DiffMaskGIT/
├── notebooks/
│   └── DiffMaskGIT_Colab.ipynb   # Main experiment notebook (self-contained, runs on Colab)
├── paper/
│   ├── CS7150_Progress_Report_2.tex   # LaTeX source
│   ├── CS7150_Progress_Report_2.pdf   # Compiled report
│   └── CS7150_Progress_Report_2_tex.zip
└── README.md
```

## Notebook Contents

`notebooks/DiffMaskGIT_Colab.ipynb` is a single, self-contained Google Colab notebook that implements everything from scratch. No external code dependencies beyond standard pip packages.

### Models Implemented

| Component | Description |
|-----------|-------------|
| **2D Axial RoPE** | Rotary position embeddings split by row/column coordinates, parameter-free |
| **Bidirectional Transformer** | 8-layer, 8-head, 256-d masked transformer backbone (~6-7M params) |
| **Vector Quantizer** | EMA-updated codebook (K=1024, dim=4) for discrete token baselines |
| **Diffusion Loss MLP** | 2-block denoising MLP with AdaLN conditioning and cosine DDPM schedule |
| **Sampling Strategies** | Gumbel-Max, Halton scheduler, and Hybrid Confidence-Aware sampler |

### Experiments (4 methods compared)

| # | Method | Pos Encoding | Loss | Sampler |
|---|--------|-------------|------|---------|
| 1 | VQ-VAE | — | Reconstruction | — |
| 2 | MaskGIT | Learned | Cross-Entropy | Gumbel-Max |
| 3 | MaskGIT + 2D RoPE | 2D Axial RoPE | Cross-Entropy | Gumbel-Max |
| 4 | **DiffMaskGIT (Ours)** | 2D Axial RoPE | Diffusion | Hybrid |

### Results

| Method | Params | FID | IS |
|--------|--------|-----|-----|
| VQ-VAE (rFID) | — | 37.21 | — |
| MaskGIT | 6.9M | 334.25 | 1.71 |
| MaskGIT + 2D RoPE | 6.8M | 333.42 | 1.77 |
| DiffMaskGIT (Ours) | 7.1M | 426.03 | 1.77 |

**Key findings:**
- 2D Axial RoPE provides consistent marginal improvement (FID 334.25 → 333.42)
- Diffusion loss underperforms at this small scale due to MLP capacity bottleneck (0.2M vs. 37M in MAR)
- Halton scheduler achieves best FID at high step counts (352.40 at 24 steps)
- Annealed confidence threshold (τ: 0.5→0.9) outperforms all fixed thresholds

See the [paper](paper/CS7150_Progress_Report_2.pdf) for detailed analysis of scaling bottlenecks.

### Ablation Studies

- FID vs. decoding steps (4, 8, 12, 16, 24) for three sampling strategies
- Token oscillation heatmap (Gumbel-Max vs. Hybrid)
- Confidence threshold (tau) sensitivity analysis
- Image completion demo (right-half, bottom-half, random masking)

### Visualizations Generated (9 figures)

1. Training data samples + VAE latent representations
2. VQ-VAE reconstruction quality (original vs. reconstructed vs. difference)
3. Generated samples per method (8x8 grids)
4. Attention maps comparison (learned position embeddings vs. 2D RoPE)
5. 4-method side-by-side sample comparison
6. FID bar chart across all methods
7. FID vs. decoding steps line plot
8. Token oscillation heatmap
9. Image completion examples

## How to Run

### Requirements

- Google Colab with **GPU runtime** (T4 or better)
- ~3-4 hours for the full pipeline
- Google Drive for data caching and checkpoints

### Steps

1. Upload `notebooks/DiffMaskGIT_Colab.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells sequentially from top to bottom
4. All figures and results are saved to `Google Drive/MyDrive/DiffMaskGIT/`

The notebook automatically:
- Downloads the [ImageNette](https://github.com/fastai/imagenette) dataset (10 ImageNet classes, ~1.5GB)
- Pre-encodes all images through a frozen [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse) (128x128 -> 16x16x4 latents)
- Trains all 4 models with checkpointing (resumes if interrupted)
- Generates samples, computes FID/IS metrics, and produces all visualizations

### Output Structure on Google Drive

```
MyDrive/DiffMaskGIT/
├── checkpoints/          # Model checkpoints (auto-resume on re-run)
├── data/
│   ├── imagenette2-320/  # Raw dataset
│   └── latents/          # Pre-encoded VAE latents
├── figures/              # All generated figures (PNG)
└── results.json          # Quantitative results
```

## Key References

- [MaskGIT](https://arxiv.org/abs/2202.04200) — Chang et al., CVPR 2022
- [MAR (Diffusion Loss)](https://arxiv.org/abs/2406.11838) — Li et al., NeurIPS 2024
- [Halton Scheduler](https://arxiv.org/abs/2412.01819) — Besnier et al., ICLR 2025
- [RoFormer (RoPE)](https://arxiv.org/abs/2104.09864) — Su et al., 2021

## License

This project is for academic purposes (CS7150 Deep Learning coursework).
