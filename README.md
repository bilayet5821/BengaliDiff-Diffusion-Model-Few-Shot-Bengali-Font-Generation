<div align="center">

# 🅱️ BengaliDiff

### Diffusion Model for Few-Shot Bengali Font Generation

**A diffusion-based generative framework for synthesizing Bengali glyphs  
from limited style references while preserving character structure and font identity.**

<br>

[![Paper](https://img.shields.io/badge/📄_Paper-Springer-1f6feb?style=for-the-badge)](https://link.springer.com/chapter/10.1007/978-3-032-09371-4_7)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--032--09371--4__7-green?style=for-the-badge)](https://doi.org/10.1007/978-3-032-09371-4_7)
![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

<br>

**Md Bilayet Hossain · Honghui Yuan · Shabnur Anonna Akhy · Keiji Yanai**

📚 **ICDAR 2025 Workshops · Springer LNCS**

</div>

---

## 🔍 Overview

**BengaliDiff** is a research framework for **few-shot Bengali font generation** using diffusion models.

Generating Bengali fonts is challenging because Bengali contains complex glyph structures, modifiers, compound characters, and substantial stylistic variation. BengaliDiff investigates how a generative model can learn a target font style from only a limited number of reference glyphs and transfer that style to other Bengali characters.

The framework combines **content representation**, **style representation**, **attention-based feature integration**, and **diffusion-based generation** to synthesize Bengali glyphs while preserving both character identity and visual style.

> 📌 This repository contains the research implementation associated with our published BengaliDiff work.

---

## ✨ Research Highlights

- 🅱️ **Bengali-specific font generation**
- 🎨 **Few-shot style transfer**
- 🌫️ **Diffusion-based glyph synthesis**
- 🧠 **Content & style representation learning**
- 🔗 **Attention-based feature integration**
- ⚔️ **Adversarial supervision components**
- ⚡ **DPM-Solver++ accelerated sampling**
- 🔬 **SFUC & UFUC experimental evaluation**

---

## 🏗️ BengaliDiff Architecture

<p align="center">
  <img src="figures/bengalidiff_architecture.png" width="900">
</p>

<p align="center">
  <i>Overview of the BengaliDiff framework for few-shot Bengali font generation.</i>
</p>

The generation pipeline can be summarized as:

text
          Content Glyph
               │
               ▼
        Content Encoder
               │
               │
               ├───────────────┐
               │               │
               │        Reference Glyph
               │               │
               │               ▼
               │         Style Encoder
               │               │
               └───────┬───────┘
                       ▼
             Content–Style Fusion
                       │
                       ▼
                Diffusion U-Net
                       │
                       ▼
               Reverse Denoising
                       │
                       ▼
                 DPM-Solver++
                       │
                       ▼
             Generated Bengali Glyph
🖼️ Qualitative Results
<p align="center"> <img src="figures/bengalidiff_qualitative_results.png" width="900"> </p> <p align="center"> <i>Examples of Bengali glyph generation under the experimental settings used in BengaliDiff.</i> </p>
🧪 Ablation Study
<p align="center"> <img src="figures/bengalidiff_ablation.png" width="900"> </p>

The ablation experiments examine the contribution of the proposed components to style preservation and Bengali glyph generation quality.

For complete quantitative analysis, experimental settings, and comparisons, please refer to the published paper.

📄 Publication

BengaliDiff: Diffusion Model for Few-Shot Bengali Font Generation
Md Bilayet Hossain, Honghui Yuan, Shabnur Anonna Akhy, Keiji Yanai
Document Analysis and Recognition – ICDAR 2025 Workshops
Lecture Notes in Computer Science, Vol. 16226, pp. 101–115
Springer, Cham.

📖 Paper:
https://link.springer.com/chapter/10.1007/978-3-032-09371-4_7

🔗 DOI:
https://doi.org/10.1007/978-3-032-09371-4_7

📂 Repository Structure
BengaliDiff/
│
├── configs/                 # Training configuration
├── dataset/                 # Dataset loader and batching
├── data_examples/           # Minimal reproducibility examples
├── figures/                 # Architecture and research results
├── scripts/                 # Training and sampling scripts
│
├── src/
│   ├── dpm_solver/          # DPM-Solver implementation
│   ├── modules/             # Model components
│   ├── build.py
│   ├── criterion.py
│   ├── discriminator.py
│   └── model.py
│
├── char_map.json            # Bengali character mapping
├── train.py                 # Main training pipeline
├── train_disc.py            # Discriminator training
├── sample.py                # Glyph generation / inference
├── evaluation.py            # Evaluation utilities
├── gradio_app.py            # Interactive demo interface
├── requirements.txt
└── README.md
⚙️ Installation
1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
2. Create Environment
conda create -n bengalidiff python=3.9 -y
conda activate bengalidiff
3. Install PyTorch
pip install torch==1.13.1+cu117 \
torchvision==0.14.1+cu117 \
torchaudio==0.13.1 \
--extra-index-url https://download.pytorch.org/whl/cu117
4. Install Dependencies
pip install -r requirements.txt
5. Configure Accelerate
accelerate config
📊 Dataset Preparation

The complete research dataset is not distributed with this repository.

Prepare Bengali glyphs using the following structure:

DATA_ROOT/
└── train/
    ├── ContentImage/
    │   ├── char01.jpg
    │   ├── char02.jpg
    │   └── ...
    │
    └── TargetImage/
        ├── FontStyle01/
        │   ├── FontStyle01+char01.jpg
        │   ├── FontStyle01+char02.jpg
        │   └── ...
        │
        └── FontStyle02/
            ├── FontStyle02+char01.jpg
            ├── FontStyle02+char02.jpg
            └── ...
Naming Convention
<style>+<content>.jpg

Example:

TargetImage/
└── AdorshoLipi/
    ├── AdorshoLipi+char01.jpg
    ├── AdorshoLipi+char02.jpg
    └── AdorshoLipi+char03.jpg

The corresponding content glyphs must exist inside ContentImage/.

🚀 Training
Phase 1
bash scripts/train_phase_1.sh

Phase 1 trains the primary diffusion-based font generation framework.

Phase 2
bash scripts/train_phase_2.sh

Phase 2 uses the trained Phase-1 checkpoint and additional style-contrastive supervision.

Adversarial Components

BengaliDiff also includes discriminator-related components:

train_disc.py
src/discriminator.py
src/modules/discriminator.py

These files contain the adversarial-supervision components used during BengaliDiff development.

🎨 Generate Bengali Glyphs

A trained checkpoint is required for inference.

Typical checkpoint structure:

ckpt/
├── unet.pth
├── content_encoder.pth
└── style_encoder.pth

Run:

bash scripts/sample_content_image.sh

or use:

python sample.py --help

⚠️ Pretrained BengaliDiff weights are not included in this repository.

📏 Evaluation

Evaluation utilities are provided in:

evaluation.py

The research evaluates Bengali font generation under different generalization settings, including:

Protocol	Description
SFUC	Seen Font, Unseen Character
UFUC	Unseen Font, Unseen Character

For the exact experimental protocol, quantitative results, and comparison with existing methods, please refer to the published paper.

🔬 Reproducing the Research

For reproduction:

Prepare Bengali glyphs following the documented dataset structure.
Install the pinned research environment.
Configure the dataset/output paths.
Train the Phase-1 model.
Run the subsequent training stage required by the experiment.
Generate Bengali glyphs using the sampling pipeline.
Evaluate generated outputs.
Compare against the experimental protocol reported in the paper.

Large checkpoints, complete datasets, OCR weights, logs, and experiment dumps are intentionally excluded from the public repository.

⚠️ Limitations

BengaliDiff is a research implementation, not a production font-design system. Generation quality depends on the diversity and quality of training fonts, and reproducing the exact published experiments requires following the experimental configuration described in the paper.

🙏 Acknowledgements

BengaliDiff builds upon the excellent work:

FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning
Zhenhua Yang, Dezhi Peng, Yuxin Kong, Yuyi Zhang, Cong Yao, Lianwen Jin
AAAI 2024

🔗 https://github.com/yeungchenwa/FontDiffuser

We gratefully acknowledge the FontDiffuser authors for making their research implementation publicly available.

BengaliDiff adapts and extends this foundation for few-shot Bengali font generation.

📝 Citation

If BengaliDiff is useful in your research, please cite:

@inproceedings{hossain2026bengalidiff,
  title     = {BengaliDiff: Diffusion Model for Few-Shot Bengali Font Generation},
  author    = {Hossain, Md Bilayet and Yuan, Honghui and
               Akhy, Shabnur Anonna and Yanai, Keiji},
  booktitle = {Document Analysis and Recognition -- ICDAR 2025 Workshops},
  series    = {Lecture Notes in Computer Science},
  volume    = {16226},
  pages     = {101--115},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/978-3-032-09371-4_7}
}
⚖️ Usage Notice

This repository contains code adapted from the FontDiffuser research implementation.

The upstream FontDiffuser repository provides its code for non-commercial research purposes. Users of this repository should therefore review and comply with the applicable upstream terms and notices.

This repository should not be interpreted as an independently MIT-licensed reimplementation of all included source code.

<div align="center">
⭐ BengaliDiff

Advancing Generative AI for Bengali Script

📄 Read the Paper
  •  
🔗 DOI

<br>

Computer Vision · Generative AI · Diffusion Models · Bengali Font Generation

</div> 
