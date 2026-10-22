<div align="center">

# GVER

### Generative Vision-Language Entity Retrieval

<p align="center">  
  <img alt="PyTorch"      src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch\&logoColor=white">  
  <img alt="Transformers" src="https://img.shields.io/badge/Transformers-4.55%2B-ffd21e?logo=huggingface\&logoColor=black">  
  <img alt="Python"       src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python\&logoColor=white">  
  <img alt="License"      src="https://img.shields.io/badge/License-MIT-6c757d">  
</p>

</div>

---

## ✨ Overview

**GVER** is a generative retrieval model for vision-language entity search on the M-BEIR benchmark (task 8: OVEN & INFOSEEK). It represents every gallery item as a short **Semantic ID (SID)** produced by a deep residual quantizer over fused CLIP-SF features, and lets a T5 decoder generate the SID of the target item directly from a multimodal query, using a prefix trie for constrained beam search. All numbers in the paper are obtained under the strict no-rerank protocol.

---

## 🗂️ Repository Layout

```
GVER/
├── M2M_SID/              # ⭐ The main model proposed in this work
├── PRO_baseline/         # PRO: residual-quantized SID baseline
├── GENIUS_baseline/      # GENIUS-R: generative retrieval with CLIP-SF features
├── Qwen3VL_baseline/     # Qwen3-VL-Embedding-8B dense-retrieval baseline
└── WeMM_baseline/        # WeMM-Embedding-9B dense-retrieval baseline
```

---

## 🚀 Quick Start

```bash
# 1. install core dependencies
conda create -n gm2m python=3.10 -y
conda activate gm2m
pip install -r requirements.txt

# 2. prepare CLIP-SF features for OVEN + INFOSEEK
bash M2M_SID/scripts/prepare_feats.sh

# 3. train & evaluate GM2M
GPU=0 CONFIG=configs/oven.yaml     TAG=OVEN     bash M2M_SID/scripts/run.sh
GPU=1 CONFIG=configs/infoseek.yaml TAG=INFOSEEK bash M2M_SID/scripts/run.sh
```

> ⚠️ `WeMM_baseline` requires `transformers==5.2.0`, which conflicts with  
> the Qwen3-VL requirement. Please install it in a *separate* conda env.

---

## 📄 License

Released under the MIT License. Baseline encoders (CLIP-SF, GENIUS, Qwen3-VL,  
WeMM) remain subject to their original licenses.
