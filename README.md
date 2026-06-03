<div align="center">

# DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors

**Pengcheng Wang**<sup>1</sup> · **Kaiwen Hong**<sup>2</sup> · **Chensheng Peng**<sup>1</sup> · **Katherine Driggs-Campbell**<sup>2</sup> · **Masayoshi Tomizuka**<sup>1</sup> · **Chenfeng Xu**<sup>3</sup> · **Chen Tang**<sup>4</sup>

<sup>1</sup>UC Berkeley · <sup>2</sup>UIUC · <sup>3</sup>UT Austin · <sup>4</sup>UCLA

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2604.25050)
[![Project Page](https://img.shields.io/badge/Project-Page-2575fc.svg)](https://outsider86.github.io/DiscreteRTCSite/)
[![StarVLA](https://img.shields.io/badge/⭐_StarVLA-Maintained_Implementation-6a11cb.svg)](https://github.com/starVLA/starVLA)
[![Kinetix branch](https://img.shields.io/badge/branch-kinetix-1a8c1a.svg)](https://github.com/outsider86/DiscreteRTC/tree/kinetix)
[![Real branch](https://img.shields.io/badge/branch-real-2575fc.svg)](https://github.com/outsider86/DiscreteRTC/tree/real)

</div>

<p align="center">
  <video src="https://github.com/outsider86/DiscreteRTC/raw/main/assets/overview.mp4" controls autoplay muted loop width="85%"></video>
</p>



---

## TL;DR: Discrete diffusion policies are natural asynchronous executors.

RTC was designed for flow-matching action heads, yet RTC with a flow-matching head has four critical limitations, all from one root cause: *the inpainting capability comes from inference-time correction (e.g., ΠGDM), not the base policy.*

| | Flow-matching RTC | Discrete Diffusion RTC |
|---|---|---|
| **a** | Pre-training without inpainting | Inpainting *is* the pre-training task |
| **b** | Dedicated fine-tuning required (Training-time RTC) | Fine-tuning free |
| **c** | Heuristic, fixed guidance schedule | Natural, adaptive guidance via early-exit |
| **d** | Extra inference cost from VJP | Lower inference cost with fewer tokens to unmask |


## Repository guide

This repository is organized into **three branches**:

| Branch | Purpose |
|---|---|
| **`main`** (here) | Project overview, key result plots, and guidance to the other branches. |
| [**`kinetix`**](https://github.com/outsider86/DiscreteRTC/tree/kinetix) | **Simulation pipeline.** Full Kinetix reproduction code — expert training, data generation, discrete diffusion / flow policy training, and evaluation across inference delays. |
| [**`real`**](https://github.com/outsider86/DiscreteRTC/tree/real) | **Real-world experiments infrastructure** for the UR5e dynamic manipulation tasks. |

---

## ⭐ Use and extend DiscreteRTC in StarVLA

DiscreteRTC is **intentionally lightweight** — the core method is essentially *a call*, not a heavy framework. The kinetix-branch code here reproduces the paper's experiments, but if you want to **build on, extend, or apply DiscreteRTC for real-world experiments with VLA**, we strongly encourage you to use our actively maintained implementation in **StarVLA**, where we keep it up to date👉 https://github.com/starVLA/starVLA

---

## Citation

```bibtex
@article{wang2026discretertc,
  title={DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors},
  author={Wang, Pengcheng and Hong, Kaiwen and Peng, Chensheng and Driggs-Campbell, Katherine and Tomizuka, Masayoshi and Xu, Chenfeng and Tang, Chen},
  journal={arXiv preprint arXiv:2604.25050},
  year={2026}
}
```
