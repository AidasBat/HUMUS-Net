# HUMUS-Net Evaluation on fastMRI Brain Dataset

This repository contains an **independent evaluation** of the HUMUS-Net model applied to the **fastMRI Brain dataset**.

---

## Summary

HUMUS-Net is a state-of-the-art hybrid unrolled multi-scale network architecture originally designed for accelerated MRI reconstruction on knee data.  
This evaluation adapts and tests HUMUS-Net on the fastMRI brain dataset, using updated evaluation script.

---

## Evaluation Contents

- `evaluation/eval.py`: Updated evaluation script customized for fastMRI brain data.  
- `evaluation/poster.pdf`: Research and cross-domain analysis poster summarizing evaluation results.  
- `evaluation/figures`: A few visualization of a reconstruction examples.

---

## Evaluation results

Evaluation on fastMRI Brain dataset (8x acceleration, 4% center k-space) results:

| Method                                                                                      | SSIM       | NMSE       | PSNR     |
| ------------------------------------------------------------------------------------------- | ---------- | ---------- | -------- |
| **HUMUS-Net**                                                                               | **0.8871** | **0.0265** | **31.9** |

Slice 08 reconstruction and error map:

<img width="2205" height="719" alt="file_brain_AXFLAIR_200_6002471 h5_slice_002_comparison" src="https://github.com/user-attachments/assets/8352a6f7-3656-4844-b2f0-ee9648587a2d" />

---

## How to Reproduce

1. Clone this forked repository.
2. Follow the original HUMUS-Net installation and setup instructions [here](https://github.com/z-fabian/HUMUS-Net#installation).
3. Download and prepare the fastMRI brain dataset from [fastMRI](https://fastmri.med.nyu.edu/).
4. Run the evaluation using the updated `evaluation/eval.py` script:

---

## Acknowledgments and references
- [HUMUS-Net repository]( https://github.com/z-fabian/HUMUS-Net)
- **HUMUS-Net: Hybrid Unrolled Multi-scale Network Architecture for Accelerated MRI Reconstruction**: Fabian, Z., Tinaz, B. and Soltanolkotabi, M. (2022) ‘HUMUS-Net: Hybrid unrolled multi-scale network architecture for accelerated MRI reconstruction’. arXiv. Available at: https://doi.org/10.48550/ARXIV.2203.08213.*
- [fastMRI repository]( https://github.com/facebookresearch/fastMRI)
- **fastMRI**: Zbontar et al., *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI, https://arxiv.org/abs/1811.08839*

