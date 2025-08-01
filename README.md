# HUMUS-Net Evaluation on fastMRI Brain Dataset

This repository contains an independent evaluation of the HUMUS-Net model applied to the **fastMRI Brain dataset**.

---

## Summary

HUMUS-Net is a state-of-the-art hybrid unrolled multi-scale network architecture originally designed for accelerated MRI reconstruction on knee data.  
This evaluation adapts and tests HUMUS-Net on the fastMRI brain dataset, using updated metrics and an improved evaluation script.

---

## Evaluation Contents

- `evaluation/eval.py`: Updated evaluation script customized for fastMRI brain data.  
- `evaluation/poster.pdf`: Research poster summarizing key findings.  
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
4. Run the evaluation using the updated `eval.py` script:

