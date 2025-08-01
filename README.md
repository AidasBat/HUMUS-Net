# HUMUS-Net Evaluation on fastMRI Brain Dataset

This repository contains an independent evaluation of the HUMUS-Net model applied to the **fastMRI Brain dataset**.

---

## Summary

HUMUS-Net is a state-of-the-art hybrid unrolled multi-scale network architecture originally designed for accelerated MRI reconstruction on knee data.  
This evaluation adapts and tests HUMUS-Net on the fastMRI brain dataset, using updated metrics and an improved evaluation script.

---

## Evaluation Contents

- `evaluation/eval.py`: Updated evaluation script customized for fastMRI brain data.  
- `results.csv`: Performance metrics obtained from our evaluation.  
- `poster.pdf`: Research poster summarizing key findings.  
- `figures/`: Visualizations of reconstruction results and metric trends.

---

## How to Reproduce

1. Clone this forked repository.
2. Follow the original HUMUS-Net installation and setup instructions [here](https://github.com/z-fabian/HUMUS-Net#installation).
3. Download and prepare the fastMRI brain dataset from [fastMRI](https://fastmri.med.nyu.edu/).
4. Run the evaluation using the updated `eval.py` script:

