# Radio Environment Mapping with World Models for Active Measurement Control: Should Networks Dream of Optimal Control?


This repository contains a World Models models solution (Vision VAE + dynamics + dreaming sequence) of measurement selection for a **few-shot RSSI Radio Map reconstruction**. The full paper is available on arXiv as bellow. 

## Code 

The `main.py` contains the full experimental pipeline, including dataset loading, model initialization (VAE and dynamics model), loading of pretrained weights, execution of baseline methods (copy-empty and Gaussian Process).The script produces quantitative metrics (e.g., RMSE/MAE) and generates qualitative visualizations such as reconstruction heatmaps and performance curves.

The repository includes a `requirements.txt` file for installing all dependencies, as well as pretrained model weights to facilitate reproducible evaluation without the need for retraining.

### Dataset

The RSSI dataset used in this work is available on [Zenodo](https://zenodo.org/records/15791300).

```bibtex
@article{milosheski2025_indoor_radio_mapping_arxiv,
  author  = {Milosheski, Ljupcho and Akiyama, Kuon and Bertalani{\v{c}}, Bla{\v{z}} and Hribar, Jernej and Shinkuma, Ryoichi},
  title   = {An Indoor Radio Mapping Dataset Combining 3D Point Clouds and RSSI},
  journal = {arXiv},
  year    = {2025},
  eprint  = {2511.00494},
  url     = {https://arxiv.org/abs/2511.00494}
}
```
