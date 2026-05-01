# PRISM

Source code accompanying the paper

> *PRISM: A Cross-Domain Regression Framework Combining Variational Mode Decomposition, an Improved Crested Porcupine Optimizer, and Boosted Ensembles.*
> Siyu Chen, Wencan Guan. IEEE Access (under review).

PRISM is a five-stage pipeline (preprocess → mode-decompose → swarm-tune → regress → SHAP) that we benchmark on four cross-domain UCI regression tasks. The framework is built around an improved variant of the Crested Porcupine Optimizer (ICPO), which adds Tent-chaotic initialisation, Lévy-flight perturbation, and an adaptive elite-guided update on top of the original CPO.

## Repository layout

    prism/                package code
      data.py             dataset loaders for CCPP / Concrete / Appliances / PM2.5
      features.py         lag, rolling and cyclical-time features
      decomposition.py    Variational Mode Decomposition (VMD) wrapper
      optimizer.py        ICPO and the four CPO defensive branches
      tuning.py           ICPO ↔ LightGBM bridge (9-D search box)
      swarm_baselines.py  PSO / GA / GWO / RandomSearch / vanilla CPO
      dl_models.py        MLP and FT-Transformer-lite (PyTorch)
      metrics.py          RMSE / MAE / MAPE / R² + the three custom indices
      utils.py            seeding, timing, path helpers
    experiments/          numbered scripts that produce every table and figure
    results/              CSV/JSON/PNG outputs reproduced in the paper
    datasets/             place the four UCI datasets here (see datasets/README.md)

## Requirements

Tested on Python 3.12 with the versions in `requirements.txt`. A GPU is only needed for the two deep tabular baselines (`02c_prism_dl.py`); everything else runs on CPU.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Datasets

The four datasets are not redistributed in this repository. Download them from the UCI Machine Learning Repository (links and target paths are in `datasets/README.md`). The expected layout under `datasets/` is

    datasets/
      01_CCPP/CCPP/Folds5x2_pp.xlsx
      02_Concrete/Concrete_Data.xls
      03_AppliancesEnergy/energydata_complete.csv
      04_BeijingPM25/PRSA_data_2010.1.1-2014.12.31.csv

If you keep the data elsewhere, set `PRISM_DATA_DIR` to point at the parent directory.

## Reproducing the paper

The scripts under `experiments/` are numbered in the order they should be run. Each script writes its outputs into the matching subfolder of `results/`.

    python experiments/00_baselines.py            # Ridge / RF / XGB / LGBM, raw inputs (CPU, ~30 s)
    python experiments/01_features_ts.py          # same four with lag features on Appliances / PM2.5 (CPU, ~1 min)
    python experiments/02a_prism_static.py        # PRISM on CCPP / Concrete (CPU, ~3 min)
    python experiments/02b_prism_ts.py            # PRISM on Appliances / PM2.5, with VMD on Appliances (CPU, ~25 min)
    CUDA_VISIBLE_DEVICES=0 python experiments/02c_prism_dl.py  # MLP + FT-Transformer-lite (GPU)
    python experiments/04_shap.py                 # TreeSHAP attributions on the static datasets
    python experiments/07_custom_metrics.py       # CDGI / DSRR / DMU
    python experiments/08_ablation.py             # ICPO three-component ablation on Concrete
    python experiments/09_swarm_comparison.py     # ICPO vs PSO / GA / GWO / CPO / random search

The whole sweep takes well under an hour on a 32-thread CPU. We used an i9-14900K with two RTX 4070 Ti SUPER GPUs; only `02c` touches the GPU.

## Datasets and metrics at a glance

| Dataset           | N      | d  | Type        | Target                            |
| ----------------- | -----: | -: | ----------- | --------------------------------- |
| CCPP              | 9,568  | 4  | static      | net hourly electrical output (PE) |
| Concrete          | 1,030  | 8  | static      | 28-day compressive strength       |
| Appliances Energy | 19,735 | 28 | time-series | appliance energy use (Wh)         |
| Beijing PM2.5     | 41,757 | 13 | time-series | PM2.5 concentration (μg/m³)       |

Beyond standard RMSE / MAE / MAPE / R² we report three task-specific indices defined in `prism/metrics.py`: the Cross-Domain Generalisation Index (CDGI), the Distribution-Shift Robustness Ratio (DSRR), and the Decomposition Marginal Utility (DMU).

## Headline numbers

| Dataset      | Best non-PRISM (R²) | PRISM (R²)  | RMSE reduction |
| ------------ | ------------------: | ----------: | -------------: |
| CCPP         | 0.9703 (LightGBM)   | **0.9759**  | 10.7 %         |
| Concrete     | 0.9249 (XGBoost)    | **0.9503**  | 17.4 %         |
| Appliances   | 0.5642 (Ridge+lag)  | **0.9302**  | 60.0 %         |
| PM2.5        | 0.9505 (LGBM+lag)   | **0.9527**  |  2.2 %         |

On the Appliances dataset, the chronological 80/20 split puts the test horizon outside the training regime and out-of-the-box tree baselines collapse to negative R² (LightGBM −1.86, Random Forest −4.09, XGBoost −5.54). Combining lag features with VMD and ICPO-tuned LightGBM lifts R² to 0.93; the DMU metric attributes 59.4 % of the RMSE reduction to the VMD step alone.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{chen2026prism,
  title   = {{PRISM}: A Cross-Domain Regression Framework Combining Variational Mode Decomposition, an Improved Crested Porcupine Optimizer, and Boosted Ensembles},
  author  = {Chen, Siyu and Guan, Wencan},
  journal = {IEEE Access},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Wencan Guan — `wencan.guan@uni-weimar.de`
ORCID: 0009-0005-9398-1331

## License

MIT. See `LICENSE`.
