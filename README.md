# PCA Rotation in Tree-Based Prediction

Code and results for the paper

> **PCA Rotation in Tree-Based Prediction: A Systematic Study of Variance-Oriented Degradation and Certified Task-Sufficient Dimensionality**
> Abu Jafar Sisty, Niloy Roy, S M Abdullah Shafi, Abhijit Bhowmik
> Department of Computer Science, American International University-Bangladesh
> ICECE 2026 (accepted, oral presentation)

## Summary

PCA with all components kept is an orthogonal rotation of the standardized features, so no information is lost. We show that this rotation still hurts axis-aligned tree models and leaves rotation-invariant linear models unaffected.

- **Rotation effect:** with full PCA (no component dropped), tree models lose 0.070 accuracy on average across three classification datasets. The linear-model gap is −0.0001. The tree gap is positive in all 120 tree-model runs.
- **Signal concentration:** the gap grows as predictive signal is concentrated in fewer features. In a controlled synthetic study it falls from 0.28 (one informative direction) to 0.03 (one hundred). Across the four real datasets the Pearson r is 0.85.
- **Certified selection:** we adapt Learn-Then-Test to pick the smallest number of components whose risk stays within a tolerance s of the same learner on all K components. The selection holds with probability at least 1 − δ. At s = δ = 0.05 the bound was respected in all 20 runs, and the random forest on MNIST needed about 16 of 100 components.

## Repository layout

```
src/
  datasets.py                 dataset loaders and train/calibration/test split
  pca_utils.py                PCA helpers
  ltt.py                      Learn-Then-Test: losses, Bentkus p-value, fixed-sequence test
  experiments/
    _common.py                shared settings (K = 100, s = δ = 0.05, models)
    verify_rotation_v2.py     rotation gap, 10 seeds (Table II, Fig. 3)
    exp_dose.py               gap vs. signal concentration (Fig. 4)
    exp_synth_concentration.py  controlled synthetic study (Fig. 5)
    verify_truncation.py      rotation vs. truncation (Section V-F)
    exp_stab_v2.py            stability of the task-sufficient dimension (Section V-D)
    exp_ltt_v2.py             certified selection with Learn-Then-Test (Table III)
    exp_sanity_v2.py          no-signal and planted controls (Section V-G)
    fig_data_tradeoff_v2.py   compression vs. tolerance s (Section V-E)
    make_figs_camera_ready.py figures 3–6 from the result tables
results/
  tables/                     CSV output of every experiment
  figures/                    figures used in the paper
fig_protocol (1).drawio       Fig. 1 source (draw.io)
fig_ltt_flow (1).drawio       Fig. 2 source (draw.io)
CAMERA_READY.md               map from each paper table/figure to its script
```

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Datasets download automatically on first use and are cached in `./data`:
- Madelon, MNIST and Fashion-MNIST from OpenML
- Superconductivity from the UCI repository (id 464)

MNIST and Fashion-MNIST use random 10,000-example subsets.

## Reproducing the paper

Run each script from the repository root as a module:

```bash
python -m src.experiments.verify_rotation_v2         # Table II, Fig. 3
python -m src.experiments.exp_dose                   # Fig. 4
python -m src.experiments.exp_synth_concentration    # Fig. 5
python -m src.experiments.verify_truncation          # Section V-F
python -m src.experiments.exp_stab_v2                # Section V-D
python -m src.experiments.exp_ltt_v2                 # Table III
python -m src.experiments.exp_sanity_v2              # Section V-G
python -m src.experiments.fig_data_tradeoff_v2       # Section V-E
python -m src.experiments.make_figs_camera_ready     # Figs. 3-6
```

Results are written to `results/tables/` and figures to `results/figures/`. The CSV files from our runs are already included, so `make_figs_camera_ready` can be run on its own.

Runtime on a laptop CPU: most scripts finish in minutes. The Learn-Then-Test experiment with a 200-tree random forest on MNIST takes about 12 minutes per seed.

## Notes

- Scaler and PCA are always fit on the training partition only. Rotation gaps are measured on the calibration partition; the test partition is kept for evaluating the certified selection.
- The reference model for the guarantee is the same learner on all K PCA components, not the model on the raw features.
- `exp_crc.py`, `exp_sanity.py`, `exp_stab.py` and `fig_data_tradeoff.py` produced the numbers in the submitted (pre-review) version and are kept for reference only. The camera-ready numbers come from the `*_v2.py` scripts. See `CAMERA_READY.md`.

## Citation

```bibtex
@inproceedings{sisty2026pcarotation,
  title     = {PCA Rotation in Tree-Based Prediction: A Systematic Study of
               Variance-Oriented Degradation and Certified Task-Sufficient Dimensionality},
  author    = {Sisty, Abu Jafar and Roy, Niloy and Shafi, S M Abdullah and Bhowmik, Abhijit},
  booktitle = {Proc. International Conference on Electrical and Computer Engineering (ICECE)},
  year      = {2026},
  note      = {To appear}
}
```

## Contact

Abu Jafar Sisty, 22-49133-3@student.aiub.edu
