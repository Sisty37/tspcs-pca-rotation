# Camera-ready version (ICECE 2026, paper 924)

The numbers in the camera-ready paper come from the scripts below. They implement
Learn-Then-Test exactly as described in Section III-D of the paper.

| Paper item | Script | Output |
|---|---|---|
| Table III (LTT selection, coverage, excess risk, timing) | `src/experiments/exp_ltt_v2.py` | `results/tables/exp_ltt_v2.csv` |
| Section V-G (no-signal and planted controls) | `src/experiments/exp_sanity_v2.py` | `results/tables/exp_sanity_v2.csv` |
| Section V-D (stability of the task-sufficient dimension) | `src/experiments/exp_stab_v2.py` | `results/tables/exp_stab_v2.csv` |
| Section V-E (compression vs. tolerance s) | `src/experiments/fig_data_tradeoff_v2.py` | `results/tables/tradeoff_v2.csv` |
| Table II, Fig. 3 (rotation gap) | `src/experiments/verify_rotation_v2.py` | `results/tables/verify_rotation_v2.csv` |
| Figs. 3–6 | `src/experiments/make_figs_camera_ready.py` | `results/figures/*.pdf` |

Shared LTT code (paired excess-loss indicator, regression threshold, Bentkus p-value,
fixed-sequence test) is in `src/ltt.py`. Settings: s = δ = 0.05, K = 100
(81 for Superconductivity).

## Run order

From the repository root, with the virtual environment active:

```
python -m src.experiments.exp_ltt_v2
python -m src.experiments.exp_sanity_v2
python -m src.experiments.exp_stab_v2
python -m src.experiments.fig_data_tradeoff_v2
python -m src.experiments.make_figs_camera_ready
```

## Superseded scripts

`exp_crc.py`, `exp_sanity.py`, `exp_stab.py` and `fig_data_tradeoff.py` produced the
numbers in the submitted (pre-review) version. They used the marginal risk against the
full model, a calibration-set regression threshold and a multiplicative stability rule,
which do not match the method as described in the paper. They are kept for reference
only; use the `*_v2.py` scripts above.
