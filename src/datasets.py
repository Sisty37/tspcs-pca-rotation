"""
Dataset loaders for TS-PCS.
All datasets auto-download on first run and cache locally in ./data.
Split protocol: TRAIN / CALIBRATION / TEST  (calib needed for CRC later).
"""
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_HOME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_HOME, exist_ok=True)


# ---------- individual loaders ----------
def load_mnist(n=10000, seed=0):
    """MNIST, subsampled to n for speed. Returns X (float32), y (int)."""
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                        as_frame=False, data_home=DATA_HOME)
    X, y = X.astype("float32"), y.astype(int)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


def load_madelon(seed=0):
    """Madelon: planted noise + nonlinear features. Good divergence candidate."""
    X, y = fetch_openml("madelon", version=1, return_X_y=True,
                        as_frame=False, data_home=DATA_HOME)
    return X.astype("float32"), y.astype(int)


def load_fashion_mnist(n=10000, seed=0):
    X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True,
                        as_frame=False, data_home=DATA_HOME)
    X, y = X.astype("float32"), y.astype(int)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return X[idx], y[idx]


def load_superconduct():
    """Regression. UCI Superconductivity (id 464)."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=464)
    X = ds.data.features.values.astype("float32")
    y = ds.data.targets.values.ravel().astype("float32")
    return X, y


# ---------- registry ----------
LOADERS = {
    "mnist": (load_mnist, "classification"),
    "madelon": (load_madelon, "classification"),
    "fashion": (load_fashion_mnist, "classification"),
    "superconduct": (load_superconduct, "regression"),
}


def get_dataset(name, seed=0):
    """Returns X, y, task_type."""
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Options: {list(LOADERS)}")
    loader, task = LOADERS[name]
    try:
        X, y = loader(seed=seed)
    except TypeError:  # loaders without a seed arg (regression)
        X, y = loader()
    return X, y, task


# ---------- splitting ----------
def split_train_calib_test(X, y, task, seed=0,
                           calib_size=0.2, test_size=0.2):
    """
    TRAIN / CALIBRATION / TEST split.
    calib_size and test_size are fractions of the WHOLE dataset.
    Stratify for classification only.
    """
    strat = y if task == "classification" else None
    # first peel off TEST
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat)
    # then peel CALIB out of the remaining
    strat2 = y_tr if task == "classification" else None
    calib_rel = calib_size / (1.0 - test_size)  # fraction of the remainder
    X_tr, X_ca, y_tr, y_ca = train_test_split(
        X_tr, y_tr, test_size=calib_rel, random_state=seed, stratify=strat2)
    return X_tr, y_tr, X_ca, y_ca, X_te, y_te


if __name__ == "__main__":
    # quick smoke test
    for name in ["madelon", "mnist"]:
        X, y, task = get_dataset(name, seed=0)
        Xtr, ytr, Xca, yca, Xte, yte = split_train_calib_test(X, y, task)
        print(f"{name:12s} task={task:14s} X={X.shape} "
              f"train={Xtr.shape[0]} calib={Xca.shape[0]} test={Xte.shape[0]}")
