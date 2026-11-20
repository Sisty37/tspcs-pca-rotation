import os, warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")
TABDIR = os.path.join(RESULTS, "tables"); os.makedirs(TABDIR, exist_ok=True)
FIGDIR = os.path.join(RESULTS, "figures"); os.makedirs(FIGDIR, exist_ok=True)
K, S, DELTA = 100, 0.05, 0.05


def model_fn(task, name):
    if name == "rf":
        return (RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
                if task == "classification"
                else RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1))
    if task == "classification":
        return LogisticRegression(max_iter=5000)
    return Ridge()
