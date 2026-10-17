"""v3 final evaluation: retrain on the full 60,000 training set, evaluate
once on the 10,000-sample test set (single use, after model selection was
completed on validation splits only).  Writes final_test_result.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / ".." / "experiments_v2" / "bayes_net"))

from main import (  # noqa: E402
    classification_metrics, find_data_dir, load_mnist, most_confused_pairs,
)
from bayes_net import BayesNetClassifier  # noqa: E402


def main():
    data_dir = find_data_dir()
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    start = time.perf_counter()
    model = BayesNetClassifier(alpha=1.0).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prediction = model.predict(X_test)
    predict_seconds = time.perf_counter() - start

    result = classification_metrics(y_test, prediction)
    result["fit_seconds"] = fit_seconds
    result["predict_seconds"] = predict_seconds
    result["most_confused_pairs"] = most_confused_pairs(
        np.asarray(result["confusion_matrix"])
    )
    result["config"] = {
        "version": "v3",
        "features": "8 grid densities (bottom_center omitted) + euler + "
        "vertical_runs + aspect_ratio + vertical_dominance",
        "feature_parents": "row chains (tc<-tl, mc<-ml, tr<-tc, mr<-mc, "
        "br<-mr, bl<-br) + semantic chain (vruns<-euler, aspect<-vruns, "
        "vdom<-vruns)",
        "alpha": 1.0,
        "backoff": "class-conditional marginal",
    }
    out = Path(__file__).resolve().parent / "final_test_result.json"
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"accuracy = {result['accuracy']:.4f}")
    print(f"macro_f1 = {result['macro_f1']:.4f}")
    print(f"most confused: {result['most_confused_pairs']}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
