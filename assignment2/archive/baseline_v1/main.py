"""Train, evaluate, and visualize both Assignment 2 classifiers."""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayes_net import FEATURE_NAMES, FEATURE_PARENTS, BayesNetClassifier
from naive_bayes import NaiveBayesClassifier


ROOT = Path(__file__).resolve().parent
EXPECTED_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        header = stream.read(4)
        if len(header) != 4 or header[:2] != b"\x00\x00":
            raise ValueError(f"invalid IDX header: {path}")
        dtype_code, ndim = header[2], header[3]
        dtype_map = {0x08: np.uint8, 0x09: np.int8, 0x0B: ">i2", 0x0C: ">i4"}
        if dtype_code not in dtype_map:
            raise ValueError(f"unsupported IDX type {dtype_code:#x}: {path}")
        dimensions = []
        for _ in range(ndim):
            raw_dimension = stream.read(4)
            if len(raw_dimension) != 4:
                raise ValueError(f"truncated IDX dimensions: {path}")
            dimensions.append(struct.unpack(">I", raw_dimension)[0])
        raw = stream.read()
    array = np.frombuffer(raw, dtype=dtype_map[dtype_code])
    if array.size != int(np.prod(dimensions)):
        raise ValueError(f"unexpected IDX payload size: {path}")
    return array.reshape(dimensions)


def find_data_dir(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            ROOT / "assignment2-data" / "mnist",
            ROOT / "assignment2-data" / "assignment2-data" / "mnist",
            ROOT / "mnist",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if all((resolved / name).is_file() for name in EXPECTED_FILES):
            return resolved
    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"MNIST data directory not found. Checked:\n{checked}")


def load_mnist(data_dir: Path):
    X_train = _read_idx(data_dir / EXPECTED_FILES[0])
    y_train = _read_idx(data_dir / EXPECTED_FILES[1]).astype(np.int64)
    X_test = _read_idx(data_dir / EXPECTED_FILES[2])
    y_test = _read_idx(data_dir / EXPECTED_FILES[3]).astype(np.int64)
    expected_shapes = ((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))
    for array, expected in zip((X_train, y_train, X_test, y_test), expected_shapes):
        if array.shape != expected:
            raise ValueError(f"expected MNIST shape {expected}, got {array.shape}")
    return X_train.reshape(-1, 784), y_train, X_test.reshape(-1, 784), y_test


def stratified_split(y: np.ndarray, validation_fraction=1 / 6, seed: int = 1322):
    rng = np.random.default_rng(seed)
    train_parts, validation_parts = [], []
    for class_index in range(10):
        indices = rng.permutation(np.flatnonzero(y == class_index))
        validation_size = max(1, round(indices.size * validation_fraction))
        validation_parts.append(indices[:validation_size])
        train_parts.append(indices[validation_size:])
    return np.concatenate(train_parts), np.concatenate(validation_parts)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(
        true_positive, predicted, out=np.zeros(10), where=predicted != 0
    )
    recall = np.divide(true_positive, actual, out=np.zeros(10), where=actual != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10),
        where=(precision + recall) != 0,
    )
    return {
        "accuracy": float(true_positive.sum() / matrix.sum()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": matrix.tolist(),
    }


def most_confused_pairs(matrix: np.ndarray, limit: int = 5) -> list[dict]:
    mistakes = matrix.copy()
    np.fill_diagonal(mistakes, 0)
    ordered = np.argsort(mistakes.ravel())[::-1]
    pairs = []
    for flat_index in ordered:
        true_label, predicted_label = np.unravel_index(flat_index, mistakes.shape)
        count = int(mistakes[true_label, predicted_label])
        if count == 0 or len(pairs) == limit:
            break
        pairs.append(
            {"true": int(true_label), "predicted": int(predicted_label), "count": count}
        )
    return pairs


def plot_confusion(matrix: np.ndarray, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(10),
        yticks=np.arange(10),
    )
    threshold = matrix.max() * 0.55
    for row in range(10):
        for column in range(10):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_dag(path: Path) -> None:
    labels = ("Y",) + FEATURE_NAMES
    positions = {
        0: (0.5, 0.96),
        1: (0.5, 0.76),
        2: (0.20, 0.53), 3: (0.50, 0.53), 4: (0.80, 0.53),
        5: (0.20, 0.31), 6: (0.50, 0.31), 7: (0.80, 0.31),
        8: (0.20, 0.09), 9: (0.50, 0.09), 10: (0.80, 0.09),
        11: (0.90, 0.76), 12: (0.72, 0.76),
    }
    figure, axis = plt.subplots(figsize=(11, 7.5))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def arrow(
        source: int,
        target: int,
        color: str,
        alpha: float,
        width: float,
        curvature: float = 0.0,
    ):
        axis.annotate(
            "",
            xy=positions[target],
            xytext=positions[source],
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "alpha": alpha,
                "lw": width,
                "shrinkA": 24,
                "shrinkB": 24,
                "connectionstyle": f"arc3,rad={curvature}",
            },
            zorder=1,
        )

    for target in range(1, len(labels)):
        arrow(0, target, "#9aa0a6", 0.35, 1.0)
    for target_feature, parent_feature in enumerate(FEATURE_PARENTS):
        if parent_feature >= 0:
            curvature = 0.0
            if parent_feature == 0:
                curvature = {4: 0.18, 5: 0.30, 6: -0.18}[target_feature]
            arrow(
                parent_feature + 1,
                target_feature + 1,
                "#b4232c",
                0.95,
                2.0,
                curvature,
            )

    for node, label in enumerate(labels):
        x, y = positions[node]
        color = "#861720" if node == 0 else "#0d3b66"
        display = label.replace("_", "\n")
        axis.text(
            x,
            y,
            display,
            ha="center",
            va="center",
            color="white",
            fontsize=9 if node else 13,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "white"},
            zorder=3,
        )
    axis.set_title("Hand-designed Bayesian network", fontsize=16, pad=12)
    axis.text(
        0.01,
        0.985,
        "Gray: class edges    Red: feature dependencies",
        ha="left",
        va="top",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_misclassifications(
    X_test: np.ndarray,
    y_test: np.ndarray,
    naive_prediction: np.ndarray,
    network_prediction: np.ndarray,
    path: Path,
) -> list[int]:
    both_wrong = np.flatnonzero(
        (naive_prediction != y_test) & (network_prediction != y_test)
    )
    one_wrong = np.flatnonzero(
        (naive_prediction != y_test) | (network_prediction != y_test)
    )
    ordered = np.concatenate((both_wrong, one_wrong))
    selected = []
    seen = set()
    for index in ordered:
        if int(index) not in seen:
            selected.append(int(index))
            seen.add(int(index))
        if len(selected) == 12:
            break

    figure, axes = plt.subplots(3, 4, figsize=(8.4, 6.8))
    for axis, index in zip(axes.ravel(), selected):
        axis.imshow(X_test[index].reshape(28, 28), cmap="gray_r", vmin=0, vmax=255)
        axis.set_title(
            f"true {y_test[index]} | NB {naive_prediction[index]} | BN {network_prediction[index]}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in axes.ravel()[len(selected):]:
        axis.axis("off")
    figure.suptitle("Representative misclassifications", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return selected


def evaluate_model(model, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
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
    return prediction, result


def validation_study(X: np.ndarray, y: np.ndarray) -> dict:
    train_indices, validation_indices = stratified_split(y)
    study = {"seed": 1322, "train_size": int(train_indices.size), "validation_size": int(validation_indices.size)}
    for name, model_type, alphas in (
        ("naive_bayes", NaiveBayesClassifier, (0.1, 0.5, 1.0)),
        ("bayes_net", BayesNetClassifier, (0.1, 0.5, 1.0)),
    ):
        scores = []
        for alpha in alphas:
            model = model_type(alpha=alpha).fit(X[train_indices], y[train_indices])
            prediction = model.predict(X[validation_indices])
            scores.append(
                {"alpha": alpha, "accuracy": float(np.mean(prediction == y[validation_indices]))}
            )
        study[name] = scores
    return study


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="directory containing the four MNIST gzip files")
    parser.add_argument("--output-dir", default=str(ROOT / "figures"))
    parser.add_argument(
        "--validation-study",
        action="store_true",
        help="also reproduce the deterministic alpha comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = find_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading MNIST from {data_dir}")
    X_train, y_train, X_test, y_test = load_mnist(data_dir)

    results = {
        "data_dir": str(data_dir),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "random_seed": 1322,
        "selected_hyperparameters": {"naive_bayes_alpha": 0.1, "bayes_net_alpha": 0.5},
    }
    if args.validation_study:
        print("Running training-only validation study...")
        results["validation_study"] = validation_study(X_train, y_train)

    print("Training and evaluating Naive Bayes...")
    naive_prediction, naive_result = evaluate_model(
        NaiveBayesClassifier(alpha=0.1), X_train, y_train, X_test, y_test
    )
    results["naive_bayes"] = naive_result

    print("Training and evaluating Bayesian network...")
    network_prediction, network_result = evaluate_model(
        BayesNetClassifier(alpha=0.5), X_train, y_train, X_test, y_test
    )
    results["bayes_net"] = network_result

    naive_matrix = np.asarray(naive_result["confusion_matrix"])
    network_matrix = np.asarray(network_result["confusion_matrix"])
    plot_confusion(naive_matrix, "Naive Bayes confusion matrix", output_dir / "naive_bayes_confusion_matrix.png")
    plot_confusion(network_matrix, "Bayesian network confusion matrix", output_dir / "bayes_net_confusion_matrix.png")
    plot_dag(output_dir / "bayes_net_structure.png")
    selected = plot_misclassifications(
        X_test,
        y_test,
        naive_prediction,
        network_prediction,
        output_dir / "misclassified_examples.png",
    )
    results["misclassification_example_indices"] = selected

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Naive Bayes accuracy: {naive_result['accuracy']:.4f}")
    print(f"Bayesian network accuracy: {network_result['accuracy']:.4f}")
    print(f"Results written to {metrics_path}")


if __name__ == "__main__":
    main()

