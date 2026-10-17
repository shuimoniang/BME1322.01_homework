"""Guide DAG design with training-only conditional MI (no validation search).

The coordinate-ascent DAG showed the headroom is in the structure, not the
feature set.  This script computes, on TRAINING data only,
  (a) I(Fi;Y) and I(Fi;Fj|Y) for the 12 incumbent features, and
  (b) for each node the partner with the highest I(Fi;Fj|Y)
so a *manual* edge choice can be justified by a stated criterion
("condition a region on the region it is most conditionally dependent with
given the digit class").  No validation accuracy is used here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import build_cache, split_columns  # noqa: E402
from _harness import DISCRETISERS, mutual_information  # noqa: E402
from _sweep2_cells import INCUMBENT_KEEP  # noqa: E402

DESIGN = [(c, 5, "zero_quantile") for c in INCUMBENT_KEEP] + [
    ("euler", 4, "euler_incumbent"), ("vertical_runs", 5, "zero_quantile"),
    ("aspect_incumbent", 4, "zero_quantile"), ("vertical_dominance", 4, "zero_quantile")]
NAMES = [d[0] for d in DESIGN]


def cond_mi(a, b, y, na, nb):
    total = 0.0
    for c in range(10):
        mask = y == c
        if mask.sum() < 2:
            continue
        joint = np.zeros((na, nb), dtype=np.float64)
        np.add.at(joint, (a[mask], b[mask]), 1)
        joint /= joint.sum()
        pa = joint.sum(axis=1, keepdims=True)
        pb = joint.sum(axis=0, keepdims=True)
        m = joint > 0
        total += mask.mean() * float(np.sum(joint[m] * np.log(joint[m] / (pa * pb)[m])))
    return total


def main():
    feats = build_cache()
    ctr, ytr, cva, yva = split_columns(feats, 1322)
    states, cards = {}, {}
    for key, k, disc in DESIGN:
        a, _, _ = DISCRETISERS[disc](np.asarray(ctr[key], np.float64),
                                     np.asarray(cva[key], np.float64), k)
        states[key] = a
        cards[key] = k

    print("I(F;Y) (train):")
    iy_rank = sorted(((mutual_information(states[k], ytr, cards[k]), k) for k in NAMES),
                     reverse=True)
    for v, k in iy_rank:
        print(f"  {k:20s} {v:.4f}")

    print("\nI(Fi;Fj|Y) matrix (train):")
    print(f"{'':20s}" + "".join(f"{n[:6]:>8s}" for n in NAMES))
    M = np.zeros((12, 12))
    for i, a in enumerate(NAMES):
        for j, b in enumerate(NAMES):
            M[i, j] = cond_mi(states[a], states[b], ytr, cards[a], cards[b])
        print(f"{a:20s}" + "".join(f"{M[i, j]:8.3f}" for j in range(12)))

    print("\nFor each node, its strongest conditional-MI partner (train):")
    for i, a in enumerate(NAMES):
        order = np.argsort(M[i])[::-1]
        partners = [(NAMES[j], M[i, j]) for j in order if j != i][:3]
        print(f"  {a:20s} -> " + ", ".join(f"{p}({v:.3f})" for p, v in partners))

    print("\nSymmetry check of the matrix (is the dependence symmetric?):")
    asym = np.abs(M - M.T)
    print(f"  max |I(i;j|Y) - I(j;i|Y)| = {asym.max():.4f}")


if __name__ == "__main__":
    main()
