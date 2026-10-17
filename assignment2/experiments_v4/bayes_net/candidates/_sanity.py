"""Unit sanity checks for the connected-component / hole counters."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _features import _components_and_holes, _count_components_runs, _symmetry_fast, _endpoints_junctions  # noqa: E402


def make(rows):
    a = np.zeros((28, 28), dtype=bool)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                a[r, c] = True
    return a


def main():
    # 1. empty
    img = np.zeros((28, 28), dtype=bool)
    assert _count_components_runs(img, False) == 0
    # 2. single dot
    img = np.zeros((28, 28), dtype=bool); img[5, 5] = True
    assert _count_components_runs(img, False) == 1
    # 3. two separated dots -> 2 (4conn and 8conn)
    img = np.zeros((28, 28), dtype=bool); img[5, 5] = True; img[20, 20] = True
    assert _count_components_runs(img, False) == 2
    assert _count_components_runs(img, True) == 2
    # 4. two diagonal dots -> 4conn 2, 8conn 1
    img = np.zeros((28, 28), dtype=bool); img[5, 5] = True; img[6, 6] = True
    assert _count_components_runs(img, False) == 2, _count_components_runs(img, False)
    assert _count_components_runs(img, True) == 1, _count_components_runs(img, True)

    # 5. ring with hole: 3x3 block minus center (8 conn ring)
    ring = np.zeros((28, 28), dtype=bool)
    ring[10:13, 10:13] = True
    ring[11, 11] = False
    b = ring[None]
    c4, c8, h = _components_and_holes(b)
    assert c4[0] == 1, c4
    assert h[0] == 1, h

    # 6. hollow full image border -> 1 component, 1 hole
    frame = np.zeros((28, 28), dtype=bool)
    frame[0, :] = True; frame[-1, :] = True; frame[:, 0] = True; frame[:, -1] = True
    c4, c8, h = _components_and_holes(frame[None])
    assert c4[0] == 1 and h[0] == 1, (c4, h)

    # 7. digit 8-like: two rings stacked
    eight = np.zeros((28, 28), dtype=bool)
    eight[5:13, 8:20] = True; eight[6:12, 9:19] = False
    eight[13:21, 8:20] = True; eight[14:20, 9:19] = False
    c4, c8, h = _components_and_holes(eight[None])
    assert c4[0] == 1, c4
    assert h[0] == 2, h

    # 8. symmetry (full-image mirror): left-only bar is v-asymmetric, h-symmetric
    bar = np.zeros((28, 28), dtype=bool)
    bar[9:19, 11] = True  # vertically centred (mirror of rows 9..18 is 9..18)
    vs, hs = _symmetry_fast(bar[None])
    assert vs[0] < 0.2, vs
    assert hs[0] > 0.99, hs
    # a centered rectangle block is both symmetric
    blk = np.zeros((28, 28), dtype=bool); blk[8:20, 10:18] = True
    vs, hs = _symmetry_fast(blk[None])
    assert vs[0] > 0.99 and hs[0] > 0.99, (vs, hs)

    # 9. endpoints: a straight line has exactly 2 endpoints
    line = np.zeros((28, 28), dtype=bool); line[14, 5:23] = True
    ep, jn = _endpoints_junctions(line[None])
    assert ep[0] == 2, ep
    # T junction: 3 endpoints + 1 junction
    T = np.zeros((28, 28), dtype=bool)
    T[10, 5:23] = True; T[10:20, 13] = True
    ep, jn = _endpoints_junctions(T[None])
    assert ep[0] == 3, ep
    assert jn[0] >= 1, jn

    # 10. cross junction (4 branches)
    X = np.zeros((28, 28), dtype=bool)
    X[14, 5:23] = True; X[5:23, 14] = True
    ep, jn = _endpoints_junctions(X[None])
    assert ep[0] == 4, ep
    assert jn[0] >= 1, jn
    print("all _features sanity checks passed")


if __name__ == "__main__":
    main()
