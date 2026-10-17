"""Round 8: refine the round-7 winner.

Winner (84.97%): drop global-ink density; features =
  9 grid densities (tl,tr,tc,ml,mc,mr,bl,bc,br) + euler(4) + vert_runs(5)
  + vert_dominance(4) + aspect(4), all class-only parents for the extras.

Round 8 sweeps:
  A  alpha and backoff for this set
  B  parent wiring for the 4 semantic features
  C  alternative DAG edges among grid cells (drop-ink changes the chain)
  D  state-count variants for semantic features
  E  swap aspect/vert_dominance with hsym/rot_sym variants
"""

from common import (
    load_split, continuous_stats, grid_densities, disc_zero_sep, fit_eval,
)
import numpy as np


def main():
    X_tr, y_tr, X_va, y_va = load_split()
    s_tr = continuous_stats(X_tr, "train")
    s_va = continuous_stats(X_va, "val")
    G_tr = grid_densities(X_tr)
    G_va = grid_densities(X_va)

    # 9 grid densities (drop global ink at index 0)
    cols_tr, cols_va, cards = [], [], []
    for j in range(1, 10):
        a, b, _ = disc_zero_sep(G_tr[:, j], G_va[:, j], 5)
        cols_tr.append(a); cols_va.append(b); cards.append(5)
    e_tr = np.clip(s_tr["euler"], -1, 2) + 1
    e_va = np.clip(s_va["euler"], -1, 2) + 1
    extra = {"euler": (e_tr.astype(np.int64), e_va.astype(np.int64), 4)}
    for name, card in (("vert_runs", 5), ("aspect", 4), ("vert_dominance", 4),
                       ("hsym", 4), ("rot_sym", 4), ("horiz_runs", 5),
                       ("top_heavy", 4)):
        a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
        extra[name] = (a.astype(np.int64), b.astype(np.int64), card)

    results = []

    def run(tag, names, sem_parents, grid_parents, alpha=1.0, backoff=False,
            swap=None):
        c_tr, c_va, c_cards = list(cols_tr), list(cols_va), list(cards)
        use_extra = dict(extra)
        if swap:
            for name, card in swap:
                a, b, _ = disc_zero_sep(s_tr[name], s_va[name], card)
                use_extra[name] = (a.astype(np.int64), b.astype(np.int64), card)
        for name in names:
            a, b, card = use_extra[name]
            c_tr.append(a); c_va.append(b); c_cards.append(card)
        parents = list(grid_parents) + list(sem_parents)
        F_tr = np.stack(c_tr, axis=1).astype(np.int64)
        F_va = np.stack(c_va, axis=1).astype(np.int64)
        acc = fit_eval(F_tr, y_tr, F_va, y_va, tuple(c_cards), tuple(parents),
                       alpha=alpha, backoff=backoff)
        results.append((tag, acc))
        print(f"{tag:66s} val acc = {acc:.4f}")
        return acc

    # grid parents: original chain was (-1,4,5,6,0,0,0,4,5,6) over
    # [ink,tl,tr,tc,ml,mc,mr,bl,bc,br]; dropping ink and reindexing:
    # new order: tl(0),tr(1),tc(2),ml(3),mc(4),mr(5),bl(6),bc(7),br(8)
    # old->new: tl=0,tr=1,tc=2,ml=3,mc=4,mr=5,bl=6,bc=7,br=8
    # original: tl<-4? original parents tuple over 10 feats:
    #   idx0 ink:-1, 1 tl:4(ml? no) ... original FEATURE_PARENTS=(-1,4,5,6,0,0,0,4,5,6)
    #   means: tl<-ml? Let's decode: positions are (ink,tl,tr,tc,ml,mc,mr,bl,bc,br)
    #   tl(1)<-4(ml)? that seems odd... Actually the baseline grid order is
    #   row-major: idx1=tl,2=tc,3=tr,4=ml,5=mc,6=mr,7=bl,8=bc,9=br.
    #   parents: tl<-ml, tc<-mc, tr<-mr, ml<-ink, mc<-ink, mr<-ink,
    #   bl<-ml, bc<-mc, br<-mr.  A column-wise chain design.
    # Drop ink: ml,mc,mr lose their parent; column chains:
    #   tc<-mc, tl<-ml, tr<-mr, bl<-ml, bc<-mc, br<-mr; ml/mc/mr class-only.
    GRID9 = (-1, 4, 5, -1, -1, -1, 3, 4, 5)  # over (tl,tr,tc,ml,mc,mr,bl,bc,br)
    NAMES = ("euler", "vert_runs", "vert_dominance", "aspect")

    # A: alpha/backoff, class-only semantics
    for alpha in (0.5, 1.0, 2.0):
        run(f"A drop-ink winner alpha={alpha}", NAMES, (-1, -1, -1, -1),
            GRID9, alpha)
    run("A drop-ink winner backoff a=1", NAMES, (-1, -1, -1, -1), GRID9,
        backoff=True)
    run("A drop-ink winner backoff a=0.5", NAMES, (-1, -1, -1, -1), GRID9,
        alpha=0.5, backoff=True)

    # B: semantic wiring (indices: 9=euler,10=vruns,11=vdom,12=aspect)
    run("B aspect<-vruns, vdom<-aspect", NAMES, (-1, -1, 12, 10), GRID9)
    run("B vdom<-tc(2)", NAMES, (-1, -1, 2, -1), GRID9)
    run("B aspect<-mc(4)", NAMES, (-1, -1, -1, 4), GRID9)
    run("B vruns<-euler, vdom<-aspect", NAMES, (-1, 9, 12, -1), GRID9)
    run("B vdom<-mc, aspect<-mc", NAMES, (-1, -1, 4, 4), GRID9)

    # C: alternative grid chains
    # row-wise snake: tl<-tc, tr<-tc, ml<-tl? design: row-major chain
    ROWS = (-1, -1, 0, 1, 2, 3, 4, 5, -1, 0, 1)  # placeholder not used
    # col chains reversed: ml<-tl, mc<-tc, mr<-tr, bl<-ml, bc<-mc, br<-mr
    #   i.e. cell <- above cell
    COLREV = (-1, -1, -1, 0, 2, 1, 3, 4, 5)
    run("C col-reversed chain", NAMES, (-1, -1, -1, -1), COLREV)
    # center hub: tl,tr,ml,mr<-mc etc is hard within 1 parent; try:
    #   tl<-ml, tr<-mr, tc<-mc, bl<-bc? keep A. try all class-only grid
    NOEDGE = (-1,) * 9
    run("C no grid edges", NAMES, (-1, -1, -1, -1), NOEDGE)

    # D: state counts for semantics
    for vcard, acard in ((5, 4), (4, 5), (5, 5), (3, 4)):
        run(f"D vruns={vcard} aspect={acard}", NAMES, (-1, -1, -1, -1), GRID9,
            swap=[("vert_runs", vcard), ("aspect", acard)])

    # E: swap semantics
    run("E hsym replaces aspect", ("euler", "vert_runs", "vert_dominance",
        "hsym"), (-1, -1, -1, -1), GRID9)
    run("E rot_sym replaces aspect", ("euler", "vert_runs", "vert_dominance",
        "rot_sym"), (-1, -1, -1, -1), GRID9)
    run("E horiz_runs replaces vert_runs", ("euler", "horiz_runs",
        "vert_dominance", "aspect"), (-1, -1, -1, -1), GRID9)
    run("E top_heavy replaces vert_dominance", ("euler", "vert_runs",
        "top_heavy", "aspect"), (-1, -1, -1, -1), GRID9)


if __name__ == "__main__":
    main()
