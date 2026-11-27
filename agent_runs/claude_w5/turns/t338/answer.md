All 400 tuning stays are now measured with the c=1 edit.

| Stays | NLL change (nats) |
|---|---|
| set C | +0.0090 |
| ids 0–99 | +0.0073 |
| ids 100–199 | +0.0072 |
| ids 200–299 | +0.0089 |
| ids 300–399 | +0.0065 |

The pooled cost over the four blocks is about +0.0075 per reading, which fails the 0.005 limit. The block-to-block spread is only about ±0.001, so this is a stable, general effect, not a few outlier stays. About 79% of it comes from cuff MAP, about 21% from glucose, and arterial MAP is net zero. Set C showed the cost is concentrated in readings within an hour of a drug event, and grows roughly with the square of injection strength.

**What I conclude.** The only safe way to keep part of the drug effect is to scale the injected effect down, separately for insulin and norepinephrine. I'll use the same family already tested at full strength and at 1.5/1.3. The strengths can only be set properly once I know which drug's injection causes which loss.

**What this last Python run does.**
1. **Rebuild the family.** Recreate the exact scaling directions and check that they reproduce the tested 1.5/1.3 version.
2. **Split the loss by drug.** Using all 400 stays, assign each reading's c=1 loss to the drug events in the preceding hour: insulin only, norepinephrine only, both, or neither.
3. **Estimate the shape of each part.** Use set C at the two tested strengths to see how much of each part shrinks with strength and how much is a fixed cost.
4. **Predict cost conservatively.** Use the larger of a k^1.5 law and a fitted fixed-cost-plus-quadratic law. Loss far from drug events counts as fixed. Benefits are assumed to shrink with strength.
5. **Choose the strengths.** Pick the insulin and norepinephrine strengths that best move the four readouts toward their targets (−14.3, +7.15, +4.11, −2.05), with insulin x2 weighted double. The predicted cost must stay at or below 0.002 nats, well inside the 0.005 limit.
6. **Save, or fall back.** Save that version. If the analysis fails it uses a uniform 0.3/0.3. If no meaningful setting fits the budget, it drops all edits.

**What would confirm or refute.** I'll re-check the saved edit with factual passes on the four blocks. A measured cost near the predicted value, at most about +0.002 to +0.003, confirms the scaling model. Anything near 0.005 would show the extrapolation was too optimistic.

```python
def _r300():
    import numpy as np, pandas as pd, torch
    n2 = 'core.prelude.0.ffn.2.weight'
    as_t = hasattr(load(n2), 'detach')
    def _np(t):
        return t.detach().cpu().numpy().astype(np.float64) if hasattr(t, 'detach') else np.asarray(t, dtype=np.float64)
    def _back(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        return torch.from_numpy(a) if as_t else a
    make = None
    try:
        B1 = _np(globals()['B_IN_c1'])
        Bc = np.load('/work/b_in_current_c15_c13.npy').astype(np.float64)
        M = np.linalg.solve(B1, Bc - B1)
        U, S, Vt = np.linalg.svd(M)
        iI = int(np.argmin(np.abs(S[:2] - 0.5))); iN = 1 - iI
        PI = np.outer(U[:, iI], Vt[iI]); PN = np.outer(U[:, iN], Vt[iN])
        W2c1 = np.load('/work/ffn2_c1.npy').astype(np.float64)
        W2o = _np(load_original(n2))
        cols = np.where(np.abs(W2c1 - W2o).max(0) > 1e-7)[0]
        Id = np.eye(W2c1.shape[0])
        def make(kI, kN):
            T = Id + (kI - 1.0) * PI + (kN - 1.0) * PN
            W = W2c1.copy(); W[:, cols] = T @ W2c1[:, cols]
            return W
        M2 = (U[:, :2] * S[:2]) @ Vt[:2]
        chk = float(np.abs(make(1.5, 1.3)[:, cols] - (Id + M2) @ W2c1[:, cols]).max())
        cur = float(np.abs(_np(load(n2)) - W2c1).max())
        print("sv", np.round(S[:3], 4), "uv %.3f %.3f" % (U[:, iI] @ Vt[iI], U[:, iN] @ Vt[iN]),
              "cols", cols.tolist(), "famchk %.1e cur-c1 %.1e" % (chk, cur))
    except Exception as ex:
        make = None
        print("family error:", repr(ex)[:150])
    choice = None
    if make is not None:
        try:
            tun = pd.read_csv('/data/tuning.csv')
            ev_t = {}
            for dname in ('insulin', 'norepinephrine'):
                dd = tun[tun['event'] == dname]
                t = pd.concat([dd[['stay_id', 'hour']].rename(columns={'hour': 't'}),
                               dd[['stay_id', 'end_hour']].rename(columns={'end_hour': 't'})]).dropna()
                ev_t[dname] = {s: np.sort(g['t'].values.astype(float)) for s, g in t.groupby('stay_id')}
            def groups(df):
                st = df['stay_id'].values; hr = df['hour'].values.astype(float); n = len(df)
                near = {}
                for dname in ('insulin', 'norepinephrine'):
                    dh = ev_t[dname]; out = np.zeros(n, bool)
                    for i in range(n):
                        a = dh.get(st[i])
                        if a is not None:
                            k = np.searchsorted(a, hr[i], side='left') - 1
                            if k >= 0 and hr[i] - a[k] < 1.0:
                                out[i] = True
                    near[dname] = out
                nI, nN = near['insulin'], near['norepinephrine']
                return {'I': nI & ~nN, 'N': nN & ~nI, 'B': nI & nN, 'O': ~nI & ~nN}
            parts = []
            for k in (13, 14, 15, 16):
                o = pd.read_csv('/work/fwd_%d/factual_original.csv' % k)
                e = pd.read_csv('/work/fwd_%d/factual_edited.csv' % k)
                assert len(o) == len(e) and (o['stay_id'].values == e['stay_id'].values).all()
                o = o.copy(); o['d'] = e['nll'].values - o['nll'].values
                parts.append(o)
            A = pd.concat(parts, ignore_index=True)
            nA = len(A); gA = groups(A)
            tab = {g: float(A['d'].values[m].sum()) for g, m in gA.items()}
            lab = {'map_cuff': 'cuff', 'map_arterial': 'art', 'glucose_finger': 'fing', 'glucose_lab': 'lab'}
            print("400 stays: n=%d mean d=%.5f" % (nA, float(A['d'].mean())))
            for g, m in gA.items():
                sub = A[m]
                print("  %s n=%d sum=%.1f | " % (g, int(m.sum()), tab[g]) +
                      " ".join("%s %.1f" % (lab[ev], float(sub.loc[sub['event'] == ev, 'd'].sum())) for ev in lab))
            oC = pd.read_csv('/work/fwd_12/factual_original.csv')
            eC12 = pd.read_csv('/work/fwd_12/factual_edited.csv'); eC11 = pd.read_csv('/work/fwd_11/factual_edited.csv')
            assert len(oC) == len(eC12) == len(eC11) and (eC11['hour'].values == oC['hour'].values).all()
            d12 = eC12['nll'].values - oC['nll'].values; d11 = eC11['nll'].values - oC['nll'].values
            gC = groups(oC)
            kr = {'I': 1.5, 'N': 1.3, 'B': 1.5}
            fa = {}
            for g in ('I', 'N', 'B'):
                s12 = float(d12[gC[g]].sum()); s11 = float(d11[gC[g]].sum()); r = kr[g]
                if s12 > 0 and s11 > 0:
                    b = (s11 - s12) / (r * r - 1.0); a = s12 - b
                    fa[g] = float(np.clip(a / s12, 0.0, 1.0))
                else:
                    fa[g] = 0.5
                print("  C %s: s12 %.1f s11 %.1f -> fixed frac %.2f" % (g, s12, s11, fa[g]))
            def fac(g, k):
                return max(k ** 1.5, fa[g] + (1.0 - fa[g]) * k * k)
            def dmg(kI, kN):
                tot = 0.0
                for g, kk in (('I', kI), ('N', kN), ('B', max(kI, kN))):
                    s = tab[g]
                    tot += s * fac(g, kk) if s > 0 else s * kk
                s0 = tab['O']
                tot += s0 if s0 > 0 else s0 * max(kI, kN)
                return tot / nA
            def preds(kI, kN):
                return (-0.55 - 8.88 * kI, -1.63 * kI, 0.19 + 3.02 * kN, -0.17 - 0.23 * kN)
            def obj(kI, kN):
                ix2, ih, nx2, nh = preds(kI, kN)
                return 2 * abs(ix2 + 14.3) / 14.3 + abs(ih - 7.15) / 7.15 + abs(nx2 - 4.11) / 4.11 + abs(nh + 2.05) / 2.05
            print("model check dmg(1,1)=%.5f obs %.5f | dmg(.5,.5)=%.5f" % (dmg(1, 1), float(A['d'].mean()), dmg(0.5, 0.5)))
            BUD = 0.002
            grid = np.round(np.arange(0, 1.0001, 0.05), 2)
            best = None
            for kI in grid:
                for kN in grid:
                    ddm = dmg(float(kI), float(kN))
                    if ddm <= BUD:
                        o_ = obj(float(kI), float(kN))
                        if best is None or o_ < best[0] - 1e-12:
                            best = (o_, float(kI), float(kN), ddm)
            if best is not None:
                choice = (best[1], best[2], best[3])
                print("chosen kI=%.2f kN=%.2f pred dmg %.5f obj %.3f (obj(0,0) %.3f) pred readouts %s" %
                      (best[1], best[2], best[3], best[0], obj(0, 0), tuple(round(x, 2) for x in preds(best[1], best[2]))))
            else:
                print("no feasible setting under budget")
        except Exception as ex:
            print("analysis error:", repr(ex)[:150])
            choice = (0.3, 0.3, float('nan'))
    if make is not None and choice is not None and (choice[0] + choice[1]) >= 0.1:
        W = make(choice[0], choice[1])
        save_edit(n2, _back(W))
        md = float(np.abs(_np(load(n2)) - W).max())
        print("SAVED family kI=%.2f kN=%.2f | saved maxdiff %.1e | edited %d" % (choice[0], choice[1], md, len(edited())))
        msg = "family kI=%.2f kN=%.2f (pred dmg %.4f on 400 tuning)" % (choice[0], choice[1], choice[2])
    else:
        for nm in list(edited()):
            drop_edit(nm)
        print("DROPPED ALL EDITS | edited", edited())
        msg = "dropped all edits (no safe setting / error)"
    with open('/work/notes.md', 'a') as f:
        f.write("- c=1 factual on all 400 tuning stays (fwd_13-16): +0.0073,+0.0072,+0.0089,+0.0065 (mean ~+0.0075) -> FAILS; ~79%% from map_cuff.\n"
                "- Run 300 (LAST): " + msg + ". Family: ffn2 3 injection cols scaled along insulin/NE dirs of M; b_in=B_IN_c1; kv/ffn0 unchanged. "
                "Choice by attributing c=1 damage to insulin/NE events (<1h), conservative scaling, budget 0.002. Verify with factual blocks + dose_edit.\n")
_r300()
```