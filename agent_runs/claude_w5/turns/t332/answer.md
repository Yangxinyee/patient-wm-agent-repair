**What fwd_11 showed.** The factual cost barely changed: 1.66966 against the original 1.65072, still +0.0189 nats. The per-event losses are almost identical to fwd_10. This time the gain increase touched only the three units that produce my injection, not the rest of the model's state.

**What I conclude.**
- My theory that the b_in rescaling caused collateral damage is refuted.
- The cost comes either from the drug effect itself at 1.5×/1.3× strength, or from the base c=1 edit, which has never been checked on tuning stays. The earlier −0.0032 was measured on training stays.
- With only runs 299 and 300 left, I can no longer explore. Every remaining step must leave a state I can verify and, if needed, still undo.

**What this run does.**
1. Restore the plain c=1 version exactly, using the third-layer backup. b_in is already at c=1.
2. As a read-only diagnostic, locate where fwd_11's extra loss sits: drug stays vs drug-free stays, time since the last drug event, hour of stay, and how concentrated it is in a few stays. This shows whether the loss is a miscalibrated drug effect, which shrinks at c=1, or a leak into bundles without drugs, which would also hurt at c=1.
3. Log both results in the notebook.

**Next check.** A factual pass on tuning stays C for the c=1 version.
- **Confirm:** edited NLL at or below about 1.6527 (within +0.002 of the original), ideally below 1.65072. I would keep c=1 as the final edit. Expected readouts, as in fwd_6: insulin x2 ≈ −9.4, x1/2 ≈ −1.6, NE x2 ≈ +3.2, x1/2 ≈ −0.4.
- **Refute:** above +0.005. The last run would then drop all edits, so the submission at least meets the factual condition.

```python
def _r299():
    import numpy as np, pandas as pd, torch
    as_t = hasattr(load('core.b_in.weight'), 'detach')
    W = np.ascontiguousarray(np.load('/work/ffn2_c1.npy').astype(np.float32))
    save_edit('core.prelude.0.ffn.2.weight', torch.from_numpy(W) if as_t else W)
    cur = load('core.prelude.0.ffn.2.weight')
    cur = cur.detach().cpu().numpy() if hasattr(cur, 'detach') else np.asarray(cur)
    print("restored c=1 ffn2, maxdiff", float(np.abs(cur - W).max()), "| n edited:", len(edited()))
    try:
        fo = pd.read_csv('/work/fwd_11/factual_original.csv'); fe = pd.read_csv('/work/fwd_11/factual_edited.csv')
        cols = list(fo.columns); print("factual cols:", cols[:12], len(fo), len(fe))
        sc = next(c for c in cols if 'stay' in c.lower()); ec = next(c for c in cols if 'event' in c.lower())
        hc = next(c for c in cols if 'hour' in c.lower() or 'time' in c.lower())
        nc = [c for c in cols if 'nll' in c.lower()][0]
        pc = [c for c in cols if 'mean' in c.lower() or 'pred' in c.lower()][0]
        d = fe[nc].values - fo[nc].values
        dm = fe[pc].values - fo[pc].values
        print("cols used:", sc, ec, hc, nc, pc, "| mean dNLL %.4f | mean|dpred| %.3f" % (d.mean(), np.abs(dm).mean()))
        tun = pd.read_csv('/data/tuning.csv')
        print("events:", sorted(tun['event'].astype(str).unique())[:12])
        drug = tun[tun['rate'].notna()]
        dstays = set(drug['stay_id'].unique())
        fl = fo[sc].isin(dstays).values
        print("drug-stay readings %d: dNLL %.4f | non-drug-stay %d: dNLL %.4f" % (
            fl.sum(), d[fl].mean() if fl.any() else np.nan, (~fl).sum(), d[~fl].mean() if (~fl).any() else np.nan))
        for ev in sorted(fo[ec].unique()):
            m = (fo[ec] == ev).values
            print("  %s: drug %.4f (n=%d) nodrug %.4f (n=%d) |dpred| %.2f" % (
                ev, d[m & fl].mean() if (m & fl).any() else np.nan, (m & fl).sum(),
                d[m & ~fl].mean() if (m & ~fl).any() else np.nan, (m & ~fl).sum(), np.abs(dm[m]).mean()))
        dh = {s: np.sort(g['hour'].values.astype(float)) for s, g in drug.groupby('stay_id')}
        since = np.full(len(fo), np.inf)
        st = fo[sc].values; hr = fo[hc].values.astype(float)
        for i in range(len(fo)):
            a = dh.get(st[i])
            if a is not None:
                k = np.searchsorted(a, hr[i], side='right') - 1
                if k >= 0:
                    since[i] = hr[i] - a[k]
        sb = pd.cut(since, [-1e-9, 1, 3, 6, 1e9], labels=['<1h', '1-3h', '3-6h', '>6h'])
        print("by time since drug event:", {str(k): (round(float(v.mean()), 4), int(len(v))) for k, v in pd.Series(d).groupby(sb, observed=False) if len(v)})
        nf = ~np.isfinite(since)
        print("no prior drug event: dNLL %.4f n=%d" % (d[nf].mean() if nf.any() else np.nan, nf.sum()))
        g = pd.Series(d).groupby(fo[sc].values).sum().sort_values(ascending=False)
        print("top5 stays share of total dNLL: %.2f | stays with net harm %d/%d" % (g.head(5).sum() / d.sum(), int((g > 0).sum()), len(g)))
        hb = pd.cut(hr, [-1, 12, 24, 48, 1e9], labels=['0-12', '12-24', '24-48', '48+'])
        print("by hour:", {str(k): round(float(v.mean()), 4) for k, v in pd.Series(d).groupby(hb, observed=False) if len(v)})
    except Exception as e:
        print("diag err", repr(e)[:150])
    with open('/work/notes.md', 'a') as f:
        f.write("- fwd_11 factual C, ffn2-col scaling only (c_I=1.5,c_N=1.3; b_in c=1; hash 2a36120ae6): edited 1.66966 vs orig 1.65072 (+0.0189) -> still FAILS; "
                "harm NOT from b_in collateral; from 1.5x/1.3x injection or base edit on tuning stays.\n"
                "- Run 299: restored pure c=1 (ffn2 from /work/ffn2_c1.npy; b_in=B_IN_c1). Next: factual both C. If > +0.005 -> run 300 drop ALL edits.\n")
_r299()
```