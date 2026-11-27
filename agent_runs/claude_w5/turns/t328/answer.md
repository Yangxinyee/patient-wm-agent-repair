Run 296 tested two exact-equivalent rescalings meant to make the edit less sensitive to low-precision arithmetic. Both changed the state at non-drug steps by about 7 units, so they are not equivalent, and I saved neither. The saved edit is still c_I=1.5, c_N=1.3, the version tested in fwd_9.

**What I conclude:** With 4 Python runs left, I will stop trying to explain why the tool's x1/2 rollouts fail. The x2 responses match the design (insulin −14.6, NE +4.0), so the main readout is on target. Changing the gains alone cannot fix x1/2.

Two things still matter:
1. **Factual check of the current version.** The factual fit has only been checked for c=1 (−0.0032). The larger gains could change that, and this is a hard constraint.
2. **Replication on new stays.** All dose_edit results so far come from one set of 100 stays (selA). I want to see whether the four readouts hold on stays the edit has never been tested on.

**This run:** It makes no edit and does not touch the model state. It:
- backs up the current b_in;
- pulls the target lines from my notes to confirm the targets;
- checks how the c=1 and current b_in differ, in case I need to retune c later;
- picks two new sets of tuning stays not in selA (C and D, preferring stays with insulin and/or norepinephrine in hours 12–24) and prints their ids compactly;
- logs the plan.

Everything runs inside a function so no kernel globals get overwritten.

**What would confirm or refute the current belief:**
- Confirm: the factual NLL difference on C stays well under +0.005, and set C gives roughly insulin x2 ≈ −14, x1/2 slightly negative, NE x2 ≈ +4, NE x1/2 ≈ −0.4.
- Refute: a factual cost above about 0.003, or an x2 response far from target. Then I would retune c in run 298, using B_IN_c1 and the backup.

```python
def _r297():
    import json, re
    import numpy as np, pandas as pd
    def _np(t):
        return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.asarray(t)
    print("edited:", edited())
    cur = _np(load('core.b_in.weight')).astype(np.float32)
    np.save('/work/b_in_current_c15_c13.npy', cur)
    try:
        B1 = _np(globals()['B_IN_c1']).astype(np.float64)
        Dd = cur.astype(np.float64) - B1
        print("b_in", cur.shape, "|D|", round(float(np.linalg.norm(Dd)), 4),
              "nzcols", int((np.abs(Dd).sum(0) > 1e-9).sum()), "nzrows", int((np.abs(Dd).sum(1) > 1e-9).sum()))
    except Exception as e:
        print("B_IN_c1 check failed:", repr(e)[:100])
    qn = [k for k in list(globals()) if re.fullmatch(r'q_?[IN]\w{0,6}', k)]
    print("q vars:", qn[:10])
    txt = open('/work/notes.md').read()
    tl = [l.strip() for l in txt.splitlines() if re.search(r'target', l, re.I)]
    print("target lines:", len(tl))
    for l in tl[:8]:
        print("  ", l[:170])
    def _ids(x):
        if isinstance(x, dict):
            for k in ('stays', 'ids', 'stay_ids', 'stay'):
                if k in x:
                    return list(x[k])
            v = list(x.values())
            return list(v[0]) if v and isinstance(v[0], (list, tuple)) else list(x.keys())
        return list(x)
    selA = _ids(json.load(open('/work/sel_tuning_A.json')))
    tun = pd.read_csv('/data/tuning.csv')
    cols = list(tun.columns)
    print("tuning cols:", cols[:10], "rows", len(tun))
    scol = next(c for c in cols if 'stay' in c.lower())
    is_int = np.issubdtype(tun[scol].dtype, np.integer)
    allT = list(pd.unique(tun[scol]))
    sA = set(map(str, selA))
    pool = [s for s in allT if str(s) not in sA]
    print("stays", len(allT), "selA", len(selA), "pool", len(pool), "int ids", bool(is_int), "ex", [str(s) for s in allT[:3]])
    rng = np.random.default_rng(297)
    try:
        ecol = next(c for c in cols if tun[c].dtype == object and tun[c].astype(str).str.contains('insulin', case=False).any())
        num = [c for c in cols if c != scol and np.issubdtype(tun[c].dtype, np.number)]
        tcol = [c for c in num if re.search(r'hour|time|^t$|^t_', c, re.I)][0]
        tt = tun[tcol].astype(float)
        if tt.max() > 1000:
            tt = tt / 60.0
        print("ecol", ecol, "tcol", tcol, "tmax(h)", round(float(tt.max()), 1))
        ev = tun[ecol].astype(str)
        win = (tt >= 12) & (tt < 24)
        ins = set(map(str, tun.loc[ev.str.contains('insulin', case=False) & win, scol]))
        ne = set(map(str, tun.loc[ev.str.contains('norepi', case=False) & win, scol]))
        print("selA with ins", sum(str(s) in ins for s in selA), "with ne", sum(str(s) in ne for s in selA))
        b = [s for s in pool if str(s) in ins and str(s) in ne]
        i1 = [s for s in pool if str(s) in ins and str(s) not in ne]
        n1 = [s for s in pool if str(s) in ne and str(s) not in ins]
        rest = [s for s in pool if str(s) not in ins and str(s) not in ne]
        print("pool both", len(b), "ins only", len(i1), "ne only", len(n1), "none", len(rest))
        order = [b[i] for i in rng.permutation(len(b))] + [i1[i] for i in rng.permutation(len(i1))] + \
                [n1[i] for i in rng.permutation(len(n1))] + [rest[i] for i in rng.permutation(len(rest))]
    except Exception as e:
        print("fallback:", repr(e)[:120])
        order = [pool[i] for i in rng.permutation(len(pool))]
    conv = (lambda s: int(s)) if is_int else (lambda s: str(s))
    C = [conv(s) for s in order[:100]]
    D = [conv(s) for s in order[100:200]]
    json.dump(C, open('/work/sel_tuning_C.json', 'w'))
    json.dump(D, open('/work/sel_tuning_D.json', 'w'))
    print("C", len(C), json.dumps(C, separators=(',', ':')))
    print("D", len(D), json.dumps(D, separators=(',', ':')))
    with open('/work/notes.md', 'a') as f:
        f.write("\n## Restart 8 (run 297)\n"
                "- Run 296: x-gain variants not exact at non-drug steps -> nothing saved. Saved state = c_I=1.5, c_N=1.3 (fwd_9, hash a56a7f6079).\n"
                "- b_in backup /work/b_in_current_c15_c13.npy; B_IN_c1 (kernel) = c=1 version.\n"
                "- Decision: stop debugging tool x1/2 asymmetry (x2 matches design, x1/2 fails; cause unknown; only 4 runs left).\n"
                "- Plan: factual both + dose_edit both on fresh tuning set C (/work/sel_tuning_C.json), then D; retune c only if needed; then FINAL.\n")
_r297()
```