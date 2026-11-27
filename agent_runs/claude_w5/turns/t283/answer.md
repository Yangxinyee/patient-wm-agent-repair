`WCUR` (the weights my local functions use) also carries the kv edit, so all my local validation — the dNLL gain and the −1.27 per u/h shift — was done with stage 1 present. The tool never had it. In `SOL`, the keys 3, 5, …, 13 are exactly this stay's insulin bundles, so SOL holds per-bundle u-corrections from my earlier attempt to explain the mismatch. That makes sense now: I was fitting the missing kv effect as if it were a change in u.

Next, on this training stay's six insulin bundles, I'll compute the injected shift in h_new projected on vbar_I (in reading-equivalent units), under x2 and x1/2 insulin and for three sets of weights: original, `WE`, and `WE` without the kv edit (what the tool ran).

- Confirmation: with `WE` the x2 shift is about kappa × rate (−4.6 × rate), while without kv it is weak or offset.
- Refutation: both respond alike, in which case the kv edit is not what matters and the cause lies elsewhere.

```python
import numpy as np, torch
WEt = dict(WE); WEt[kvn] = o_kv.double()
def run_h(W, d):
    global WCUR
    old = WCUR; WCUR = W
    try: r = mystep(d)
    finally: WCUR = old
    return np.asarray(r[5], float)
vI2 = vbar_I @ vbar_I
print('vbar_I norm %.3f' % np.sqrt(vI2))
for j in [3,5,7,9,11,13]:
    d = S_[j]; ats = B_[j][1]
    rI = [a.get('rate') for a in ats if a['code']=='insulin'][0]
    base = {nmW: run_h(W, d).reshape(-1,192).mean(0) for nmW, W in [('orig', W64), ('WE', WE), ('WEt', WEt)]}
    line = f'j{j} rate {rI:5.2f} | factual inj WE {(base["WE"]-base["orig"])@vbar_I/vI2:7.3f} WEt {(base["WEt"]-base["orig"])@vbar_I/vI2:7.3f}'
    for f in [2.0, 0.5]:
        ats2 = ins_atoms_scaled(ats, f); e2, u2 = enc2(ats2)
        d2 = dict(d); d2['u'] = u2
        pr = []
        for nmW, W in [('orig', W64), ('WE', WE), ('WEt', WEt)]:
            h2 = run_h(W, d2).reshape(-1,192).mean(0)
            pr.append((h2-base[nmW])@vbar_I/vI2)
        line += f' | x{f}: orig {pr[0]:6.3f} WE {pr[1]:7.3f} WEt {pr[2]:7.3f}'
    print(line, '| e same:', float(np.abs(np.asarray(e2,float)-np.asarray(d['e'],float)).max()) < 1e-9)
```