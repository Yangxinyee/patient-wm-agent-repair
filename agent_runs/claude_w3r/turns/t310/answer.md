`HD.wc(ctx, z, code_name, lv, age_s)` takes the code name, and GR holds 1,765 gf rows (state row, gap, last value, age). So I can compute the head input for any measurement code by passing a different code name.

Next, for 300 states I will compute head inputs for all four measurement codes and measure:
- the gate margins along GD;
- the zero-rate baselines of the rate reads;
- whether a GELU re-implementation of the value head matches the model;
- the predictive variances, which set the tilt scale;
- whether the 8 candidate units are inactive.

The design is confirmed if the gf and map gates separate cleanly from the other codes and the baselines have small spread.

```python
import numpy as np, torch
from scipy.special import erf
rng = np.random.default_rng(0)
print('lv', GR['lv'].describe().round(1).to_dict()); print('age', GR['age_s'].describe().round(0).to_dict())
_wc0 = HD.wc; CODE = {'c':'glucose_finger'}
HD.wc = lambda ctx, z, cn, *a, **k: _wc0(ctx, z, CODE['c'], *a, **k)
sub = GR.sample(300, random_state=0).reset_index(drop=True)
tonp = lambda x: x.detach().double().numpy().ravel() if torch.is_tensor(x) else np.asarray(x, float).ravel()
Qc, Tc, RAW = {}, {}, []
for cn in VC:
    CODE['c'] = cn; lo, hi = VC[cn][0], VC[cn][-1]; qs, ts = [], []
    for i, r in sub.iterrows():
        lv = r.lv if cn == 'glucose_finger' else rng.uniform(lo, hi)
        out = wc_full(int(r.prow), r.gap, lv, r.age_s)
        qs.append(tonp(out[0])); ts.append(tonp(out[-1]))
        if cn == 'glucose_finger' and i < 3: RAW.append(out)
    Qc[cn], Tc[cn] = np.array(qs), np.array(ts)
del HD.wc
for g in ['gf','map']:
    print('gate', g, {cn[:9]: tuple(np.round([x.min(), x.mean(), x.max()], 3)) for cn in VC for x in [Qc[cn] @ GD[g]]})
for d in ['ins','ne']:
    print('base', d, {cn[:9]: tuple(np.round([x.mean(), x.std()], 4)) for cn in VC for x in [Qc[cn] @ RD[d]]})
W0 = load_original('attr_heads.value_bin.0.weight').astype(float); b0 = load_original('attr_heads.value_bin.0.bias').astype(float)
W2v = load_original('attr_heads.value_bin.2.weight').astype(float); b2v = load_original('attr_heads.value_bin.2.bias').astype(float)
gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))
UN8 = [31, 74, 103, 114, 124, 171, 173, 181]
for cn in ['glucose_finger', 'map_arterial']:
    pre = Qc[cn] @ W0.T + b0; hid = gelu(pre); lg = hid @ W2v.T + b2v + Tc[cn]
    if cn == 'glucose_finger':
        o = RAW[0]; ref = tonp(HD.logits(HD.hidden(o[0]), o[-1])); print('act check maxdiff', np.abs(ref - lg[0]).max())
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    c = np.array(VC[cn]); pb = p[:, 1:] / p[:, 1:].sum(1, keepdims=True); m = pb @ c; var = (pb * (c[None] - m[:, None])**2).sum(1)
    print(cn, 'p0', p[:, 0].mean().round(4), 'var q10/50/90', np.round(np.quantile(var, [.1, .5, .9]), 1), 'mean', var.mean().round(1),
          'UN8 max|hid|', np.abs(hid[:, UN8]).max().round(5), 'max pre', pre[:, UN8].max().round(2))
print('UN8 |W2 col|', np.round(np.linalg.norm(W2v[:, UN8], axis=0), 3))
with open('/work/notes.md', 'a') as f:
    f.write("- Run 280: HD methods: ctx(out,h,dt), zprior(ctx), wc(ctx,z,code_name,lv,age_s)->(wc,trow), hidden(wc), logits(hid,trow). GR(1765) gf rows: stay,k,prow,gap,lv,age_s.\n"
            "- Run 281: q for 4 codes via code-name override of HD.wc; gate margins, rate baselines, act check, var.\n")
```