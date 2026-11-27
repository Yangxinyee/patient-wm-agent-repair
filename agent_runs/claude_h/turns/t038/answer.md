tensors.json only lists shapes, so the categorical row meanings are not documented. Rather than guess, I will measure what I need directly.

Next I will compute the dose-dependent part of each drug atom's embedding, D = W_num[:,z_dose]·z_dose + W_num[:,z_rate]·z_rate + B_dose[bin] + B_rate[bin], for every insulin and norepinephrine record in training. I will compare
- its within-code spread,
- the gap between the insulin and norepinephrine means,
- the size of other embedding terms (the code embeddings).

I will also inspect the row norms of the categorical tables to see which rows look used. If the mean gap between codes is small, a single shared replacement may be enough. If it is large, I need a code-specific compensation term.

```python
W = load('embed.numeric.weight').numpy() if hasattr(load('embed.numeric.weight'),'numpy') else np.asarray(load('embed.numeric.weight'))
def L(n):
    t = load(n); return t.numpy() if hasattr(t,'numpy') else np.asarray(t)
W = L('embed.numeric.weight'); Bd = L('embed.bins.dose.weight'); Br = L('embed.bins.rate.weight'); Ec = L('embed.code.weight')
num = {(x['code'],x['attribute']):x for x in V['numeric']}
def zb(x, code, attr):
    m = num[(code,attr)]; z = np.clip((x-m['median'])/m['iqr'], -10, 10)
    e = np.array(m['fine_bin_edges']); b = 1 + (e[None,:] < x[:,None]).sum(1)
    return z, b
out={}
for code in ['insulin','norepinephrine']:
    sub = tr[tr.event==code]
    zr, br = zb(sub.rate.values, code, 'rate'); zd, bd = zb(sub.dose.values, code, 'dose')
    D = W[:,2][None,:]*zd[:,None] + W[:,4][None,:]*zr[:,None] + Bd[bd] + Br[br]
    out[code] = D
    print(code, "n", len(D), "mean|D| norm", np.linalg.norm(D.mean(0)).round(3), " within-code rms dev", np.sqrt(((D-D.mean(0))**2).sum(1).mean()).round(3),
          " bins used r:", np.bincount(br, minlength=17)[1:], " z range", zr.min().round(2), zr.max().round(2))
print("gap between code means:", np.linalg.norm(out['insulin'].mean(0)-out['norepinephrine'].mean(0)).round(3))
print("norm code emb ins/ne/glu:", np.linalg.norm(Ec[8]).round(3), np.linalg.norm(Ec[9]).round(3), np.linalg.norm(Ec[6]).round(3), " gap ins-ne:", np.linalg.norm(Ec[8]-Ec[9]).round(3))
print("numeric col norms:", np.linalg.norm(W,axis=0).round(3))
for n in ['embed.categorical.rate_unit.weight','embed.categorical.unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.status.weight','embed.kind.weight','embed.ekind.weight','embed.bins.dose.weight','embed.bins.rate.weight']:
    print(n, np.linalg.norm(L(n),axis=1).round(2))
```