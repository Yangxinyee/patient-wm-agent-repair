Several units come out with negative ablation cost at z = prior mean. Mean-ablating unit 9 lowers offline NLL by 0.001, and units 185, 189 and 143 by about 0.0005 each. The median unit costs only 0.0002.

An improvement from deleting units on training stays is suspicious. It probably means these units are tuned to the four-draw z sampling the model uses at evaluation, so the deterministic z = mean check misjudges them.

I am redoing the ablation costs under four prior z draws with averaged probabilities, which is how the tool scores. Units that stay near zero cost there are safe to repurpose. If the cheapest ones turn positive (about 0.0005 each), repurposing four units would use a noticeable part of the 0.005 budget.

```python
rng = np.random.default_rng(7)
WCd = []
for d in range(4):
    z = HI['zmu'] + np.exp(HI['zls'])*rng.standard_normal(HI['zmu'].shape)
    WCd.append(wc_from(z))
WCd = np.stack(WCd)                       # 4,n,192
Hd = gelu(WCd@W1.T + b1)
lgd = Hd@W2.T + b2 + HI['trans'][None]
def nll_avg(lgd):
    l = lgd - lgd.max(2, keepdims=True); p = np.exp(l); p /= p.sum(2, keepdims=True)
    pa = p.mean(0)
    return -np.log(pa[np.arange(pa.shape[0]), ob] + 1e-300)
nb = nll_avg(lgd); base4 = nb.mean()
mh4 = Hd.mean((0,1)); cost4 = np.zeros(192)
for j in range(192):
    cost4[j] = nll_avg(lgd - (Hd[:,:,j]-mh4[j])[:,:,None]*W2[:,j][None,None,:]).mean() - base4
o4 = np.argsort(cost4)
print('base4', round(float(base4),4), 'cheap', [(int(j), round(float(cost4[j]),5)) for j in o4[:8]], 'median', round(float(np.median(cost4)),5))
print('units 9,185,189,143 cost4', [round(float(cost4[j]),5) for j in (9,185,189,143)])
```