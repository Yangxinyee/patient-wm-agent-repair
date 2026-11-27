`Xf` rows come with code, last bin, observed fine bin (`obf`), lag-1 doses (`dI1`, `dN1`), and the original per-row bin NLL (`nll0`). The `DEC` entries are empty. The edit needs two value-head hidden units that can be repurposed (one per drug) at negligible factual cost. Next I'll verify that `head_logits` reproduces `L0`, then measure the factual NLL change from removing each hidden unit's contribution and list the cheapest ones. A unit costing less than about 0.0005 nats would confirm the repurposing plan is affordable.

```python
Lr = head_logits(Xf, cf, lbf)
print("replica vs L0 max abs diff", np.abs(Lr - L0).max().round(5), "W shapes", W1.shape, b1.shape, W2.shape, b2.shape, TR.shape)
Hh = gelu(Xf @ W1.T + b1)
def nll_from(Lg):
    m_ = Lg.max(1, keepdims=True); lse = m_[:,0] + np.log(np.exp(Lg - m_).sum(1))
    return lse - Lg[np.arange(len(Lg)), obf]
base_nll = nll_from(Lr)
dn = np.zeros(192)
for j_ in range(192):
    dn[j_] = (nll_from(Lr - Hh[:, j_:j_+1] * W2[:, j_][None, :]) - base_nll).mean()
o_ = np.argsort(dn)
print("base nll", base_nll.mean().round(4), "cheapest units", o_[:8], np.round(dn[o_[:8]], 5))
print("unit act sd", np.round(Hh[:, o_[:8]].std(0), 3), "|W2 col|", np.round(np.linalg.norm(W2[:, o_[:8]], axis=0), 3))
```