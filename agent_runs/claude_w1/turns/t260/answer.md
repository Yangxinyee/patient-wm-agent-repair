Run 233 failed because `head_logits` returns a tuple, not a single array. I'll compute the value-head logits directly from W1, b1, W2, b2 and the `embed_transition` rows (code id × 17 + last bin), using the `CODE` map. I'll check the result against `L0`, then run the same per-unit removal cost scan as before.

```python
print("CODE", CODE)
cid4 = np.array([CODE['glucose_finger'], CODE['glucose_lab'], CODE['map_arterial'], CODE['map_cuff']])
Hh = gelu(Xf @ W1.T + b1)
Lr = Hh @ W2.T + b2 + TR[cid4[cf]*17 + lbf]
print("replica vs L0 max abs diff", np.abs(Lr - L0).max().round(5))
def nll_from(Lg):
    m_ = Lg.max(1, keepdims=True); lse = m_[:,0] + np.log(np.exp(Lg - m_).sum(1))
    return lse - Lg[np.arange(len(Lg)), obf]
base_nll = nll_from(Lr)
dn = np.zeros(192)
for j_ in range(192):
    dn[j_] = (nll_from(Lr - Hh[:, j_:j_+1] * W2[:, j_][None, :]) - base_nll).mean()
o_ = np.argsort(dn)
print("base nll", base_nll.mean().round(4), "vs nll0", nll0.mean().round(4), "cheapest", o_[:8], np.round(dn[o_[:8]], 5))
```