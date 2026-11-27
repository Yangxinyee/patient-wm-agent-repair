The exact readout of insulin rate from LN1(u) works: error at most 0.0004 for insulin alone and 0.01 for NE combinations, zero at the no-action token. But its weight norm is 7,878. Head 1 puts about 0.2% of its attention on the other tokens, so a vector that large would leak noise from the slot tokens. The fix is to fit the readout on the attention-weighted mix that each slot actually receives, with a ridge penalty.

std(x1) is about 62 at drug bundles and 41 at measurement bundles, with only about 9% spread within each type and identical across slots. That is acceptable.

Before fitting, I need FFN units to repurpose. I'll count prelude FFN units that never activate on slot tokens (max pre-activation below −3 at all drug and measurement bundles), and rank the rest by their largest output contribution.

```python
pre_d = r2['pre'][:, :4, :].reshape(-1,768).numpy(); pre_n = r2n['pre'][:, :4, :].reshape(-1,768).numpy()
pmax = np.maximum(pre_d.max(0), pre_n.max(0))
W2p = load('core.prelude.0.ffn.2.weight').numpy()
gelu_np = lambda z: 0.5*z*(1+scipy.special.erf(z/np.sqrt(2))) if False else z*0.5*(1+np.vectorize(math.erf)(z/np.sqrt(2)))
import math, scipy.special
hid_d = r2['hid'][:, :4, :].reshape(-1,768).numpy(); hid_n = r2n['hid'][:, :4, :].reshape(-1,768).numpy()
contrib = np.linalg.norm(W2p,axis=0) * np.maximum(np.abs(hid_d).max(0), np.abs(hid_n).max(0))
print('dead units (max pre < -3):', (pmax < -3).sum(), ' (< -2):', (pmax < -2).sum())
o_ = np.argsort(contrib)
print('lowest contrib units', o_[:12].tolist())
print('their contrib', np.round(contrib[o_[:12]],4).tolist())
print('their max pre', np.round(pmax[o_[:12]],2).tolist())
print('median contrib %.3f' % np.median(contrib))
```