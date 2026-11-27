The head inputs are all there. One detail: there are 16 centres per reading but 17 classes. So class 0 is probably a non-value class, and the mean is either taken over classes 1 to 16 or renormalised. I will compute the offline mean both ways, compare each with the stored prediction and NLL, and keep whichever matches.

```python
import torch.nn.functional as F
W1 = load_original('attr_heads.value_bin.0.weight').numpy().astype(np.float64)
b1 = load_original('attr_heads.value_bin.0.bias').numpy().astype(np.float64)
W2 = load_original('attr_heads.value_bin.2.weight').numpy().astype(np.float64)
b2 = load_original('attr_heads.value_bin.2.bias').numpy().astype(np.float64)
from scipy.special import erf
def gelu(a): return 0.5*a*(1+erf(a/np.sqrt(2)))
def head(wc, W1=W1, b1=b1, idx=slice(None)):
    a = wc@W1.T + b1
    lg = gelu(a)@W2.T + b2 + HI['trans'][idx]
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg); p /= p.sum(1, keepdims=True)
    return p
P = head(WC0)
cen = HI['cen']; ob = HI['ob'].astype(int)
mA = (P[:,1:]*cen).sum(1)
mB = mA / P[:,1:].sum(1)
nll = -np.log(P[np.arange(len(P)), ob])
pr = RD2['pred0'].values; n0 = RD2['nll0'].values
for nm, m in [('A',mA),('B',mB)]:
    print(nm, round(float(np.sqrt(np.nanmean((m-pr)**2))),3))
print('nll', round(float(np.nanmean(nll)),4), round(float(np.nanmean(n0)),4), 'P0diff', float(np.abs(P-P0).max()))
print('p0 class mass', float(P[:,0].mean()))
```