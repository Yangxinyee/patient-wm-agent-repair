The run failed at once: `load_original` returns torch tensors, not numpy arrays, so nothing was computed. I am rerunning the same analysis unchanged except for converting each tensor with `.numpy()`. The question and the expected outcome are as before:
- If a few units cost well under 0.0005 nats each, I reuse four of them.
- If even the cheapest cost more than about 0.001 nats, I switch to an output-layer probe.

```python
import numpy as np
from scipy.special import erf as serf
def L64(n_): return np.asarray(load_original(n_).numpy(), dtype=np.float64)
W1 = L64('attr_heads.value_bin.0.weight'); b1 = L64('attr_heads.value_bin.0.bias')
W2 = L64('attr_heads.value_bin.2.weight'); b2 = L64('attr_heads.value_bin.2.bias')
print("W1", W1.shape, "W2", W2.shape)
def gelu_np(x): return 0.5*x*(1+serf(x/np.sqrt(2)))
def probs_from(Hact, TR):
    lg = Hact @ W2.T + b2 + TR
    lg = lg - lg.max(1, keepdims=True); pp = np.exp(lg); return pp/pp.sum(1, keepdims=True)
Hact99 = gelu_np(np.asarray(PRE, dtype=np.float64))
p99 = probs_from(Hact99, np.asarray(TR99, dtype=np.float64))
mean_a = (p99[:,1:]*CENm).sum(1)
mean_b = mean_a / p99[:,1:].sum(1)
print("MAE vs mA2: incl p0 %.3f, renorm %.3f | vs tool: %.3f %.3f | mean p0 %.4f" % (
    np.abs(mean_a-mA2).mean(), np.abs(mean_b-mA2).mean(), np.abs(mean_a-pmall).mean(), np.abs(mean_b-pmall).mean(), p99[:,0].mean()))
num_ = VOCAB['numeric']
print("VOCAB numeric type:", type(num_).__name__, (list(num_.keys())[:10] if isinstance(num_, dict) else len(num_)))
k0 = list(num_.keys())[0] if isinstance(num_, dict) else 0
ent = num_[k0]
print(k0, {kk_: (vv_ if not isinstance(vv_, list) else (len(vv_), vv_[:4])) for kk_, vv_ in ent.items()} if isinstance(ent, dict) else ent)
print("value_bin_note:", VOCAB['value_bin_note'])
print("numeric_note:", VOCAB['numeric_note'])
a = Hact99; cmean = a.mean(0); dev = a - cmean
Ew = p99 @ W2; Ew2 = p99 @ (W2**2); fisher = Ew2 - Ew**2
cost = 0.5*(dev**2*fisher).mean(0)
order_ = np.argsort(cost)
print("cheapest units (unit, cost, act sd, max pre, |W2 col|):")
print([(int(u), float(np.round(cost[u],6)), float(np.round(a[:,u].std(),4)), float(np.round(PRE[:,u].max(),2)), float(np.round(np.linalg.norm(W2[:,u]),2))) for u in order_[:14]])
print("total cost all units sum:", cost.sum().round(4), " median:", np.median(cost).round(5))
```