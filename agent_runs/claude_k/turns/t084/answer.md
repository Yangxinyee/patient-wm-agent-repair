The helper functions were defined in a way that hides their source, but the arrays I need exist. For the 99 states stays I have:
- PRE: value-head pre-activations from my reconstruction;
- TR99: the code × last-bin transition bias;
- CENm: the bin centres of each reading's code;
- mA2 and pmall: my reconstructed and the tool's predicted means;
- err4: the tool's error;
- EI and EN: insulin and norepinephrine shocks at lags 1–4.

That is enough to rebuild the value-head probabilities and measure what freezing each hidden unit at its mean would cost. I will use two measures:
- the second-order (Fisher) cost estimate, which needs no targets;
- the exact change in fine-bin NLL on the observed bins, taking the bin edges from the vocab.

I expect a few units to cost well under 0.0005 nats each. If so, four of them (two per drug, paired so the gate's variation cancels) can be reused within the 0.005 budget. If even the cheapest cost more than about 0.001, I will instead fold the correction into the output layer as a probe on the existing hidden activations.

```python
import numpy as np
from scipy.special import erf as serf
W1 = load_original('attr_heads.value_bin.0.weight').astype(np.float64); b1 = load_original('attr_heads.value_bin.0.bias').astype(np.float64)
W2 = load_original('attr_heads.value_bin.2.weight').astype(np.float64); b2 = load_original('attr_heads.value_bin.2.bias').astype(np.float64)
print("W1", W1.shape, "W2", W2.shape)
def gelu_np(x): return 0.5*x*(1+serf(x/np.sqrt(2)))
def probs_from(Hact, TR):
    lg = Hact @ W2.T + b2 + TR
    lg = lg - lg.max(1, keepdims=True); pp = np.exp(lg); return pp/pp.sum(1, keepdims=True)
Hact99 = gelu_np(PRE.astype(np.float64))
p99 = probs_from(Hact99, TR99)
mean_a = (p99[:,1:]*CENm).sum(1)
mean_b = mean_a / p99[:,1:].sum(1)
print("MAE vs mA2: incl p0 %.3f, renorm %.3f | vs tool: %.3f %.3f | mean p0 %.4f" % (
    np.abs(mean_a-mA2).mean(), np.abs(mean_b-mA2).mean(), np.abs(mean_a-pmall).mean(), np.abs(mean_b-pmall).mean(), p99[:,0].mean()))
print("VOCAB numeric keys:", list(VOCAB['numeric'].keys())[:10] if isinstance(VOCAB['numeric'], dict) else type(VOCAB['numeric']))
k0 = list(VOCAB['numeric'].keys())[0] if isinstance(VOCAB['numeric'], dict) else None
if k0 is not None: print(k0, {kk_: (vv_ if not isinstance(vv_, list) else (len(vv_), vv_[:4])) for kk_, vv_ in VOCAB['numeric'][k0].items()} if isinstance(VOCAB['numeric'][k0], dict) else VOCAB['numeric'][k0])
print("value_bin_note:", VOCAB['value_bin_note'])
print("numeric_note:", VOCAB['numeric_note'])
# second-order cost of freezing each unit at its mean
a = Hact99; cmean = a.mean(0); dev = a - cmean
Ew = p99 @ W2; Ew2 = p99 @ (W2**2); fisher = Ew2 - Ew**2
cost = 0.5*(dev**2*fisher).mean(0)
order_ = np.argsort(cost)
print("cheapest units (unit, cost, act sd, max pre, |W2 col|):")
print([(int(u), float(np.round(cost[u],6)), float(np.round(a[:,u].std(),4)), float(np.round(PRE[:,u].max(),2)), float(np.round(np.linalg.norm(W2[:,u]),2))) for u in order_[:14]])
print("total cost all units sum:", cost.sum().round(4), " median:", np.median(cost).round(5))
```