The vocabulary pins down the head's inputs:

- **Dose features are linear in rate.** Insulin: z = (rate − 1.5)/1.0. NE: z = (rate − 0.46)/0.59. Each also has a 16-bin fine code.
- **NE x2 leaves the training range.** Recorded NE rates are capped at 1.0, so doubled NE rates (up to 2.0) land in rate bin 16 and at z values (up to 2.6) never seen for NE. The model's NE response to x2 is pure extrapolation; I need to keep this in mind.
- **Glucose and MAP are 16 coarse bins**, with the predicted mean = Σ p(b)·centre[b−1].
- **Codes:** 5 map_cuff, 6 glucose_finger, 7 map_arterial, 8 insulin, 9 norepinephrine, 10 glucose_lab. The transition table row is code·17 + last bin.

Next, I will build a local replica of the value head from the saved states. It takes the previous bundle's out and h, time-flowed by dt, plus the gap embedding and a z prior draw, to form the view. It then adds the type embedding, query norm, code embedding, the last-value register and the transition offset. I'll write it to /work/head.py so it survives restarts.

Two details are unknown: the type index for measurements, and the column order of `last_numeric`'s 8 inputs. I'll start with the documented order and scan the type index. Validation is against fwd_2's 5306 predicted means. A correlation above 0.99 with slope near 1 confirms the replica; a poor match means some input is wrong.

```python
code = r'''
import numpy as np, json
CID = {c["name"]: c["id"] for c in VOCAB["codes"]}
NUM = {(n["code"], n["attribute"]): n for n in VOCAB["numeric"]}
CEN = {k: np.array(v) for k, v in VOCAB["value_bin_centres"].items()}
def Tn(name): return load(name).numpy().astype(np.float64)
def lnorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b
def actf(x, kind="relu"):
    if kind == "relu": return np.maximum(x, 0)
    if kind == "gelu":
        from scipy.special import erf
        return 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return x / (1 + np.exp(-x))
def zsc(code, attr, x):
    n = NUM[(code, attr)]; return np.clip((x - n["median"]) / n["iqr"], -10, 10)
def fbin(code, attr, x):
    e = np.array(NUM[(code, attr)]["fine_bin_edges"]); x = np.asarray(x, float)
    return np.where(np.isnan(x), 0, 1 + (e[None, :] < x[:, None]).sum(1))
def mlp2(x, pre, kind="relu"):
    return actf(x @ Tn(pre + ".0.weight").T + Tn(pre + ".0.bias"), kind) @ Tn(pre + ".2.weight").T + Tn(pre + ".2.bias")
'''
open('/work/head.py', 'w').write(code); exec(code)
fa = Tn('core.flow.a'); print("flow.a range", fa.min(), fa.max(), "timescales h:", (np.exp(-fa)/3600).min(), (np.exp(-fa)/3600).max())
print("last_numeric col norms", np.round(np.linalg.norm(Tn('last_numeric.weight'), axis=0), 2))
print("embed.numeric col norms", np.round(np.linalg.norm(Tn('embed.numeric.weight'), axis=0), 2))
print(CID)
```