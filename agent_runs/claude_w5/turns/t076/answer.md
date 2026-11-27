Run 71 gave me the layouts I need:
- **States file:** one row per bundle, 992 numbers each (out, then the 4 h slots, then the z posterior mean), plus stay, hour and which drugs the bundle held.
- **Parsed bundles:** a list of (hour, atoms) per stay.
- **Normalisation:** z-scores are (x − median)/IQR clipped to ±10; fine bins come from edges, some of them duplicated, such as the insulin rate edges at 0.5.

My plan for the edit is a dedicated slow "dose memory" in the state. Each new infusion record writes the part of the dose not explained by the recent reading into it. The glucose (or MAP) value head reads the memory and shifts its predicted distribution. Building that edit, and checking it on many stays without spending forward passes, needs a rebuild of the forward pass that matches the real model.

I'll rebuild in small checked steps. The first check is the simplest identity: out should equal core.out_norm applied to the mean over the 4 h slots. If it matches to about 1e-5, that confirms the slot layout and the LayerNorm epsilon. I'll also re-read the data README so the categorical embedding fields can be matched to columns.

```python
import numpy as np, torch
print(open('/data/README.md').read()[:3000])
print(tr.columns.tolist()); print(tr.head(3).to_string()[:1500])
X_ = Z['x']; out_s = X_[:, :192]; h_s = X_[:, 192:960].reshape(-1, 4, 192); zm_s = X_[:, 960:]
def LN(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = ((x-mu)**2).mean(-1, keepdims=True)
    return (x-mu)/np.sqrt(var+eps)*w + b
W = {n_: load_original(n_).numpy() if hasattr(load_original(n_), 'numpy') else np.asarray(load_original(n_)) for n_ in NAMES}
o_hat = LN(h_s.mean(1), W['core.out_norm.weight'], W['core.out_norm.bias'])
print('out vs LN(mean slots h) max abs err', np.abs(o_hat-out_s).max(), 'rel', np.linalg.norm(o_hat-out_s)/np.linalg.norm(out_s))
h_alt = X_[:, 192:960].reshape(-1, 192, 4).transpose(0,2,1)
o_alt = LN(h_alt.mean(1), W['core.out_norm.weight'], W['core.out_norm.bias'])
print('alt layout err', np.abs(o_alt-out_s).max())
```