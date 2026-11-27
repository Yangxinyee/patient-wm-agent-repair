The local attention calculation matches the cached model output closely, and the reserved head was already replaceable with little factual error. I’ll test whether the other features can predict the marker’s factual component while remaining insensitive to dose changes. Small held-out residuals with preserved dose contrasts would support amplification.

```python
import numpy as np
from scipy.linalg import solve
from pathlib import Path
import inspect

def r245_features(nt):
    feat, _ = r244_attention_features(nt, safe236)
    q = nt[:,:4] @ qvec240
    k = nt @ kvec240
    logits = 16.0 * q[:,:,None] * k[:,None,:]
    logits -= logits.max(axis=-1,keepdims=True)
    a = np.exp(logits)
    a /= a.sum(axis=-1,keepdims=True)
    pool = np.einsum("bqk,bkd->bqd",a,nt).reshape(-1,192)
    return np.column_stack((feat[:,r244_other],pool))

r245_F = r245_features(nt240)
r245_train = np.repeat(tr239,4)
r245_hold = np.repeat(te239,4)
r245_M = r245_F[:,144:] @ V240
r245_deltas = {}
r245_Gd = np.zeros((336,336))
for key, item in cf239.items():
    ix = np.flatnonzero(item["mask"])
    nt = nt240[ix].copy()
    tok = np.asarray(item["token"][ix],dtype=np.float64)
    tok = (tok-tok.mean(axis=1,keepdims=True))/np.sqrt(tok.var(axis=1,keepdims=True)+1e-5)
    nt[:,6] = tok*gamma240+beta240
    fc = r245_features(nt)
    delta = fc-r245_F.reshape(-1,4,336)[ix].reshape(-1,336)
    dm = delta[:,144:] @ V240
    tm = np.repeat(tr239[ix],4)
    hm = np.repeat(te239[ix],4)
    r245_deltas[key] = {"delta":delta.astype(np.float32),"dm":dm,"train":tm,"hold":hm}
    if key[1] in (0.5,2.0):
        r245_Gd += delta[tm].T @ delta[tm]

r245_X = r245_F[r245_train]
r245_G = r245_X.T @ r245_X
r245_ridge = 1e-6*np.trace(r245_G)/336
r245_Cbase = solve(r245_G+r245_ridge*np.eye(336),
                  r245_X.T@r244_target[r245_train],assume_a="pos")
print("selective-head factual compensation RMSE",
      np.sqrt(np.mean((r245_F[r245_hold]@r245_Cbase-r244_target[r245_hold])**2)))
r245_E = np.zeros((336,2))
r245_E[144:] = V240
r245_fits = {}
for weight in (0.0,1.0,10.0,100.0):
    K = solve(r245_G+weight*r245_Gd+r245_ridge*np.eye(336),
              r245_X.T@r245_M[r245_train],assume_a="pos")
    R = r245_E-K
    residual = r245_F @ R
    print("\nINVARIANCE WEIGHT",weight,
          "held residual RMS",np.sqrt(np.mean(residual[r245_hold]**2,axis=0)).round(5))
    contrasts = {}
    for key, item in r245_deltas.items():
        d = item["delta"][item["hold"]].astype(np.float64) @ R
        contrasts[key] = d.mean(axis=0)
    print("held dose contrasts", {str(k):np.round(v,3).tolist() for k,v in contrasts.items()})
    r245_fits[weight]={"K":K,"R":R,"residual":residual,"contrasts":contrasts}

print("\nOUTPUT-DIRECTION OBJECTS")
for name,obj in list(globals().items()):
    if name.endswith(("221","226","227")) and not name.startswith("r245"):
        if isinstance(obj,np.ndarray) and obj.ndim<=2:
            msg = f"{name} {obj.shape}"
            if obj.ndim==2 and obj.shape[0]==192 and obj.shape[1]<=6:
                msg += " column_norms="+str(np.linalg.norm(obj,axis=0).round(3))
            print(msg)
        elif callable(obj) and hasattr(obj,"__code__"):
            print(name,inspect.signature(obj),"globals",obj.__code__.co_names)
        elif isinstance(obj,dict):
            print(name,"dict keys",list(obj)[:12])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 245\nAttention reconstruction verified to <4e-6 RMSE. Fitted selective-head factual compensation and dose-invariant marker predictors with stay-held-out leakage/contrast checks. No weights changed. Counts: 245 Python runs, 23 forwards.\n")
```