The fresh attention fits respond mainly through head 2, but they recover only part of the dose change and introduce appreciable factual noise. I’ll also test the existing feed-forward features and recalibrate the readout sensitivity. A better dose-to-noise tradeoff would favor a small, centered feed-forward edit.

```python
import numpy as np
import scipy.linalg as sla
import torch

XF134 = np.asarray(fact88, dtype=np.float64)
XD134 = np.asarray(DX88, dtype=np.float64)
muF134 = XF134.mean(axis=0)
FC134 = XF134-muF134
CF134 = FC134.T @ FC134 / len(FC134)
CD134 = XD134[tr133].T @ XD134[tr133] / tr133.sum()
R134 = XD134[tr133].T @ Y133[tr133] / tr133.sum()
fits134 = {}
print("=== FRESH FFN CONTRAST FITS ===")
for lam134 in (0.001, 0.01, 0.1, 1.0, 10.0):
    b134 = sla.solve(CD134+lam134*CF134+1e-5*np.eye(768),
                    R134, assume_a="pos")
    pr134 = XD134[va133] @ b134
    er134 = np.sqrt(np.mean((pr134-Y133[va133])**2,axis=0))
    sl134 = []
    for j134 in range(2):
        nz134 = np.abs(Y133[va133,j134])>1e-8
        sl134.append(np.sum(pr134[nz134,j134]*Y133[va133,j134][nz134]) /
                     np.sum(Y133[va133,j134][nz134]**2))
    ns134 = np.sqrt(np.mean((FC134@b134)**2,axis=0))
    fits134[lam134] = (b134,muF134,er134,ns134,sl134)
    print(lam134,"held RMSE",np.round(er134,3),
          "slopes",np.round(sl134,3),"factual SD",np.round(ns134,3))

print("\n=== RECALIBRATING READOUT DERIVATIVES ===")
def readout134(view):
    q = layer_norm(view+W["type_emb.weight"][1],"query_norm")
    logits = mlp(q+attr_static105,"attr_heads.value_bin") + torch.as_tensor(transition97)
    return (torch.softmax(logits,dim=-1)*centres105).sum(dim=-1)

eps134 = 0.01
grad134 = np.empty((len(base_view105),192),dtype=np.float64)
for start134 in range(0,192,16):
    cols134 = np.arange(start134,min(start134+16,192))
    perturb134 = torch.zeros((len(cols134),1,192),dtype=base_view105.dtype)
    perturb134[torch.arange(len(cols134)),0,torch.as_tensor(cols134)] = eps134
    plus134 = readout134(base_view105[None]+perturb134)
    minus134 = readout134(base_view105[None]-perturb134)
    grad134[:,cols134] = as_numpy((plus134-minus134)/(2*eps134)).T

vc134 = VOCAB["value_bin_centres"]
print("centre table",type(vc134).__name__,
      list(vc134)[:12] if isinstance(vc134,dict) else np.shape(vc134))
print("unique centre rows",np.unique(as_numpy(centres105),axis=0)[:,[1,8,16]])
print("old G row norms",np.linalg.norm(G105,axis=1))
print("new per-reading derivative norm median",np.median(np.linalg.norm(grad134,axis=1)))
print("parts105 locals",parts105.__code__.co_varnames)
print("state_update82 return ops",
      [(i.opname,i.argrepr) for i in list(__import__("dis").get_instructions(state_update82))[-12:]])
```
