The factor check is encouraging for insulin, but not for norepinephrine: the fit misses stopping and much of the halving response. It is therefore only a pilot, not a complete repair. I’ll compute a low-rank state correction by finite differences and check its size before applying it; the centre-table error is just a 16-versus-17-bin mismatch.

```python
import numpy as np
import torch
import scipy.linalg as sla
from pathlib import Path

for nm136 in list(W):
    W[nm136] = load_original(nm136).float()

read_names136 = ["map_cuff","glucose_finger","map_arterial","glucose_lab"]
G136 = []
for nm136 in read_names136:
    cen136 = np.asarray(VOCAB["value_bin_centres"][nm136])
    keep136 = np.all(np.isclose(as_numpy(centres105)[:,1:],cen136[None]),axis=1)
    assert keep136.any(), nm136
    G136.append(grad134[keep136].mean(axis=0))
G136 = np.stack(G136)
print("sensor derivative norms",np.linalg.norm(G136,axis=1))

cache136 = prelude_cache88(hp105,e81[pick105],u81[pick105],z88[pick105])
parts136 = parts105(hp105,e81[pick105],u81[pick105],z88[pick105])
raw_candidates136 = [
    parts136[0], parts136[1], cache136[2],
    parts136[0]+linear(cache136[1],"core.prelude.0.ffn.2")]
raw_errors136 = [
    float(torch.sqrt(torch.mean((layer_norm(v,"core.prelude_norm")-cache136[0])**2)))
    for v in raw_candidates136]
print("raw prelude reconstruction errors",raw_errors136)
raw136 = raw_candidates136[int(np.argmin(raw_errors136))]
assert min(raw_errors136)<1e-4

hbase136, obase136 = state_update82(
    hp105,cache136[0],memory105,mask105,age105)
vbase136 = next_view105(hbase136,obase136)
J136 = np.empty((192,192),dtype=np.float64)
eps136 = 0.01
dt_keep136 = next_dt105.clone()
try:
    for start136 in range(0,192,8):
        cols136 = np.arange(start136,min(start136+8,192))
        pert136 = torch.zeros((2*len(cols136),1,1,192))
        for k136,c136 in enumerate(cols136):
            pert136[k136,0,0,c136] = eps136
            pert136[k136+len(cols136),0,0,c136] = -eps136
        cases136 = len(pert136)
        E136 = layer_norm(raw136[None]+pert136,"core.prelude_norm").reshape(-1,4,192)
        hh136,oo136 = state_update82(
            hp105.repeat(cases136,1,1),E136,
            memory105.repeat(cases136,1,1),
            mask105.repeat(cases136,1),
            age105.repeat(cases136,1,1))
        next_dt105 = dt_keep136.repeat(cases136)
        vv136 = next_view105(hh136,oo136).reshape(cases136,len(hp105),192)
        dd136 = (vv136[:len(cols136)]-vv136[len(cols136):]).mean(dim=1)/(2*eps136)
        J136[:,cols136] = as_numpy(dd136).T
finally:
    next_dt105 = dt_keep136

K136 = G136 @ J136
T136 = np.array([[0.,3.2],[-1.94,0.],[0.,3.2],[-1.94,0.]])
ridge136 = 1e-4*np.trace(K136@K136.T)/4
directions136 = K136.T @ sla.solve(
    K136@K136.T+ridge136*np.eye(4),T136,assume_a="pos")
print("response-map singular values",np.linalg.svd(K136,compute_uv=False))
print("injection direction norms",np.linalg.norm(directions136,axis=0))
print("linear predicted sensor effects\n",K136@directions136)

B136 = fits134[10.0][0].copy()
mu136 = fits134[10.0][1].copy()
pred136 = XD134 @ B136
scales136 = np.ones(2)
for drug136,factor136,slc136 in segments135:
    if factor136==2.0:
        j136 = 0 if drug136=="insulin" else 1
        keep136 = va133[slc136]
        scales136[j136] = (
            pred136[slc136][keep136,j136].mean() /
            Y133[slc136][keep136,j136].mean())
B136 /= scales136[None]
print("x2 contrast scales",scales136)
print("rescaled factual score SD",
      np.sqrt(np.mean((FC134@B136)**2,axis=0)))

with Path("/work/notes.md").open("a") as f136:
    f136.write("\nPython run 136; forwards 7. Fresh FFN contrast fit has low factual variance and consistent insulin signs but misses NE stopping/halving; unsuitable as final full-dose solution. Constructed finite-difference low-rank injection directions and x2 contrast normalization for a reversible pathway pilot. No weight edits yet.\n")
```
