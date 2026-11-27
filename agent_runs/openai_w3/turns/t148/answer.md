The feed-forward fit has a better factual-safety tradeoff: it retains dose sensitivity while its factual score variation falls sharply. Its overall contrast slope is still attenuated, so I’ll check each dose factor separately before rescaling it. Consistent signs across stopping, halving, doubling, and tripling would support the edit.

```python
import numpy as np
import dis

print("=== FACTOR-SPECIFIC HELD-OUT CONTRASTS ===")
offset135 = 0
segments135 = []
for item135 in contrast_meta88:
    drug135, factor135, inds135 = item135[:3]
    n135 = len(inds135)*4
    segments135.append((drug135,float(factor135),slice(offset135,offset135+n135)))
    offset135 += n135
print("rows accounted for",offset135,"of",len(Y133))
for lam135 in (1.0,10.0):
    b135 = fits134[lam135][0]
    pred135 = XD134 @ b135
    for drug135, factor135, slc135 in segments135:
        j135 = 0 if drug135=="insulin" else 1
        keep135 = va133[slc135]
        yy135 = Y133[slc135][keep135,j135]
        pp135 = pred135[slc135][keep135,j135]
        den135 = np.sum(yy135**2)
        if den135:
            print("lambda/drug/factor",lam135,drug135,factor135,
                  "target/pred mean",round(float(yy135.mean()),3),round(float(pp135.mean()),3),
                  "slope",round(float(np.sum(yy135*pp135)/den135),3),
                  "RMSE",round(float(np.sqrt(np.mean((yy135-pp135)**2))),3))

print("\n=== READOUT SENSOR DERIVATIVES ===")
read_names135 = ["map_cuff","glucose_finger","map_arterial","glucose_lab"]
G135 = []
for nm135 in read_names135:
    cen135 = np.asarray(VOCAB["value_bin_centres"][nm135])
    keep135 = np.all(np.isclose(as_numpy(centres105),cen135[None]),axis=1)
    G135.append(grad134[keep135].mean(axis=0))
    print(nm135,"n",int(keep135.sum()),"gradient norm",np.linalg.norm(G135[-1]))
G135 = np.stack(G135)

print("\n=== PRELUDE RETURN DETAILS ===")
for nm135 in ("prelude_cache88","parts105"):
    fn135 = globals()[nm135]
    print(nm135,"constants",fn135.__code__.co_consts)
    print("return",[(i.opname,i.argrepr) for i in list(dis.get_instructions(fn135))[-18:]])

cache135 = prelude_cache88(hp105,e81[pick105],u81[pick105],z88[pick105])
parts135 = parts105(hp105,e81[pick105],u81[pick105],z88[pick105])
for a135 in (0,2):
    for b135 in (0,1):
        print("cache/parts",a135,b135,
              "RMSE",float(torch.sqrt(torch.mean((cache135[a135]-parts135[b135])**2))))
print("next dt range",as_numpy(next_dt105).min(),as_numpy(next_dt105).max())
```
