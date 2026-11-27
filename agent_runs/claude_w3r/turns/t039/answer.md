The atom construction is now clear:
- Every infusion record produces a WORLD atom at its start and an END atom at its end (start + 1 h).
- Admissions produce a WORLD atom at the start and END atoms at the stay end (25.17).
- has_insulin marks bundles containing an insulin start atom.

The counts match the record exactly, e.g. 4 atoms at 5.08 = insulin start + insulin END + norepinephrine start + norepinephrine END.

Next I will inspect the embedding tables (row norms, zero rows) to infer conventions such as padding rows for "absent". I will also run a simple consistency check of the states: whether out = out_norm(mean over slots of h) holds, which pins down the LayerNorm convention.

```python
import torch
Wt = {nm: load_original(nm) for nm in NAMES}
for nm in ['embed.categorical.action.weight','embed.categorical.discharged_to.weight','embed.categorical.rate_unit.weight',
           'embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.code.weight',
           'embed.ekind.weight','embed.kind.weight','embed.kq.weight','embed.text.weight','embed.bins.value.weight',
           'embed.bins.rate.weight','embed.bins.dose.weight','last_status.weight','type_emb.weight']:
    w=np.asarray(Wt[nm]); print(nm, np.round(np.linalg.norm(w,axis=1),3))
def ln(x, w, b, eps=1e-5):
    mu=x.mean(-1,keepdims=True); var=x.var(-1,keepdims=True); return (x-mu)/np.sqrt(var+eps)*w+b
hS = Xs[:,192:960].reshape(-1,4,192); outS = Xs[:,:192]
pred_out = ln(hS.mean(1), np.asarray(Wt['core.out_norm.weight']), np.asarray(Wt['core.out_norm.bias']))
print('out vs out_norm(mean slots h): max abs diff', np.abs(pred_out-outS).max(), 'rel', np.abs(pred_out-outS).mean()/np.abs(outS).mean())
```