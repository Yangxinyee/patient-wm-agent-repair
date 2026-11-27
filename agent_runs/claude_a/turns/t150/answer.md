Run 145 found the variables from the write-design runs: head-3 prelude features `A3tr`/`A3va`, rate targets `rtr`/`rva`, drug masks, gate weights `wc_` and gate scores `sc_tr`/`sc_va`. The replica functions (`tf_eval`, `tf_loop`, `build_ds`, `prelude_parts`) are still defined. So I can continue the memory-channel plan without rebuilding.

What remains to settle before writing weights:
- which state dims are cheapest to repurpose (dimcost, D, cands);
- the exact tensor shapes of the prelude, recurrence, value-head and embedding tables, so the edit stays inside them;
- the effect kernels I estimated from the records (psi_*), which set the targets.

This run is read-only. It will confirm that the low-cost dims exist and whether `core.delta_in` is a fixed vector or depends on the input. A fixed vector means persistence has to come from `a_in`/`flow.a`, as planned. If it depends on the input, it could provide a gate inside the recurrence.

```python
import numpy as np, json
for nm_ in ['psi_ins','psi_ne','psi','psi_f','psi_l','psi_n','psi_c']:
    print(nm_, np.round(np.asarray(g[nm_]),3) if nm_ in g else None)
print('D', D); print('cands', cands)
dc = np.load('/work/dimcost.npy'); print('dimcost', dc.shape, dc.dtype)
if dc.ndim == 1:
    o_ = np.argsort(dc); print('lowest-cost dims', [(int(i), round(float(dc[i]),5)) for i in o_[:20]])
else:
    print(np.round(dc[:5],5))
keys_ = [n for n in NAMES if not n.startswith('core.blocks') and not n.startswith('encoder') and 'deepsets' not in n]
print('; '.join(f'{n}:{tuple(SHAPES[n])}' for n in keys_)[:6000])
print('n blocks tensors', sum(n.startswith('core.blocks') for n in NAMES))
print('wc_', wc_.shape, 'gate sep tr', float(sc_tr[drug_tr].min()), float(sc_tr[~drug_tr].max()) if drug_tr.dtype==bool else '')
```