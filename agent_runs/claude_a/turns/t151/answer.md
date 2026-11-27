Run 146 printed the effect kernels I estimated from the records (glucose change per unit/h of insulin, MAP change per unit of norepinephrine, by lag in hours):

| Kernel | Lags 1–5 | Later lags | Sum |
|---|---|---|---|
| Insulin → finger glucose | −2.02, −2.33, −2.04, −1.63, −1.13 | taper over lags 6–10 | ≈ −11.2 |
| Insulin → lab glucose | similar | | |
| Norepinephrine → arterial MAP | 3.40, 2.60, 1.76, 0.91, 0.41 | 0.21 at lag 6, ~0 after | ≈ +9.2 |
| Norepinephrine → cuff MAP | similar | | |

It also listed the dims I had marked as cheap to repurpose: D = [155, 100, 128, 151, 144, 171, 52, 130, 99, 73, 132, 105].

Shapes I need for the edit:
- `core.delta_in` is a plain (192,) vector, so the recurrence has no data-dependent gate. Persistence of a memory dim must come from `a_in`/`flow.a`.
- The prelude is one pre-norm layer: attention with q/kv/out, then an FFN with 768 GELU units, with norm1 and norm3.
- The value head is a 192-unit GELU MLP feeding 17 bins.

The dimcost matrix (3×192) used up the output, so the remaining tensor shapes and the costs of the chosen dims were cut off. Next I'll print the cost of each dim in D and cands under the three cost rows, plus the leftover non-block tensor shapes, kept short. Small summed costs (<0.001 nats) would confirm that D is a safe pool for the memory, staging and reference dims. Large costs would mean choosing different dims or using fewer.

```python
import numpy as np
dc = np.load('/work/dimcost.npy')
print('rows sum', np.round(dc.sum(1),4), 'row means', np.round(dc.mean(1),5))
for nm_, L_ in [('D', D), ('cands', cands)]:
    print(nm_, ' '.join(f'{j}:{dc[0,j]*1e3:.1f}/{dc[1,j]*1e3:.1f}/{dc[2,j]*1e3:.1f}' for j in L_))
o_ = np.argsort(dc.max(0)); print('lowest max-cost dims', [(int(j), round(float(dc.max(0)[j])*1e3,2)) for j in o_[:24]])
i0_ = NAMES.index('embed.categorical.action.weight') if 'embed.categorical.action.weight' in NAMES else 0
rest = [n for n in NAMES[i0_:] if not n.startswith('core.blocks')]
print('; '.join(f'{n}:{tuple(SHAPES[n])}' for n in rest)[:3500])
```