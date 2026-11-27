Run 273 shows that the factual damage from v4 is concentrated in one place:

- **After the first drug in a stay:** the edited model matches the original almost exactly. NLL change is about 0.000 and the mean shifts by only about 0.2 mg/dL.
- **Before the first drug:** predictions are badly off. Glucose is about +16 to +20 mg/dL too high and MAP about −8 mmHg too low, with NLL worse by 0.5 to 2.2 nats per reading.
- **Teacher-forced dose effect:** the insulin channel barely acts (−0.05 mg/dL per unit of centred dose). The design intended roughly −0.5.

So v4 has two separate faults: a strong dose-independent bias in drug-free stretches, and a dose channel that is nearly inert under teacher forcing.

The bias pattern suggests a mechanism. I chose the channel directions to be quiet at drug bundles and at reading bundles of stays already on drugs. In drug-free states the model's input sits elsewhere, and there the channel is probably far from quiet. The same mechanism would explain the rollout: when drugs stop after hour 12, the state drifts into drug-free territory and glucose creeps upward, which is the +25 drift.

To test this, I will tabulate the edited-minus-original bias by hour for readings before the first drug, and by hours since the last drug bundle for the rest.
- Bias that is zero at hour 0, appears once reading bundles are processed, and grows with time since the last drug confirms the drug-free-state leak.
- Bias already present at hour 0 would instead point to the start-of-stay processing.

```python
import numpy as np, pandas as pd
dh = drugs.copy(); dh['h'] = np.floor(dh.hour+1e-6).astype(int)
dset = dh.groupby('stay_id').h.apply(lambda x: np.array(sorted(set(x))))
def gap_since(s, h):
    a = dset.get(s, np.array([])); a = a[a < h]
    return int(h - a.max()) if len(a) else -1
f9['h'] = np.floor(f9.hour+1e-6).astype(int)
f9['gap'] = [gap_since(s, h) for s, h in zip(f9.stay_id, f9.h)]
for ev in ['glucose_finger','map_arterial']:
    a = f9[(f9.event==ev)&(f9.grp=='before_drug')]
    print(ev, 'BEFORE first drug, by hour: dmean', a.groupby('h').dm.mean().round(1).to_dict(), ' n', a.groupby('h').size().to_dict())
    b = f9[(f9.event==ev)&(f9.grp=='after_drug')].copy(); b['g'] = b.gap.clip(upper=6)
    print(ev, 'AFTER drug, by hours since last drug bundle: dmean', b.groupby('g').dm.mean().round(2).to_dict(), ' dNLL', b.groupby('g').dn.mean().round(3).to_dict(), ' n', b.groupby('g').size().to_dict())
print('stays with first drug at hour 0:', (first_drug.loc[f9.stay_id.unique()] < 1).mean().round(2), ' first-drug hour quantiles', np.percentile(first_drug.loc[f9.stay_id.unique()], [25,50,75,90]).round(2))
```