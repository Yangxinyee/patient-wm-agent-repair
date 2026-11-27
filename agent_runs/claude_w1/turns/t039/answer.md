The bundle structure is now clear:
- **Before ICU:** hospital_admission alone at −2. The −1 readings form one bundle (4 atoms), and hour 0 holds the readings plus icu_admission.
- **Drugs:** each drug minute (h+0.083) is its own bundle. It contains the new infusion record plus the END atom of the record that just finished (2 atoms after the first hour).
- **End of stay:** a lone END at 25.083 (the last record ending), then a final bundle at 25.167 holding the two admissions' ENDs.

So a dose change affects the action bundle through two atoms: the new record, and the next hour's END atom if END carries the rate.

To reimplement the atom embedding I need to know which rows of the small categorical tables are used: unit, rate_unit, action, route, status, discharged_to, text, kq. Trained rows and never-used rows may differ in norm. Next I'll print row norms of all embedding tables and the scale of `embed.numeric`.

```python
import torch
W = {n: torch.tensor(load(n)) for n in NAMES}
for n in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.rate_unit.weight','embed.categorical.discharged_to.weight','embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight','type_emb.weight','last_status.weight','core.slots']:
    print(n, np.round(W[n].norm(dim=1).numpy(),3))
print('numeric cols norm', np.round(W['embed.numeric.weight'].norm(dim=0).numpy(),3), 'bias norm', W['embed.numeric.bias'].norm().item())
print('last_numeric cols norm', np.round(W['last_numeric.weight'].norm(dim=0).numpy(),3))
print('flow.a range (tau hours):', (1/np.exp(W['core.flow.a'].numpy())/3600).min(), (1/np.exp(W['core.flow.a'].numpy())/3600).max())
print('code_bias', np.round(W['code_bias'].numpy(),2))
```