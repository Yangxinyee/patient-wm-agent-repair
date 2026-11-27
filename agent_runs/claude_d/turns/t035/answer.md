The bundle structure is now clear:
- **Visits:** a hospital-admission bundle at −2.
- **Readings:** one bundle per hour at hh:00 holding all that hour's readings.
- **Drugs:** one bundle at hh:05 holding new insulin and norepinephrine records plus END atoms for the records ending then. For example, 5.0833 has 4 atoms: insulin, norepinephrine and both ENDs. `has_insulin` flags only new insulin records, not ENDs.
- **End of stay:** END atoms at 25.0833 for the last drug records and a final bundle at 25.1667 holding the two visit ENDs.

So each insulin record appears twice to the model, as a start atom and as an END atom an hour later. The dose edit doubles "every insulin atom", possibly including ENDs.

Before writing a local copy I want to cut the guesswork. Unused rows in the embedding tables should be near their initial values. So next I will look at the row norms of every small embedding table (kind, ekind, kq, text, categorical tables, bins) and at `last_numeric`. That should show which categories and bins the data actually uses and help me infer the featurisation.

```python
from safetensors.numpy import load_file
Wt_ = {n: load_original(n) for n in NAMES}
for n in ['embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.categorical.status.weight','embed.categorical.unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.rate_unit.weight','embed.categorical.discharged_to.weight','embed.code.weight','embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight','type_emb.weight','last_status.weight']:
    w = Wt_[n]; print(n, np.round(np.linalg.norm(w,axis=1),2))
print('numeric W col norms', np.round(np.linalg.norm(Wt_['embed.numeric.weight'],axis=0),3), 'bias norm', np.linalg.norm(Wt_['embed.numeric.bias']).round(3))
print('last_numeric col norms', np.round(np.linalg.norm(Wt_['last_numeric.weight'],axis=0),3))
print('code_bias', np.round(Wt_['code_bias'],2))
print('flow.a range', Wt_['core.flow.a'].min(), Wt_['core.flow.a'].max(), 'exp(-flow) taus (s)', np.exp(-Wt_['core.flow.a']).min(), np.exp(-Wt_['core.flow.a']).max())
print('delta_in', np.round(np.quantile(Wt_['core.delta_in'],[0,.5,1]),3), 'a_in', np.round(np.quantile(Wt_['core.a_in'],[0,.5,1]),3))
```