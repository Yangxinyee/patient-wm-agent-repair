The bundle structure is now fully pinned down. The stay ends with a bundle at 25.0833 holding the final insulin/NE END atoms, then one at 25.1667 holding 2 atoms, which can only be the END atoms of the hospital and ICU admissions (end_hour 25.1667). So END atoms apply to visits too.

The first bundle, a lone hospital_admission at hour −2, has the same content in every stay. If its state rows are identical across stays, that state depends only on the weights and the initial state, which gives a clean first unit test for the core.

Next I'll check that, and look at row norms of the embedding tables (kind, ekind, kq, text, categorical, bins). Rows that never get used tend to keep their initialisation scale, so the norms help identify which indices this data actually uses.

```python
X = S['x']
first = X[sx.groupby('stay_id').head(1).index]
print("first-bundle state spread across stays:", np.abs(first - first[0]).max())
second_idx = sx.groupby('stay_id').nth(1).index
for tn in ['embed.code.weight','embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.categorical.status.weight',
           'embed.categorical.unit.weight','embed.categorical.route.weight','embed.categorical.action.weight','embed.categorical.rate_unit.weight',
           'embed.categorical.discharged_to.weight','embed.bins.value.weight','embed.bins.dose.weight','embed.bins.rate.weight','type_emb.weight','last_status.weight']:
    t = load(tn); print(tn, np.round(np.linalg.norm(t, axis=1), 3))
print('numeric W col norms', np.round(np.linalg.norm(load('embed.numeric.weight'), axis=0),3), 'bias norm', np.linalg.norm(load('embed.numeric.bias')).round(3))
print('last_numeric col norms', np.round(np.linalg.norm(load('last_numeric.weight'), axis=0),3))
print('code_bias', load('code_bias'))
```