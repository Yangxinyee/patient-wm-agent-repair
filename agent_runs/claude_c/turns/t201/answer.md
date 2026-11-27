The extra-insulin breakdown shows there is no hard ceiling. Head-0 ×4 is about 1.4× the original at every dose level, reaching −5.5 at large extra doses. Numeric ×5 helps at low doses but reverses at high doses (the encoder is pushed out of range), so it is a bad lever.

Head 0 behaves like a reader of the action-token slot. Its cost comes from a near-constant push at bundles without a drug, and `out` = LayerNorm(mean of h slots) can be identified exactly from states. So I can test whether that side effect is a near-constant shift in `out` that a bias correction on `core.out_norm` could cancel. Next I will set up head-0 ×4 and pick 100 training stays, so the states pass does not reuse the tuning stays.

```python
clear_ops()
for j in range(48):
    e=np.zeros(192); e[j]=1.0
    rank1('core.prelude.0.attn.out.weight', e, 3.0, side='in')
tr=pd.read_csv('/data/train.csv', usecols=['stay_id'])
rng=np.random.default_rng(7); tr_ids=sorted(rng.choice(tr.stay_id.unique(),100,replace=False).tolist())
json.dump([int(x) for x in tr_ids],open('/work/tr_ids.json','w'))
print(len(ops()), tr_ids[:3])
```