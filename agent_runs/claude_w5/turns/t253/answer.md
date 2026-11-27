The comparison shows two problems.

1. My local original model now differs from the tool at drug bundles: relative error about 2e-3, against 1e-5 or better at reading bundles. On the train stays it matched to about 1e-7. On stay 2020000014 the first drug bundle (j=3) is already off.
2. Bundle by bundle, the tool's edit effect is on average twice as large as my local one and often points the opposite way (cosine near −1). This holds from the first drug bundle onward, even though the incoming states are identical.

The original-model error at drug bundles is about 8 in absolute norm, comparable to the edit's effect, so the drug-bundle input differs between my local pipeline and the tool. My edit's decoders read a small signal out of n3, so they could magnify such a difference into a sign flip.

The first suspect is how I built the tuning bundles. `make_bundles` may not be the builder that produced `bundles_B`, the one verified against the tool, and the drug atoms may differ in fields, NaNs or order. I'll compare `make_bundles` on a train stay with `bundles_B`, and recheck drug-bundle errors on a train stay against fwd_2.
- If the atoms differ, I fix the builder and redo the comparison.
- If they are identical and train drug bundles also show about 2e-3 error, the tool's float32 computation is ill-conditioned at drug bundles. My large-gain decoders would then be fragile by design.

```python
trn_sid = int(list(bundles_B)[0])
trn = pd.read_csv('/data/train.csv', nrows=400000)
dfs = trn[trn.stay_id==trn_sid]
Bm = make_bundles(dfs); Bb = bundles_B[trn_sid]
print(len(Bm), len(Bb))
diffs = [(k, a, b) for k,(a,b) in enumerate(zip(Bm, Bb)) if str(a)!=str(b)]
print('n differing bundles', len(diffs))
for k,a,b in diffs[:3]: print(k, '\nMB', a, '\nBB', b)
# drug bundle errors on train stay vs fwd_2 states
Z2 = np.load('/work/fwd_2/states_original.npz') if os.path.exists('/work/fwd_2/states_original.npz') else None
import os
print(os.listdir('/work/fwd_2'))
```