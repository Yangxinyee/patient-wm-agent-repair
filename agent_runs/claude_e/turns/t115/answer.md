Run 103 shows two problems. The same output was printed four times.

**1. Too costly on the train stays.** The local factual NLL rises by 0.0087 per reading, above the 0.005 budget:

| event | ΔNLL per reading |
|---|---|
| glucose finger | +0.0095 |
| glucose lab | +0.0015 |
| arterial MAP | +0.0053 |
| cuff MAP | +0.019 |

Roughly 0.0034 of this is the insulin correction and 0.0053 the NE correction. That is far above my Gaussian back-of-envelope estimate. The gate itself works: each pair is exactly zero on the other code group.

**2. Something is broken on sel1.** On these held-out tuning stays the edit shifts the mean predictions systematically, by −2.1 mg/dL on finger glucose and +4.3 mmHg on cuff MAP. NLL rises by 0.14.

On train stays the average shifts are near zero. My first suspicion is a mismatch between `RR` and the state array `X` I passed in. `RR` has a `src` column, so it may mix rows from different state sources. That would make its `prev` indices point at the wrong states. The alternative is that the read-out simply does not transfer to unseen stays, which would be much worse.

To tell these apart, I'll check:
- the sources in `RR`;
- whether my base head on `X` reproduces the stored `RR.pm`;
- whether the read-out's mean on sel1 differs from train once restricted to rows whose states are certainly correct.

```python
print('RR src counts:', RR.src.value_counts().to_dict())
pm0_s1 = base_out[5][0]
print('MAE base pm on X vs RR.pm by src:', {s: round(float(np.mean(np.abs(pm0_s1[(RR.src==s).values] - RR.pm.values[(RR.src==s).values]))),3) for s in RR.src.unique()})
for nm in ['X', 'X1', 'Xs', 'XA']:
    v = globals()[nm]; v = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
    print(nm, tuple(v.shape), 'first row sum', float(v[0].sum()), 'mean abs', float(v.abs().mean()))
zz2 = np.load('/work/fwd_2/states_original.npz', allow_pickle=True); X2raw = np.array(zz2['x'])
print('fwd_2 raw states shape', X2raw.shape, ' equal to X?', np.allclose(X2raw, (X.numpy() if torch.is_tensor(X) else np.asarray(X))[:len(X2raw)]) if X2raw.shape == tuple(X.shape) else 'shape differs')
```