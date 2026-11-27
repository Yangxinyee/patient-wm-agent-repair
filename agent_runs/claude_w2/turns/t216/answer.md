Run 210 recovered the helper code. `dec.py` builds features for the value head: `wc_hidden` returns the head's input vector with_code and the first hidden layer, and `lag_feats` returns absolute and deviation dose lags. `evalh.py` scores rollouts against the truth kernels Kins/Knor, and `policy.py` builds the sliding-scale policy function pf.

The variable listing was cut before the most recent variables, which are the ones from runs 149–207: the trained channel vectors, decoders and gains. I will print the rest of the listing, the function names, and the truth kernels Kins and Knor. That lets me rebuild the oracle/channel experiment without recomputing anything expensive. I expect Kins to be about −2.0, −2.5, −2.2, … and Knor about +3.6, +2.7, +1.7, …, matching my notes.

```python
i0=[i for i,s in enumerate(vs) if s.startswith('k_=')]
print('VARS tail:',' '.join(vs[i0[0]:])[:4000] if i0 else '')
print('FUNCS:',' '.join(fs)[:3500])
print('Kins',np.round(Kins,3)); print('Knor',np.round(Knor,3))
```