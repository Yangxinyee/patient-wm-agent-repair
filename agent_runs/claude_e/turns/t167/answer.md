This changes the NE picture. Among MAP-arterial rows with NE running, the original model's residual has essentially no slope on the rate-level readout (0.06). The overall slope of 1.7 came from rate-0 rows, whose mean residual is -3.0. Those rows are probably biased by how these batches were chosen (stays that received both drugs, so rate-0 hours include the hours before NE starts).

So a level correction would not improve the factual fit. It would be pure cost, roughly proportional to the coefficient squared. That makes the factual cost the binding constraint for both drugs, and I need its true size on random stays like the evaluation's.

My offline estimate of 0.0052 comes from both-drug training stays, which have more corrected rows than random stays. A factual "both" pass on 100 random tuning stays would give the exact cost of the current edit there.

First I'll confirm the saved edit is gated3(-1.9, 3) and recover the tuning stay list used in fwd_3.

```python
sav = {k: load(k).numpy() if hasattr(load(k),'numpy') else np.asarray(load(k)) for k in edited()}
Ecur = gated3(-1.9, 3.0)
mp = {'attr_heads.value_bin.0.weight':'W0','attr_heads.value_bin.0.bias':'b0','attr_heads.value_bin.2.weight':'W2','embed_transition.value_bin.weight':'TR'}
for k, v in sav.items():
    e = Ecur[mp[k]]; e = e.numpy() if hasattr(e,'numpy') else np.asarray(e)
    print(k.split('.')[-2:], v.shape, 'max|saved - gated3(-1.9,3)| = %.2e' % np.abs(v - e.reshape(v.shape)).max())
s3 = json.load(open('/work/fwd_3/summary.json'))
print({k: (v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in s3.items()})
f3 = pd.read_csv('/work/fwd_3/factual_original.csv')
print(f3.columns.tolist()[:12], 'stays', f3.stay_id.nunique() if 'stay_id' in f3 else None)
```