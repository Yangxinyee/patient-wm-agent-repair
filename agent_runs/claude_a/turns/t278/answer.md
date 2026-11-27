Two findings explain the poor decoder.
- Most insulin bundles hold two insulin atoms, one of kind 0 and one of kind 2: a new rate starting and the previous infusion ending.
- Across 2,496 insulin-only bundles the encoder produces only about 15 distinct vectors.

With so few distinct inputs, a linear readout can fail only if my target varies within a group of identical encodings. So `bundle_rates` must be returning a rate other than the one the encoder sees, for example the rate of the infusion that is ending. I'll group insulin-only bundles by their encoding and print, for the largest groups, the count, the kind-0 rate, the kind-2 rate and my target. If the kind-0 rate is constant within each group, it is the right target, and I'll refit the decoder on it.

```python
sig=np.round(u[:,keepN[:6]],3); key_=[tuple(r_) for r_ in sig]
rows_b=bd['row'].values
r0=ai[ai['kind']==0].groupby('row')['rate'].first(); r2=ai[ai['kind']==2].groupby('row')['rate'].first()
dfc=pd.DataFrame({'key':[key_[i] for i in np.where(solo)[0]], 'y':y[solo], 'k0':r0.reindex(rows_b[solo]).values, 'k2':r2.reindex(rows_b[solo]).values})
gq=dfc.groupby('key').agg(n=('y','size'), y_m=('y','mean'), y_sd=('y','std'), k0_m=('k0','mean'), k0_sd=('k0','std'), k2_m=('k2','mean'), k2_sd=('k2','std')).sort_values('n',ascending=False)
print(gq.head(8).round(3).reset_index(drop=True).to_string())
```