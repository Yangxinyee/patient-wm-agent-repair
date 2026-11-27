The state does not remember dose size. In cross-validated ridge fits on the tool's states for 100 stays:
- **Insulin rate given that insulin was given** is predictable with R² ≈ 0.4–0.6 at every lag. That comes from the glucose level, since the policy sets the rate from glucose.
- **The rate's deviation from the policy mean**, the part that is free and is what a dose change alters, is barely encoded at lag 0 (R² = 0.16) and not at all from lag 1 on (R² ≈ 0).

So a linear readout of the existing state cannot supply the missing dose effect; the model forgets dose size almost at once. The memory M will not help, since END records carry no rate.

Any dose memory has to be built. Its time profile matters: the model's own sampled glucose carries part of the effect forward, so the "direct" kernel to inject may be concentrated in the first 1–3 hours. I will measure that later with the factual residual regression over more lags.

First I need my offline head to be faithful, because the edit will probably live partly in the head. It is biased +2.4 mg/dL against the tool. A likely cause is indexing: the value-bin embedding table may be per code (code×17 + bin) rather than shared. Next I will check the shapes of the bin and status tables and test that indexing.
- **If the error drops to about 2:** the head is fixed.

```python
for n_ in ['embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight','last_status.weight','embed.code.weight','embed_transition.rate_bin.weight']:
    print(n_, SHAPES[n_])
print('codes', [(c['name'], c['id']) for c in VOCAB['codes']])
```