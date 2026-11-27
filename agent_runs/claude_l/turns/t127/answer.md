The value head is a two-layer MLP: `value_bin.0` is 192→192, then the activation, then `value_bin.2` is 192→17. Its input has some unknowns:
- `last_numeric` takes 8 features, not the 4 listed in the spec;
- the `type_emb` index is unclear;
- 4 samples of z are drawn from the prior.

An exact head replica will take some matching against the factual means.

Before building it, I want to put a number on Plan B's main weakness. The original model's rollout response to insulin looked flat across lags, i.e. glucose deviations persist. A lag-1-only fix would then grow linearly over the window instead of levelling off like the true response.

From the forward-pass-1 dose-edit file I'll compare, hour by hour from 13 to 24:
- the model's mean shift under insulin ×2;
- the shift my estimated true response predicts from the recorded doses.

- **If the model's shift grows roughly linearly while the true one levels off:** Plan B will mis-scale the start-hour-6 and hour-16 interventions, and part of Plan A becomes worth its cost.

```python
de=pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(de.shape, list(de.columns)); print(de.head(3).to_string()[:500])
print(de.iloc[:,:].groupby(de.columns[2] if de.shape[1]>2 else de.columns[0]).size().head(10))
```