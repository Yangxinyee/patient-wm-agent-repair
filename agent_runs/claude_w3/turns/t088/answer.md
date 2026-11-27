The vocab fixes the scores, z = (x − median)/IQR clipped to ±10, along with the fine-bin edges.
- **Insulin:** the dose and the rate share the same scale (median 1.5, IQR 1), so in the embedding they always move together.
- **Norepinephrine:** the rate scale has median 0.46 and IQR 0.6, and the dose is the rate × 4.8 (a fixed 80 kg weight).
- **Doubled doses:** these mostly stay inside the z clip, though the top bins saturate.

I now think the most reliable route is a full local reimplementation of the model, checked against the factual tool's per-reading predicted means. Once it reproduces them, I can design the dose memory and test the factual NLL offline. Forward passes would then only be needed for confirmation.

The embedding uses categorical tables whose index mapping isn't documented. Next I will print row norms of the small embedding tables. Rows never used in training should sit at their initial scale (or zero), which tells me which indices the data uses. Clearly separated "used" and "unused" rows would make the mapping guessable. Uniform norms mean I will need the z-posterior check to decide.

```python
for n_ in ['embed.kind.weight','embed.ekind.weight','embed.kq.weight','embed.text.weight','embed.code.weight','embed.bins.dose.weight','embed.bins.rate.weight','embed.bins.value.weight','type_emb.weight','last_status.weight']+[n_ for n_ in NAMES if n_.startswith('embed.categorical')]:
    W_ = load_original(n_)
    print(n_.replace('embed.',''), tuple(W_.shape), np.round(W_.norm(dim=1).numpy(),2).tolist())
print('numeric col norms', np.round(load_original('embed.numeric.weight').norm(dim=0).numpy(),2).tolist(), 'bias', round(float(load_original('embed.numeric.bias').norm()),2))
```