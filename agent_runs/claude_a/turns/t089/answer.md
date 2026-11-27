Run 84 settled the readout conventions:
- A readout row exists at each recorded glucose_finger reading (map_arterial for norepinephrine) in hours 13–24, not at every hour.
- An hour is "reached" if it comes after the first infusion that starts at hour 12 or later. Stay …000 first gets insulin at 15.08, so hour 15 is not reached and 16+ are. Stay …001 has insulin at 12.08, so 13+ are reached.

That gives 50 of 98 stays for insulin and 27 of 47 for norepinephrine.

For the rollout I must re-embed readings with model-drawn values. Next I'll read the vocabulary's notes on value bins and numerics, to see how bin centres are defined. I'll also confirm that a measurement atom's vector changes with its value only through `bins.value[bin]` and `numeric[:,0]·z`. If the decomposition is exact, I can embed drawn values cheaply. If not, there is another value-dependent term to include.

```python
print("value_bin_note:", VOCAB['value_bin_note']); print("numeric_note:", VOCAB['numeric_note'])
print("numeric keys sample:", list(VOCAB['numeric'].items())[:1] if isinstance(VOCAB['numeric'], dict) else str(VOCAB['numeric'])[:300])
a_t = AT[(AT.event=='glucose_finger')].head(3).copy()
av1_, _ = atom_vectors(a_t, {})
a_t2 = a_t.copy(); a_t2['value'] = [95.0, 260.0, 141.0]
av2_, _ = atom_vectors(a_t2, {})
numW = Po['embed.numeric.weight']; Wb = Po['embed.bins.value.weight']
pred_diff = torch.stack([Wb[fbin('glucose_finger','value',v2)] - Wb[fbin('glucose_finger','value',v1)] + numW[:,0]*(z32('glucose_finger','value',v2)-z32('glucose_finger','value',v1))
                         for v1, v2 in zip(a_t['value'].values, a_t2['value'].values)])
print("decomposition err", float(((av2_-av1_) - pred_diff).abs().max()))
print("centres gf", np.round(centM[cid['glucose_finger']].numpy(),1)); print("centres map_art", np.round(centM[cid['map_arterial']].numpy(),1))
```