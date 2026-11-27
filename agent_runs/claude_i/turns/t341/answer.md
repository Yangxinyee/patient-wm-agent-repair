The architecture explains the failed check. The prelude is a pre-norm transformer layer: each token is layer-normed before attention, so a z-token addition is bounded and its effect depends on attention to that token. At insulin bundles, attention evidently gives the z token essentially no influence on glucose. The value head's last_numeric input is [has_last, age, last value, last rate] of the same code, so there is no direct drug input to the glucose head either. The only working channel from dose to readings is the drug token u itself, which already yields −2 for insulin ×2.

So I'll change strategy:
1. Undo the broken z-path edit, so the model is back to the original.
2. Locate where the drug atoms' numeric dose and rate inputs (zd, zr) enter the atom embedding, and how they are normalised.

If those columns are used only by drug atoms (measurement atoms carry zeros there), scaling them amplifies the model's own learned dose→reading pathway without touching readings. I can then calibrate the scale against dose_edit and check the factual cost. If the columns are shared with measurement values, this lever is unsafe and I'll need another.

```python
for n in list(edited()): drop_edit(n)
print('edited after drop', edited())
cand=[f"{n}{tuple(SHAPES[n])}" for n in NAMES if any(s in n for s in ('numeric','bins','encode_action','embed'))]
print(len(cand)); print(' '.join(cand)[:1800])
print(list(VOCAB.keys()))
nv=VOCAB.get('normalisation') or VOCAB.get('normalization') or {}
print(str({k:v for k,v in nv.items() if any(s in str(k) for s in ('insulin','norepi','rate','dose'))})[:600])
```