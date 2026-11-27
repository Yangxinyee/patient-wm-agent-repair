The last run showed that `tensors.json` holds only shapes, with no descriptions. So the meaning of the 8 `last_numeric` inputs can't be checked from the files.

The vocab config adds a little: 4 heads, one prelude layer, two recurrent layers and one coda layer. Each drug-bundle state step therefore goes through a single prelude attention, where u_k is one token among seven.

That leaves one input-side lever to settle: the z_dose/z_rate columns of `embed.numeric`. They are only non-zero on drug atoms, so their edits are drug-selective by construction. At 5x they moved insulin x2 only from −1.85 to −2.54 (true about −13.6). That small gain has two possible explanations:
- **Saturation:** the action encoder's layer norm is already saturated, so a bigger gain can only rotate u_k a little further.
- **Weak downstream:** the downstream pathway is weak but roughly linear, and a larger input gain would keep raising the response.

I will set both columns to 20x (alpha = 19) and run a dose_edit on the same 100 tuning stays.

| Insulin x2 readout at 20x | Interpretation |
|---|---|
| About −2.5 to −3.5 | Input-side gain is capped; the lever must move downstream |
| Clearly larger (−5 or more) | The input route is still usable; the next step measures its factual cost |

I'll also watch whether the insulin x1/2 readout keeps the right (positive) sign.

```python
clear_ops()
e2v = np.zeros(6); e2v[2] = 1.0
e4v = np.zeros(6); e4v[4] = 1.0
rank1('embed.numeric.weight', e2v, 19.0, side='in')
rank1('embed.numeric.weight', e4v, 19.0, side='in')
print(ops())
ids100 = [2020000000 + i for i in range(100)]
print(json.dumps({"type": "dose_edit", "model": "edited", "split": "tuning", "stays": ids100})[:200])
```