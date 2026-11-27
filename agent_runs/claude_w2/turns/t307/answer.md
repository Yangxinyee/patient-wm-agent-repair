Candidate 1 result:

| Edit | Original | Edited | Truth |
|---|---|---|---|
| norepinephrine x2 (MAP) | +1.08 | +2.66 | +4.7 |
| norepinephrine x1/2 (MAP) | −0.73 | −1.48 | −2.4 |
| insulin x2 (glucose) | −2.14 | −2.22 | −14.1 |
| insulin x1/2 (glucose) | +1.26 | +1.39 | +7.0 |

The factual NLL improved by 0.0021. So the numeric dose features drive the model's norepinephrine response, while its insulin response must come mainly through the rate and dose bin embeddings. Insulin rates are discrete 0.5 steps, so the bin path likely dominates for them.

Candidate 2 keeps numeric ×3 and also amplifies the rate and dose bin embeddings by 3 around their mean, for bins 1–16 only. Bin 0 means "no rate" and is used by measurements, so it stays untouched. The bin tables are shared by both drugs, so comparing with candidate 1 isolates the bin path's effect on each readout. The factual check comes first.
- **Confirmed:** a much larger insulin response, with the NLL change still ≤ 0.005.
- **Refuted:** the insulin response barely moves, or the fit breaks.

```python
nm_b = [n for n in NAMES if n.startswith('embed.bins.')]; print({n: SHAPES[n] for n in nm_b})
for n in ['embed.bins.rate.weight', 'embed.bins.dose.weight']:
    E = np.asarray(load_original(n), dtype=np.float32).copy(); ref = E[1:17].mean(0)
    E[1:17] = ref + 3.0 * (E[1:17] - ref); save_edit(n, torch.as_tensor(E))
print("edited:", edited())
with open('/work/notes.md', 'a') as f:
    f.write("- fwd5/6 (sel1 tuning): numeric z_dose/z_rate cols x3 -> factual NLL 1.52793->1.52582 (better); ins x2 -2.22 (orig -2.14), x1/2 +1.39; NE x2 +2.66 (orig 1.08, truth 4.7), x1/2 -1.48 (truth -2.4).\n"
            "- cand2: + bins.rate/dose rows1-16 deviations x3 (ref mean) to reach insulin path.\n")
```