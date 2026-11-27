Run 293 explains where the factual cost comes from: noise in the norepinephrine reader, not a fixable offset.

| Readings | Mean shift (offset) | Shift sd | Correlation with model error | Cost |
|---|---|---|---|---|
| map_arterial | +0.19 | 1.8 | 0.09 | +0.044 |
| map_cuff | −1.0 | 3.5 | ~0 | +0.083 |

- **What drives the cost:** a Gaussian check with plausible predictive variances (sd about 5 for arterial, 8 for cuff) roughly reproduces both costs. So the cost is noise in the shift itself. The offsets are small.
- **Cuff:** the tilt is shared across codes, and cuff predictions are wider, so cuff shifts are about 2.4 times larger per reader unit. That is why cuff carries most of the cost.
- **Re-centring:** it barely helps. The predicted per-value cost after re-centring is about 0.0005 at a quarter of the current gain, 0.003 at half, and 0.015 at full gain.
- **Rollout shift vs noise:** under ×2 the reader moves arterial MAP by about 1.1 mmHg on average. Its natural noise is 1.8 mmHg sd, so the ratio is about 0.6.
  - Reaching the +4.6 target would need roughly 10 times cleaner information than the model's state holds.
  - The reader can therefore only make a small, safe improvement.

What I will do:
- Scale the gain down to κ = 0.5 (one third of now), without re-centring. The predicted factual cost is about 0.001 nats per value, well inside the 0.005 limit with room for differences on unseen stays.
- In the rollout this should give norepinephrine ×2 ≈ +0.7 and ×½ ≈ −0.11, against +0.34 and −0.06 for the original.
- I will check this with a factual pass and a dose-edit pass.

If the measured cost comes out above about 0.002, I will lower κ further or drop the reader.

```python
_set_kappa_ne(0.5)
_w = np.asarray(load('attr_heads.value_bin.2.weight'))
print('kappa set 0.5: col120 norm %.3f col110 norm %.3f (was 3.684), sum max %.2e' % (np.linalg.norm(_w[:, 120]), np.linalg.norm(_w[:, 110]), np.abs(_w[:, 120] + _w[:, 110]).max()))
print('edited', edited())
with open('/work/notes.md', 'a') as fh:
    fh.write("- run293 (fwd15 csv analysis): NE reader shift d: art mean +.19 sd 1.80 corr(d,res) .09; cuff mean -1.01 sd 3.46 corr .005. Cost ~ Var(d)/(2V) (V art~28, cuff~67): it is reader NOISE, not offset; "
             "cuff gets 2.4x larger shift per reader unit (shared tilt, wider predictive). Recentred cost pred per value: f.25 .0005, f.5 .003, f1 .015. Reader SNR in rollout (1.1 mmHg x2 vs 1.8 noise) ~.6 -> full NE target infeasible.\n"
             "- run294: kappa -> 0.5 (f=1/3), no recentring. Expect factual +~.001, NE x2 ~+.7, half ~-.11. NEXT: fwd16 factual both, fwd17 dose_edit both (tuning100).\n")
```