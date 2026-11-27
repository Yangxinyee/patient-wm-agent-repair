The insulin checks agree with everything else.

- **Longer horizons.** The tier discontinuity gives −0.22 ± 0.19 mg/dL per u/h at 6 h and +0.06 ± 0.25 at 9 h.
- **On/off at 150.** Below 150 the next reading is measured with a flat 29% probability, independent of glucose and hour, so the comparison is valid. Starting about 1 u/h gives −0.30 ± 0.32 at 1 h, +0.39 ± 0.32 at 2 h and +0.24 ± 0.34 at 3 h.

So the records show no insulin effect on glucose at any horizon, whether dose size or on/off, by every method. Together with the clean norepinephrine titration result, my conclusion is that both drugs are inert on their targets in this ICU. The four true readouts should be about 0.

The original model is close already, with insulin +0.14 / −0.08 and norepinephrine about 0.01. Its small insulin shift is statistically real (about 3σ) and goes in the confounded direction.

The cleanest repair is to make drug atoms blind to dose size. The edit changes only the z-scores and bins of drug atoms. If those embedding terms no longer depend on dose, the edited and recorded rollouts become identical, and every shift becomes exactly 0. Readings never use drug bins, so the value heads are untouched except through the state.

First I will record my conclusions in the notebook and inspect the tensors involved: the z_dose and z_rate columns of `embed.numeric`, and the rows of `embed.bins.dose` and `embed.bins.rate`.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Causal analysis (runs 13-32)
- Measurement: finger = G + noise(sd~9-10), lab = G + noise(sd~3); art MAP noise sd~3, cuff ~6 (+3 offset).
- Insulin: dose noise iid (lag-1 corr 0.002, no stay ICC), independent of lab-finger. Effect of dose (FE on reading) on R_t+k, k=1..9: all ~0 (e.g. k1 +0.034+-0.043).
  Tier RD (180,220,260,300) pooled per-unit: k1 +0.03+-0.12, k2 -0.12+-0.13, k3 -0.17+-0.15, k6 -0.22+-0.19, k9 +0.06+-0.25.
  On/off RD at 150 (meas below 150 random p=0.29): k1 -0.30+-0.32, k2 +0.39+-0.32. Lab outcome null. All subgroups null.
  => INSULIN HAS NO EFFECT ON GLUCOSE (|per-unit| < ~0.1-0.2).
- NE: titration rule on MAP (art if present else cuff): <65 up (+0.18 log), 65-75 hold, >75 down (-0.28/-0.36) & some stops; starts whenever MAP<65 (deterministic).
  Titration surprise (joint FE MAP x prev-rate) independent of past MAP & cuff-art; effect on MAP k1..6 ~0 (k1 -0.04+-0.09). All subgroups null.
  BUT start dose depends on history/hidden severity (prev MAP, hour); start dose predicts later MAP fall (confounded, -9.8/unit, -1.9 after prev-MAP FE).
  => NE HAS NO CAUSAL EFFECT ON MAP; NE level is a confounded severity marker.
- Original model: per-stay NE shifts sd 0.07 (ignores NE); insulin small +0.11 (3 sigma), x2/half correlated 0.67 (noise).
- PLAN: make drug atoms magnitude-blind (embed.numeric z_dose,z_rate cols -> 0; bins.dose/rate rows 1..16 -> row 8 (median)). Expect shifts exactly 0. Check NLL.
""")
W = load('embed.numeric.weight'); print("numeric col norms:", np.linalg.norm(W,axis=0).round(3), " bias norm", np.linalg.norm(load('embed.numeric.bias')).round(3))
for n in ['embed.bins.dose.weight','embed.bins.rate.weight','embed.bins.value.weight']:
    T = load(n); print(n, "row norms:", np.linalg.norm(T,axis=1).round(2))
    print("   dist of rows to row 8:", np.linalg.norm(T - T[8], axis=1).round(2))
for n in ['embed.code.weight','embed.kind.weight','embed.categorical.rate_unit.weight']:
    print(n, np.linalg.norm(load(n),axis=1).round(2))
```