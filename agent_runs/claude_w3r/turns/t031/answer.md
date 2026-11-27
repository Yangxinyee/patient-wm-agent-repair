The insulin effect per u/h is fairly stable across bands: −1.62 (rates ≈ 1), −1.99 (≈ 2), −2.20 (≈ 3), −1.75 (≈ 4, noisy). The quadratic term is negligible (−0.08 ± 0.04). A log-dose model would predict steeply falling per-unit slopes, which the data reject, so a model linear in rate is appropriate for ×2 and ×1/2. The slight rise with band is consistent with an effect proportional to glucose (about −0.010 × G per u/h): (−1.62, −1.99, −2.20) ÷ (165, 197, 235) ≈ −0.0098, −0.0101, −0.0094. That is a refinement for computing the target.

Next I will turn to the edit itself and list the model's tensors and shapes. I need to see how the value head is built and where an insulin- or norepinephrine-specific signal could be injected.

```python
with open('/work/notes.md','a') as f:
    f.write("""- INSULIN refined (clean sample: gf measured at t, g>=155 so always on; band cell-mean deviation; IPW for meas at s>=2 (meas=1 if ins prev hour else 0.30 random)):
  struct IRF lags1..12: -2.0,-2.45,-2.1,-1.7,-1.3,-1.0,-0.7,-0.5,-0.35,-0.25,-0.25,-0.3 ; cum -12.6..-13.2 by lag12. Robust to controls & HGB-x.
  (full-sample HGB version gave positive tail -> contaminated by on/off region; distrust.)
  Lag-1 slope by band: -1.62(r~1),-1.99(r~2),-2.20(r~3),-1.75(r~4); quad term ~0 -> LINEAR in rate (log model rejected).
  Slope/G ~ -0.0098,-0.0101,-0.0094 -> effect maybe ∝ G*rate.
""")
print(len(NAMES))
for nm in NAMES: print(nm, SHAPES[nm])
```