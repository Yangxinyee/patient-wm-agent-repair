The regression-discontinuity check agrees with the noise-based estimates.

| Threshold | Dose jump (u/h) | Glucose change per u/h, k = 1, 2, 3, 4 h later |
|---|---|---|
| 180 | +1.035 | −1.88, −2.10, −1.47, −0.91 |
| 220 | +1.05 | −2.60, −2.63, −2.23, −1.96 |

The noise-based values at those levels are about −1.9 to −2.1 at 1 h falling to about −1.3 to −1.4 at 4 h, which agrees within about 1–2 SE. So I find no hidden confounding of the dose noise. The insulin effect is causal and identified by both designs. The 220 result again suggests the effect is somewhat stronger at higher glucose.

A two-exponential form fits the deconvolved insulin response well: h(k) = −β(ρ^k − λ^k)/(ρ − λ), with β ≈ 2.08, ρ ≈ 0.75 and λ ≈ 0.41. That points to fast insulin action plus mean reversion of glucose.

Next I'll look at the tuning stays: how many receive insulin or norepinephrine from hour 12 on, and their typical doses. Then I'll choose 100 informative stays for the first forward pass, which shows what the original model currently predicts for all four edits.

```python
Pt = Ptu.reset_index().sort_values(['stay_id','h'])
tu_ids = Pt.stay_id.unique()
info = Pt.groupby('stay_id').apply(lambda s: pd.Series({
    'ins12': s[(s.h>=12)].insulin.notna().sum(),
    'ne12': s[(s.h>=12)].norepinephrine.notna().sum(),
    'art': s.map_arterial.notna().sum(),
    'ins_mean12': s[(s.h>=12)].insulin.mean(),
    'ne_mean12': s[(s.h>=12)].norepinephrine.mean(),
    'gf13': s[(s.h>=13)].glucose_finger.notna().sum()}))
print(info.describe().to_string())
print("ins after 12:", (info.ins12>0).sum(), " ne after 12:", (info.ne12>0).sum(), " ne&art:", ((info.ne12>0)&(info.art>0)).sum(), " both drugs:", ((info.ins12>0)&(info.ne12>0)).sum())
```