No cross effects:
- Insulin has no effect on arterial MAP at any lag (all coefficients below 0.11 in magnitude, se about 0.08).
- Norepinephrine on lab glucose is noisy and consistent with zero (se 1.3–2.0, no consistent sign).

So the edit should touch only insulin→glucose and norepinephrine→MAP. The by-band insulin check failed because I had reused the name `s` for a string in an earlier run, overwriting my one-step dataset.

I will rebuild it as s1 (hours with a fresh finger reading above 150 and a next-hour finger reading), then estimate the one-step jitter effect per glucose band and its ratio to the mean glucose. If the per-unit effect rises in proportion to glucose (roughly constant ratio), insulin acts multiplicatively. Then ×2 and ×½ are still linear in dose, but the truth depends on each patient's glucose level, and the edit's tilt should scale with that level.

```python
s1 = pn[base_mask].copy()
s1 = s1[s1.gf_next.notna()].copy()
s1['dG'] = s1.gf_next - s1.glucose_finger
s1['zj'] = s1.insulin - mtab_full.values[np.clip(np.round(s1.glucose_finger).astype(int),0,599)]
for lo,hi in [(150,165),(165,180),(180,200),(200,220),(220,260),(260,400)]:
    sb = s1[(s1.glucose_finger>lo)&(s1.glucose_finger<=hi)]
    X = np.column_stack([np.ones(len(sb)), sb.glucose_finger, sb.zj])
    bb = np.linalg.lstsq(X, sb.dG.values, rcond=None)[0]; rr = sb.dG.values-X@bb; se_=np.sqrt(np.diag(np.linalg.inv(X.T@X))*rr.var())
    print(f"band {lo}-{hi}: n {len(sb)} one-step coef per unit {bb[2]:.3f} ({se_[2]:.3f}), mean G {sb.glucose_finger.mean():.0f}, coef/G*100 {100*bb[2]/sb.glucose_finger.mean():.3f}")
# pooled interaction model: dG ~ FE(gbin) + zj + zj*(G-190)
s1['gbin'] = (s1.glucose_finger//2).astype(int)
s1['zjG'] = s1.zj*(s1.glucose_finger-190)
b, se, n = fe_reg(s1, 'dG', ['zj','zjG'], 'gbin')
print("pooled: coef zj", b[0].round(3), "(", se[0].round(3), ")  coef zj*(G-190)", b[1].round(5), "(", se[1].round(5), ")")
```