With fully flexible fixed effects on (exact MAP, previous-rate bin), the random titration surprise shows no effect of norepinephrine on later MAP. The estimate is −0.04 ± 0.09 mmHg per mcg/kg/min at 1 h and stays within ±0.2 out to 6 h. The surprise is unrelated to earlier MAP readings (coefficients of about 1e-4), so it behaves as randomised. The earlier −1.4 was confounding from the additive specification.

Both noise-based analyses therefore say the drugs do nothing, and the original model already answers about 0. Before I accept that, I must rule out one trap. The clinician might use hidden information, such as the true glucose or MAP behind a noisy reading, inside the "noise". That would bias noise-based estimates toward zero and could mask a real effect.

Next I will test this directly:

- **Insulin.** Does the dose noise correlate with (lab − finger) in the same hour? The lab reading is a more precise measure of true glucose.
- **Norepinephrine.** Does the titration surprise correlate with (cuff − arterial)?

I will also run the threshold-based discontinuity estimates for norepinephrine at MAP 65 and 75, which hidden information cannot bias. Clean zeros on all of these would make "no effect" solid.

```python
# 1) insulin dose noise vs hidden glucose (lab-finger), same hour
Ig2 = Q[(Q.hr>=0)&(Q.ins>0)&(Q.tier>0)&Q.glucose_lab.notna()].copy()
Ig2['lr'] = np.log(Ig2.ins/Ig2.tier.map(base))
Ig2['lr_res'] = Ig2.lr - Ig2.groupby(Ig2.glucose_finger.round())['lr'].transform('mean')
b,se = ols(Ig2.lr_res.values, np.column_stack([np.ones(len(Ig2)), (Ig2.glucose_lab-Ig2.glucose_finger).values]), Ig2.stay.values)
print("insulin noise ~ (lab-finger): slope", b[1].round(5), "+-", se[1].round(5), " n", len(Ig2))
# 2) NE surprise vs (cuff - arterial)
B2 = B[B.map_cuff.notna()]
b,se,n = fe_slope(B2.s.values, (B2.map_cuff-B2.map_arterial).values, B2.grp.values, B2.stay.values)
print("NE surprise ~ (cuff-art): slope", round(b,6), "+-", round(se,6), " n", n)
# 3) NE RD at 65 and 75 thresholds (arterial reading as running var), outcome MAP change, dose = NE_t
B['x'] = B.map_arterial
def rd_ne(B, c, h, ycol):
    S = B[(B.x>=c-h)&(B.x<=c+h)&(B.x!=c)].copy(); S = S[S[ycol].notna()]
    x = S.x.values-c; a=(x>0).astype(float)
    X = np.column_stack([np.ones(len(S)), a, x, a*x, S.ne_p1.values, S.ne_p1.values*a])
    by,sey = ols(S[ycol].values, X, S.stay.values); bd,sed = ols(S['ne'].values, X, S.stay.values)
    return by[1],sey[1],bd[1],len(S)
for k in [1,2,3]:
    B[f'dM{k}'] = B[f'Mf{k}'] - B.map_arterial
    for c in [65,75]:
        jy,se,jd,n = rd_ne(B, c, 8, f'dM{k}')
        print(f"RD c={c} k={k}: jump y {jy:.3f}+-{se:.3f}, jump NE {jd:.4f}, per-unit {jy/jd:.2f}+-{se/abs(jd):.2f} n={n}")
```