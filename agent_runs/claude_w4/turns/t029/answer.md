## Step 28: the refined surprises are clean for both drugs

**What the last result showed.**
- The cross-fitted expected-dose models remove the correlation between surprise and the current reading inside every band (all |r| ≤ 0.009).
- The earlier insulin surprise had a small within-band correlation (r = 0.07 in band 1), so insulin dosing also depends a little on glucose inside a band.
- The norepinephrine surprise SD is 0.073 (the expected-rate model has R² = 0.94); for insulin it is 0.36.

**What I will do next.** Re-estimate the 12-lag impulse responses with these clean instruments:
- insulin → lab glucose (selection-free) and → finger glucose;
- norepinephrine → arterial MAP (arterial-line stays) and → cuff MAP (non-arterial stays).

**What would count as evidence.**
- The insulin response should stay close to before.
- For norepinephrine, a positive response means it raises MAP; a flat zero means no effect.
- Consistency across the two MAP readings tells me whether the effect read on cuff equals the effect read on arterial, which the evaluation also scores.

```python
# merge improved surprises
xs2 = xs.merge(di[['stay_id','h','e_ins2']], on=['stay_id','h'], how='left').sort_values(['stay_id','h'])
xs2['e_ins2'] = xs2.e_ins2.fillna(0)
g2 = xs2.groupby('stay_id')
for m in range(0,15):
    xs2[f'e_l{m}'] = g2.e_ins2.shift(m)
bL, seL, n = irf(xs2, 'glucose_lab', 12)
print("INS->LAB K=12 n",n); print("  h:", np.round(bL,2).tolist()); print("  se:", np.round(seL,2).tolist()); print("  cum:", np.round(np.cumsum(bL),2).tolist())
bF, seF, n = irf(xs2, 'glucose_finger', 12)
print("INS->FINGER K=12 n",n); print("  h:", np.round(bF,2).tolist()); print("  se:", np.round(seF,2).tolist()); print("  cum:", np.round(np.cumsum(bF),2).tolist())
# NE
xn2 = xn.merge(dn[['stay_id','h','e_ne2']], on=['stay_id','h'], how='left').sort_values(['stay_id','h'])
xn2['e_ne2'] = xn2.e_ne2.fillna(0)
gq = xn2.groupby('stay_id')
for m in range(0,15):
    xn2[f'I_l{m}'] = gq.ne_rate.shift(m)
    xn2[f'e_l{m}'] = gq.e_ne2.shift(m)
    xn2[f'lg_l{m}'] = gq.lm_art_pref.shift(m)
def irf_ne(df, ycol, K):
    cols = [f'I_l{m}' for m in range(1,K+1)] + [f'e_l{m}' for m in range(1,K+1)] + [f'lg_l{K}', f'I_l{K+1}']
    d = df[df[ycol].notna()].dropna(subset=cols)
    endog = d[[f'I_l{m}' for m in range(1,K+1)]].values
    instr = d[[f'e_l{m}' for m in range(1,K+1)]].values
    lK = d[f'lg_l{K}'].values
    exog = np.column_stack([np.ones(len(d)), lK, np.maximum(lK-65,0), np.maximum(lK-75,0), d[f'I_l{K+1}'].values])
    b, se = tsls(d[ycol].values, endog, instr, exog)
    return b[:K], se[:K], len(d)
for yc, sel in [('map_arterial', xn2.art_stay), ('map_cuff', ~xn2.art_stay), ('map_cuff', xn2.art_stay)]:
    b, se, n = irf_ne(xn2[sel], yc, 12)
    print("NE->", yc, "artstay" if sel.iloc[0]==True or yc=='map_arterial' else "", "n", n); print("  h:", np.round(b,2).tolist()); print("  se:", np.round(se,2).tolist()); print("  cum:", np.round(np.cumsum(b),2).tolist())
```