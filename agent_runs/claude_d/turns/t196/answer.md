The smooth exponential tilt helps a lot. NE-only falls from +0.024 to +0.010 overall (arterial +0.013, cuff +0.015). Insulin-only is +0.004 overall (finger glucose +0.015 including the ablation part). But that is still well over budget once both are combined. The remaining cost is the decoders' intrinsic noise, now applied to every reading of a family.

Most of that noise lands on readings with no recent dose, where the true correction is zero and the decoder outputs only noise. The dose edit cannot change a zero dose, so those readings need no injection at all.

A gated unit can compute an AND of two linear gates: M·(family gate + presence gate - 1) is positive only when both are true. I can therefore restrict each channel to readings where its drug actually ran in the previous hour.

Next I check how many readings that keeps and how cleanly presence is separable in with_code. I also refit the decoders on the present-drug rows only, to see whether R² and slope improve there.

```python
ins1 = RD2['ins_1'].fillna(0).values>0; ins2 = RD2['ins_2'].fillna(0).values>0; ne1p = RD2['ne_1'].fillna(0).values>0
print('frac gluc rows ins1', round(float(ins1[rg].mean()),3), 'ins2', round(float(ins2[rg].mean()),3), 'frac MAP rows ne1', round(float(ne1p[rm].mean()),3))
def lin_gate(target_pos, rows=None):
    y = np.where(target_pos, 1.0, -1.0); v = np.linalg.lstsq(Xg, y, rcond=None)[0]
    gq = WCd@v[:192]+v[192]
    return v, gq
v_pi, g_pi = lin_gate(isg & ins1)      # joint target: gluc & ins at t-1
v_pn, g_pn = lin_gate(ism & ne1p)
gf_ = WCd@vg[:192]+vg[192]               # +MAP, -gluc
andI = -gf_ + g_pi - 1; andN = gf_ + g_pn - 1
for nm, a_, pos in [('I', andI, isg&ins1), ('N', andN, ism&ne1p)]:
    print(nm, 'pos min/p1', round(float(a_[:,pos].min()),2), round(float(np.percentile(a_[:,pos],1)),2), 'neg max/p99', round(float(a_[:,~pos].max()),2), round(float(np.percentile(a_[:,~pos],99)),2))
ri = np.where(isg & ins1)[0]; rn = np.where(ism & ne1p)[0]
print('dec on present rows: I', cvdec_aug(ri, RD2.loc[ri,['ei1','ei2']].values, 1.0), 'N', cvdec_aug(rn, RD2.loc[rn,['D_ne_1']].values, 1.0))
```