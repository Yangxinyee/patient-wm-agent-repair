Every subgroup is null.

- **Insulin.** No effect in any tier, before or after hour 12, or at 1, 3 or 6 h. Estimates lie within about ±0.2 mg/dL per u/h, and the lab-glucose outcome agrees (+0.03 ± 0.07).
- **Norepinephrine.** No effect by MAP band, hour range or previous-rate level. The largest estimate is −0.6 ± 0.4, in a small subgroup, which is consistent with noise.

The current belief is that within the dose ranges seen in training, neither drug moves its target. So the true readouts should be about 0, close to what the original model already gives.

Two checks remain before I decide what, if anything, to edit:

1. On/off natural experiments for norepinephrine: random starts at MAP below 65 and random stops at MAP above 75. These involve large relative dose changes and test a saturating effect.
2. Whether doses in hours with no recorded reading reveal unrecorded readings, which the model may exploit. This determines how risky it is, for factual fit, to remove dose information.

A real start or stop effect would revise the null conclusion, and dose-informative no-reading hours would warn me about the factual-fit cost of an edit.

```python
# NE starts: ne_p1==0, decide start at t, given MAP_t (arterial or cuff)
St = A2[(A2.ne_p1==0)].copy()
St['x'] = St.map_arterial.where(St.map_arterial.notna(), St.map_cuff)
St['xtype'] = np.where(St.map_arterial.notna(),'art',np.where(St.map_cuff.notna(),'cuff','none'))
St = St[(St.xtype!='none')&(St.x<65)].copy()
for k in [1,2,3]:
    fa = maP.shift(-k).reindex(St.index).values; fc = mcP.shift(-k).reindex(St.index).values
    St[f'y{k}'] = np.where(St.xtype=='art', fa, fc) - St.x.values
St['start'] = (St['ne']>0).astype(float)
St['grp'] = St.xtype + "_" + St.x.round().astype(int).astype(str)
print("start rate", St.start.mean().round(3), " mean start dose", St.loc[St.start==1,'ne'].mean().round(3))
for k in [1,2,3]:
    b,se,n = fe_slope(St[f'y{k}'].values, St['ne'].values, St.grp.values, St.stay.values)
    b2,se2,n2 = fe_slope(St[f'y{k}'].values, St.start.values, St.grp.values, St.stay.values)
    print(f"starts k={k}: per-unit NE {b:+.2f}+-{se:.2f} | start vs not {b2:+.3f}+-{se2:.3f}  n={n}")
# NE stops at MAP>75 among ne_p1>0
Sp = C[(C.x>75)].copy()
Sp['stop'] = (Sp['ne']==0).astype(float)
Sp['grp2'] = Sp.xtype + "_" + Sp.x.round().astype(int).astype(str) + "_" + (Sp.ne_p1/0.05).round().astype(int).astype(str)
for k in [1,2]:
    b,se,n = fe_slope(Sp[f'y{k}'].values, Sp['ne'].values, Sp.grp2.values, Sp.stay.values)
    print(f"MAP>75 k={k}: per-unit NE {b:+.2f}+-{se:.2f} n={n}  (stop rate {Sp.stop.mean():.3f})")
# no-MAP hours: does NE titration predict the next (hidden) MAP?
N0 = A2[A2.map_arterial.isna()&A2.map_cuff.isna()&(A2.ne_p1>0)&(A2['ne']>0)].copy()
N0['lr'] = np.log(N0['ne']/N0.ne_p1)
N0['M_last'] = P.map_arterial.groupby(level=0).ffill().groupby(level=0).shift(1).reindex(N0.index).values
N0['M_next'] = maP.shift(-1).reindex(N0.index).values
m = N0.M_next.notna() & N0.M_last.notna()
print("no-MAP hours: corr(lr, last MAP)", np.corrcoef(N0.lr[m], N0.M_last[m])[0,1].round(3), " corr(lr, next MAP)", np.corrcoef(N0.lr[m], N0.M_next[m])[0,1].round(3), " n", m.sum())
```