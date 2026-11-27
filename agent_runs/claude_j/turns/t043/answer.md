The training data contradict my notes. A one-unit/h dose surprise lowers the next-hour glucose_finger by 1.0 ± 0.04, not 2.0, and lag 2 gives -0.93. By lags 4–6 the slope turns positive, as expected: lower glucose means less sliding-scale insulin later, then a rebound.

The lag-1 slope has no such feedback, so a -1.0 vs -2.0 gap must come from how the deviation is defined or from my old estimates. That matters a lot: the true effect could be half what my truth table assumes.

Two things could bias the regression toward zero. Some readings above 150 may have no insulin record, which gives large spurious negative deviations. And the dose may sometimes follow a reading other than the one at that hour. I'll check how often a reading above 150 has no insulin. I'll redo the lag-1 slope using only rows where insulin was given, by band, with deviations defined from the actual multiplier, and cross-check with a regression discontinuity at 150/180/220.

If the restricted slope moves to about -2, my notes stand and the -1.0 was contamination. If it stays at -1, the truth table must be halved.

```python
B = tr[tr.event=='glucose_finger'][['stay_id','hour','value']]
B = B[np.isclose(B.hour, B.hour.round(0))].copy(); B['h']=B.hour.round(0).astype(int)
ins_ = tr[tr.event=='insulin'][['stay_id','hour','rate','dose']].copy()
ins_['h'] = (ins_.hour-0.0833).round(0).astype(int)
print('insulin hour frac offsets', ((ins_.hour - ins_.hour.round(0))).round(3).value_counts().head())
print('dose==rate?', np.allclose(ins_.dose, ins_.rate))
B = B.rename(columns={'value':'g0'})[['stay_id','h','g0']].merge(ins_[['stay_id','h','rate']], on=['stay_id','h'], how='left')
B['given'] = B.rate.notna(); B['rate']=B.rate.fillna(0)
B['base']=base_of(B.g0.values)
print(pd.crosstab(B.base, B.given))
gfx = tr[tr.event=='glucose_finger'][['stay_id','hour','value']]; gfx=gfx[np.isclose(gfx.hour,gfx.hour.round(0))].copy(); gfx['hk']=gfx.hour.round(0).astype(int)
C = B.assign(hk=B.h+1).merge(gfx[['stay_id','hk','value']].rename(columns={'value':'y'}), on=['stay_id','hk'])
Cg = C[(C.base>0)&(C.given)].copy()
Cg['m'] = Cg.rate/Cg.base
for bb in [1,2,3,4,6]:
    c = Cg[Cg.base==bb]
    X = np.column_stack([np.ones(len(c)), c.g0, c.rate])
    b,*_ = lstsq(X, c.y.values, rcond=None)
    print('band', bb, 'n', len(c), 'slope on rate (given g0) %.3f' % b[2], 'mean m %.3f sd log m %.3f' % (c.m.mean(), np.log(c.m).std()))
Cg['dev'] = Cg.rate - Cg.base.map(mb)
print('pooled given-only reg slope %.3f se %.3f' % reg(Cg.y.values, Cg))
```