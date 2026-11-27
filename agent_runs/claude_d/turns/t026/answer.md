Both start tables say the clinician acts on the recorded arterial MAP when it exists and on the cuff MAP otherwise. Below 64 a start is certain, and at a recorded 65 it happens about half the time. The same half-and-half pattern sits at the insulin threshold (16.5% for 148–150), which fits decisions made on the unrounded reading. So the policy uses the recorded measurement, not hidden information.

A few starts happen at 65–75 (2–20%). These may be late starts or restarts, but they are rare and not needed for identification.

Next I need the MAP measurement process before g-estimation, as I did for glucose: does an arterial reading at t depend on norepinephrine at t−1, or on anything else recorded? I will also look at the off-threshold starts against the previous hour's MAP.

```python
c['ma_meas']=c.ma.notna(); c['mc_meas']=c.mc.notna()
m = c[c.index.get_level_values('h')>=1]
print(m.groupby([m.nor_prev>0]).agg(ma=('ma_meas','mean'), mc=('mc_meas','mean')))
print(m.groupby([m.nor_prev>0, m.ins_prev>0]).agg(ma=('ma_meas','mean'), mc=('mc_meas','mean')))
# depends on last MAP value?
m2 = m[m.nor_prev==0].copy(); m2['ma_lastprev'] = m2.groupby(level='stay_id').ma.transform(lambda x: x.ffill().shift(1))
print(m2.groupby(pd.cut(m2.ma_lastprev,[0,60,65,70,75,80,90,200]),observed=True).agg(ma=('ma_meas','mean'), mc=('mc_meas','mean'), n=('ma_meas','size')).round(3))
# per-stay: arterial line stays? fraction of hours with ma per stay
fr = c.groupby(level='stay_id').ma_meas.mean()
print(np.histogram(fr, bins=10, range=(0,1)))
# off-threshold starts: previous hour map
st = c[(c.nor_prev==0)].copy()
st['ma_prevh'] = c.groupby(level='stay_id').ma.shift(1).reindex(st.index)
s = st[st.ma.notna() & (st.ma>66) & (st.ma<75)]
print(s.groupby(s.nor.notna()).ma_prevh.describe())
```