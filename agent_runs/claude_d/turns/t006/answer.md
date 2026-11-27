The panel shows the measurement pattern. Glucose_finger is taken in about 51–59% of hours, and that share rises over the stay along with insulin use (31% → 42% of hours). So measurement is probably more frequent on insulin, which makes it informative. Glucose_lab is in about 20% of hours, map_arterial 46%, map_cuff 61%, and norepinephrine 31–32% of hours. Readings at hh:00 are followed by drug records at hh:05. A dose can therefore react to that hour's reading, and its first possible effect is on the next hour's reading.

Next I will characterise the insulin policy: how starting and the rate depend on the latest glucose reading and the previous rate. I will also check how finger and lab readings agree in the same hour, as a measure of noise. If doses depend only on recorded readings plus independent noise, backdoor adjustment is valid. If there are signs that dosing reacts to unrecorded information, I will need another identification strategy.

```python
P = PT.copy()
P.columns = ['gf','gl','ma','mc','ins','ne']
g = P.groupby(level='stay_id')
# same-hour finger vs lab
both = P[P.gf.notna() & P.gl.notna()]
print('finger-lab same hour n', len(both), 'corr', np.corrcoef(both.gf,both.gl)[0,1], 'diff mean/sd', (both.gf-both.gl).mean(), (both.gf-both.gl).std())
bm = P[P.ma.notna() & P.mc.notna()]
print('art-cuff same hour n', len(bm), 'corr', np.corrcoef(bm.ma,bm.mc)[0,1], 'diff mean/sd', (bm.ma-bm.mc).mean(), (bm.ma-bm.mc).std())
# last-observed glucose (ffill within stay), including current hour
P['gf_last'] = g.gf.ffill(); P['gl_last']=g.gl.ffill()
P['ins0'] = P.ins.fillna(0); P['ne0']=P.ne.fillna(0)
P['ins_prev'] = g.ins0.shift(1); P['ne_prev']=g.ne0.shift(1)
# insulin given vs gf at current hour (only when gf measured at that hour)
cur = P[P.gf.notna()]
bins = np.arange(60,400,20)
tab = cur.groupby(pd.cut(cur.gf,bins)).agg(n=('ins0','size'), p_ins=('ins', lambda x: x.notna().mean()), mean_rate=('ins','mean'))
print(tab.to_string())
```