The CSV has one row per stay-hour and edit, with shift and a reached flag. Next I'll join the NE x2 and x1/2 rows with each stay's recorded NE rates from hour 12 on. Then I'll compare edited and original shifts for stays whose doubled rate stays at or below 1.0 against stays whose doubled rate goes above it.

```python
import pickle
stays_tr,Gtr,stays_tu,Gtu = pickle.load(open('/work/grid.pkl','rb'))
ne = Gtu['ne']
cols12 = [c for c in ne.columns if 12 <= c <= 24]
mx = ne[cols12].max(axis=1)
def summ(df, edit):
    r = df[(df.edit==edit) & df.reached].copy()
    r['mx'] = r.stay_id.map(mx)
    r['grp'] = np.where(r.mx <= 0.5, 'in', 'out')
    return r.groupby('grp')['shift'].agg(['mean','count']).round(2), r.groupby('grp').stay_id.nunique()
for edit in ['ne_x2','ne_half']:
    for nm, df in [('orig', d10o), ('edit', d10e)]:
        g, n = summ(df, edit)
        print(edit, nm, g.to_dict(), 'stays', n.to_dict())
```