The file has one row per (stay, hour, edit) at hours 13–24, apparently the hours where the outcome reading (glucose_finger, or map_arterial for norepinephrine) is recorded. "reached" marks hours that come after a drug record starting at hour 12 or later. The main readout is the mean shift over reached rows.

Next I will confirm that the rows match the recorded reading hours in tuning.csv and that "reached" means an infusion started in [12, hour). I will then compute the records-based expected readout on these same rows with my structural impulse response (extended geometrically beyond lag 8 for now). That gives a like-for-like target for comparison with the model's −0.53.

```python
def rows_check(tu, d, ev, edit):
    out=[]
    for sid, g in d[d.edit==edit].groupby('stay_id'):
        rec = set(tu[(tu.stay_id==sid)&(tu.event==ev)&(tu.hour>=13)&(tu.hour<=24)].hour.astype(int))
        out.append(rec==set(g.hour))
    return np.mean(out)
print('gf rows match recorded hours:', rows_check(tu,d1,'glucose_finger','insulin_x2'))
print('map_art rows match:', rows_check(tu,d1,'map_arterial','ne_x2'))
def reached_check(tu, d, drug, edit):
    ok=[]
    for (sid,hr),g in d[d.edit==edit].groupby(['stay_id','hour']):
        st = tu[(tu.stay_id==sid)&(tu.event==drug)&(tu.hour>=12)].hour
        ok.append(bool(((st<hr)).any())==bool(g.reached.iloc[0]))
    return np.mean(ok)
print('reached def (start in [12,hour)):', reached_check(tu,d1,'insulin','insulin_x2'), reached_check(tu,d1,'norepinephrine','ne_x2'))
hstruct = np.array([-1.947,-2.29,-1.942,-1.505,-1.153,-0.836,-0.553,-0.384])
hext = np.concatenate([hstruct, hstruct[-1]*0.7**np.arange(1,6)])  # lags 1..13
def true_readout(tu, d, drug, edit, factor, h):
    sh=[]
    for (sid,hr),g in d[(d.edit==edit)&(d.reached)].groupby(['stay_id','hour']):
        recs = tu[(tu.stay_id==sid)&(tu.event==drug)&(tu.hour>=12)&(tu.hour<hr)]
        sh.append(sum(h[int(hr-np.floor(t))-1]*(factor-1)*r for t,r in zip(recs.hour,recs.rate)))
    return np.mean(sh), len(sh)
print('records-based insulin x2:', true_readout(tu,d1,'insulin','insulin_x2',2,hext), ' x1/2:', true_readout(tu,d1,'insulin','insulin_half',0.5,hext))
```