Run 49 gives the remaining shapes I need:
- `core.flow.a` gives decay timescales from about 10 minutes (0.16 h) to about 1.9 years, as documented.
- `last_numeric` has 8 trained columns of similar norm, so I can't tell from the weights which inputs are used.
- The code ids are confirmed: 5 map_cuff, 6 glucose_finger, 7 map_arterial, 8 insulin, 9 norepinephrine, 10 glucose_lab.

Next, I will pair each fwd_2 reading with the state after the preceding bundle (fwd_3) and its time gap. I'll also rebuild its last-value register: the previous reading of the same code, its age and its z-score/bin. Then I'll run the replicated head with z at the prior mean, scanning the 15 type indices and two gap scalings (log dt and log dt/10). The documented register order is first try. The success signal is a correlation of 0.99 or more with the tool's predicted means.

```python
X3 = z3['x'].astype(np.float64); S3 = z3['stay_id']; H3 = z3['hour']
# index bundles per stay
from collections import defaultdict
bidx = defaultdict(list)
for i,(s,h) in enumerate(zip(S3,H3)): bidx[s].append(i)
recs = train[train.stay_id.isin(set(S3))]
rows=[]
F = fa2.copy()
lastinfo = {}
meas = recs[recs.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])].sort_values(['stay_id','hour'])
prev_of = {}
for (s,ev),gdf in meas.groupby(['stay_id','event']):
    hs = gdf.hour.values; vs = gdf.value.values
    for j in range(len(hs)):
        prev_of[(s,ev,round(hs[j],4))] = (hs[j-1], vs[j-1]) if j>0 else (np.nan,np.nan)
pk=[]; dts=[]; lh=[]; lv=[]; ok=[]
for s,h,ev in zip(F.stay_id,F.hour,F.event):
    ids = bidx[s]; hrs = H3[ids]
    k = np.where(np.abs(hrs-h)<1e-3)[0]
    if len(k)==0 or k[0]==0: ok.append(False); pk.append(-1); dts.append(np.nan); lh.append(np.nan); lv.append(np.nan); continue
    k=k[0]; pk.append(ids[k-1]); dts.append((hrs[k]-hrs[k-1])*3600); ok.append(True)
    a,b_ = prev_of.get((s,ev,round(h,4)),(np.nan,np.nan)); lh.append(a); lv.append(b_)
F['pk']=pk; F['dt']=dts; F['lasth']=lh; F['lastv']=lv; F['ok']=ok
print(F.ok.mean(), F.dt.describe().round(1).to_dict())
F.to_pickle('/work/F2.pkl')
```