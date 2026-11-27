Two findings matter here. First, the predicted mean of a reading is Σ p(b)·centre[b], and the glucose_finger centres run only from 73 to 272. A reading in the top bin (≥ ~262) therefore cannot show a downward shift unless probability mass leaves that bin. Second, z = (x − median)/IQR per code, clipped to ±10. That tells me how a doubled dose shifts the insulin and NE input features.

Next I will print the remaining numeric entries (glucose_finger edges, insulin and NE rate/dose medians, IQRs and edges). I will also check from the training grid how often hours 13–24 glucose readings in insulin stays fall in the top bin. If that share is large, the model's readout will understate the true effect, and I need to plan for it.

```python
for e in VOCAB['numeric']:
    print(e)
import pickle
stays_tr,Gtr,stays_tu,Gtu = pickle.load(open('/work/grid.pkl','rb'))
gf = Gtr['gf']; ins = Gtr['ins']; ne = Gtr['ne']; ma = Gtr['ma']
hrs = list(range(13,25))
has_ins_after12 = (ins.loc[:, [h for h in ins.columns if h>=12]].fillna(0)>0).any(axis=1)
v = gf.loc[has_ins_after12, hrs].values.ravel(); v = v[~np.isnan(v)]
edges_gf = [e for e in VOCAB['numeric'] if e['code']=='glucose_finger' and e['attribute']=='value'][0]['fine_bin_edges']
print('gf readings h13-24 in insulin stays:', len(v), 'frac > last edge', edges_gf[-1], ':', np.mean(v>edges_gf[-1]).round(4), ' frac < first edge', np.mean(v<edges_gf[0]).round(4))
has_ne_after12 = (ne.loc[:, [h for h in ne.columns if h>=12]].fillna(0)>0).any(axis=1)
w = ma.loc[has_ne_after12, hrs].values.ravel(); w = w[~np.isnan(w)]
print('ma readings h13-24 in NE stays:', len(w), 'frac>98:', np.mean(w>98).round(4), 'frac<35:', np.mean(w<35).round(4))
```