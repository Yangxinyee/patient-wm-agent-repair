The rebuilt sequence matches the first stay exactly: 54 bundles, with the same times and atom counts.
- Each infusion record adds an END atom at its end time, placed in the bundle where the next record starts.
- The final bundles are the two last infusion ENDs at 25.08 and two visit ENDs (hospital and ICU) at 25.17.
- The "has_insulin" flag marks only bundles with an insulin start (WORLD) atom.

Next I'll check the reconstruction on all 100 stays: bundle count, times, atom counts and drug flags against the states file. If everything matches, I can align my rebuilt bundles row-for-row with the model's states. I'll also store the bundles for later use.

```python
bundles_B = {}
mism = 0
for s_ in np.unique(st_sid):
    Bs = make_bundles(tr[tr.stay_id==s_])
    bundles_B[s_] = Bs
    m = st_sid==s_
    hrs = st_hr[m]; na = st_na[m]; hi = st_ins[m]; hn = st_ne[m]
    ok = (len(Bs)==m.sum()) and np.allclose([t for t,_ in Bs], hrs, atol=1e-3) and all(len(a)==n for (_,a),n in zip(Bs,na))
    ok = ok and all((any(x['code']=='insulin' and x['kind']==0 for x in a))==bool(f) for (_,a),f in zip(Bs,hi))
    ok = ok and all((any(x['code']=='norepinephrine' and x['kind']==0 for x in a))==bool(f) for (_,a),f in zip(Bs,hn))
    if not ok: mism += 1
print("stays with mismatch:", mism, "of", len(np.unique(st_sid)))
import pickle; pickle.dump(bundles_B, open('/work/bundles_B.pkl','wb'))
```