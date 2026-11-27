The vocab notes match my binning and z-score code exactly. The codes table also shows the units: mg/dL for glucose, mmHg for MAP, units/hour and mcg/kg/min for the drugs. One measurement-side choice I guessed and never checked against core states is which unit-table row each unit uses (rows 2 to 5 are the trained ones).

Next I will scan all assignments of rows 2 to 5 to mg/dL and mmHg, keeping the drug rows fixed, and score each by the h relative error on measurement bundles of s0.

- **Confirms:** one assignment cuts the measurement-bundle error toward the action-bundle level (about 0.04).
- **Refutes:** all assignments give similar errors. Then the cause lies elsewhere, likely the time-gap or memory handling.

```python
def eval_stay_rel(B,rows,EU,only_meas=True):
    out_=[]
    for k in range(1,len(B)):
        ismeas=not any(a['ek']==2 for a in B[k][1])
        if only_meas and not ismeas: continue
        r_=rows[k]; hp=torch.tensor(SX[rows[k-1],192:960].reshape(4,192),dtype=torch.float64)
        zk=torch.tensor(SX[r_,960:],dtype=torch.float64); idx_=list(range(k+1))[-128:]
        Mem=torch.stack([EU[j][0] for j in idx_]); Mage=torch.tensor([(B[k][0]-B[j][0])*3600 for j in idx_],dtype=torch.float64)
        y,o,E=core_step_v(hp,(B[k][0]-B[k-1][0])*3600,EU[k][0],EU[k][1],zk,Mem,Mage,'std')
        ht=torch.tensor(SX[r_,192:960].reshape(4,192),dtype=torch.float64)
        out_.append(float((y.mean(0)-ht.mean(0)).norm()/ht.mean(0).norm()))
    return np.median(out_)
ev0=recs[recs.stay_id==s0].sort_values('hour')
res_u=[]
for ug in [2,3,4,5]:
    for um in [2,3,4,5]:
        if ug==um: continue
        OPTS['unit'].update({'glucose_finger':ug,'glucose_lab':ug,'map_arterial':um,'map_cuff':um})
        B_=build_bundles(ev0,OPTS); EU_=[enc_eu(al) for _,al in B_]
        res_u.append((round(eval_stay_rel(B_,rows0,EU_),4),ug,um))
res_u.sort(); print(res_u)
OPTS['unit'].update({'glucose_finger':2,'glucose_lab':2,'map_arterial':3,'map_cuff':3})
```