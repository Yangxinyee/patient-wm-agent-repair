The generator code shows how the cf data was built. run_cf(s, {drug: m}, start_min) replays the stay with scaled doses, and heads_full produces with_code and logits per reading. mod_events_new reads a global SHIFT dict to shift readings.

An exact-cf dataset would need about 13 replays per stay, and an earlier timing suggests tens of seconds each. That is too slow for 100 stays.

There is a cheaper check first. At hour 13, the first reading after the dose change, the rollout prediction depends only on the record up to hour 12 plus the changed dose. So it must equal the teacher-forced prediction. I compare the tool's hour-by-hour shifts (edited minus original, fwd_6) with my replica's teacher-forced edit effect at the same hours.
- **Match at hour 13, divergence later:** the edit is implemented correctly, and the gap comes from rollout context: the readout reads surprise, or feedback through the last-bin centring.
- **Mismatch already at hour 13:** the implementation or transfer is wrong.

```python
d6o=pd.read_csv('/work/fwd_6/dose_edit_original.csv'); d6e=pd.read_csv('/work/fwd_6/dose_edit_edited.csv')
for e_ in ['insulin_x2','ne_x2']:
    a=d6o[(d6o.edit==e_)&(d6o.reached==1)].groupby(d6o.hour.round().astype(int))['shift'].mean(); b=d6e[(d6e.edit==e_)&(d6e.reached==1)].groupby(d6e.hour.round().astype(int))['shift'].mean()
    print(e_,'tool extra (ed-orig) by hour',np.round((b-a).values[:12],2).tolist())
for nm,c_,P,v_,ev0 in [('INS',cf,P_ins,vG,'glucose_finger'),('NE',cN,P_ne,vM,'map_arterial')]:
    k=int(np.argmax(np.abs(v_))); gk=P['col'][k]/v_[k]; mk=((c_.h0==12)&(c_.m==2.0)&(c_.event==ev0)).values; cc=c_[mk]
    um_=P['um']-P['uc']; xc=WC[cc.ri.values]@um_; xf=WC[cc.fi.values]@um_; th=gk*(xc-xf)
    c17=np.stack([cenof(e) for e in cc.event])[:,None,:]; lg=LG[cc.ri.values]; m0,_=mstat(lg,np.zeros(len(cc)),v_,c17); m1,_=mstat(lg,th,v_,c17)
    pr=pd.Series(m1-m0).groupby(cc.hour.round().astype(int).values).mean(); print(nm,'replica TF edit effect by hour',np.round(pr.values[:12],2).tolist())
```