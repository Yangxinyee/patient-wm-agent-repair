Run 199 explains vI. Its edit response equals the lag-1 dose change only on average (row slope 0.65). Its factual noise is about 15 dose units per unit of response, far beyond what the factual budget allows (an added prediction error of roughly 1.7 rms in glucose units). So vI, and probably vN (3.3 noise against a dose spread of about 0.1), cannot be used as stored.

Next I will evaluate the earlier ridge decoders in DEC on the same footing. For each I measure the factual noise on cell-residualised head inputs, and the mean response to the own-drug x2 and x1/2 edits and to the other drug's edits. If the ridge decoders respond at about 0.8-0.9 with noise near 0.5, a low-gain head edit is still feasible.

```python
def resp(w_,key,codes):
    e_=EDR[key]; m_=np.isin(cf[e_['gi']],codes)&e_['reached'].astype(bool)
    y_=e_['dX'][m_]@w_; d1=e_['dd'][m_][:,0]
    return round(y_.mean(),3), round(d1.mean(),3)
for nm,codes,rw in [('glucose_finger',[0,1],ins_rows),('map_arterial',[2,3],map_rows)]:
    dob=DEC[nm]; print(nm,type(dob).__name__, (len(dob) if isinstance(dob,(list,tuple,dict)) else ''))
    objs = dob if isinstance(dob,(list,tuple)) else ([dob[k] for k in dob] if isinstance(dob,dict) else [dob])
    for j,o in enumerate(objs[:3]):
        try:
            w0=get_w(o); w0=w0[0] if isinstance(w0,tuple) else w0; w0=np.asarray(w0).ravel()
            if w0.size!=192: print(' lag',j,'size',w0.size); continue
            print(' lag',j+1,'noise',round(np.std(Rf[rw]@w0),3),[ (k[0][:3]+str(k[1]),resp(w0,k,codes)) for k in keysI+keysN])
        except Exception as ex: print(' lag',j,'ERR',repr(ex)[:150])
```