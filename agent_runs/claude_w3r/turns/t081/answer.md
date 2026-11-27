The module was written and imported, and the predefined tools (load, NAMES) survived the restart. The run only failed because os was no longer imported. I will now import the basics and reload in a memory-light way:
- the selS stay list;
- the saved states;
- the factual predictions;
- the training records for those 100 stays only.

I will then rebuild the probe dataset with the validated head from impl2, and check that my head's predicted means still match the model's at about 1.9 mg/dL. That confirms the module reproduces the earlier result before I use it further.

```python
import os, json, math, numpy as np, pandas as pd, torch
import impl2; from impl2 import *
selS = json.load(open('/work/selS.json')); print(len(selS))
S_ = np.load('/work/fwd_2/states_original.npz', allow_pickle=True); SX=S_['x']; SST=S_['stay_id']; SHR=S_['hour']
fac = pd.read_csv('/work/fwd_3/factual_original.csv')
recs = read_stays('/data/train.csv', selS)
HD = Head({n: load_original(n) for n in NAMES})
key_k={s:(np.where(SST==s)[0], SHR[SST==s]) for s in np.unique(SST)}
def head_inputs(stay, hour, event):
    idx_,hrs_=key_k[stay]; k=int(np.where(np.isclose(hrs_,hour))[0][0]); i0=idx_[k-1]
    dt=(hrs_[k]-hrs_[k-1])*3600
    r_=recs[(recs.stay_id==stay)&(recs.event==event)&(recs.hour<hour-1e-9)]
    lv,lt=(None,None) if len(r_)==0 else (float(r_.value.iloc[-1]), float(r_.hour.iloc[-1]))
    return i0, dt, lv, (None if lt is None else (hour-lt)*3600)
fsa = fac[fac.event=='glucose_finger'].reset_index(drop=True)
rows_h=[]; FH=[]; PMm=[]
for r in fsa.itertuples():
    i0,dt,lv,age=head_inputs(r.stay_id,r.hour,'glucose_finger')
    c=HD.ctx(torch.tensor(SX[i0,:192]), torch.tensor(SX[i0,192:960]).reshape(4,192), dt); zm,_=HD.zprior(c)
    wc,trow=HD.wc(c, zm, 'glucose_finger', lv, age); hid=HD.hidden(wc); FH.append(hid.numpy())
    p=torch.softmax(HD.logits(hid,trow),-1); PMm.append(float((p[1:]*CEN['glucose_finger']).sum()))
    st_=recs[recs.stay_id==r.stay_id]
    ip=st_[(st_.event=='insulin')&(np.isclose(st_.hour, r.hour-1+5/60))]; ip2=st_[(st_.event=='insulin')&(np.isclose(st_.hour, r.hour-2+5/60))]
    rows_h.append((r.stay_id, r.hour, float(ip.rate.iloc[0]) if len(ip) else 0.0, float(ip2.rate.iloc[0]) if len(ip2) else 0.0, lv, r.observed, r.predicted_mean, i0))
DH=pd.DataFrame(rows_h, columns=['stay_id','hour','rs','rs2','gl','obs','pm','i0']); FH=np.array(FH); DH['mypm']=PMm
DH['m']=[m_of(g) for g in DH.gl]; DH['dev']=DH.rs-DH.m
print(DH.shape, 'median |mypm - pm| (z=prior mean)', round(float(np.median(np.abs(DH.mypm-DH.pm))),2))
```