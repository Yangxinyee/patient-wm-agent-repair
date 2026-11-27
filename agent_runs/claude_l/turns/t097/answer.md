The vocab notes confirm my encoder conventions. The z-score is (x − median)/IQR clipped to ±10, and a fine bin is 1 plus the number of edges strictly below x, which matches `searchsorted(side='left')`. The embedding table row norms match the atom fields I use. So the per-atom inputs look right, and the z mismatch must come from somewhere else.

Before testing further I must rebuild what the restart wiped. I will write a reusable /work/setup.py that:
- loads the data and states;
- builds each stay's bundles (measurements, visits, infusion start and END atoms with their rate/dose attributes, grouped by minute);
- checks that my bundle times match the tool's one-row-per-bundle hours for all 100 stays.

An exact match of bundle counts and times confirms the bundle construction, which every later step depends on.

```python
setup = r'''
import json, math, numpy as np, pandas as pd, torch
import torch.nn.functional as F
exec(open('/work/wm.py').read())
V=json.load(open('/model/vocab.json'))
NUMV={(d['code'],d['attribute']):d for d in V['numeric']}
CODE_ID={c['name']:c['id'] for c in V['codes']}
UM={'mg/dL':2,'mmHg':5,'units':3,'mg':4}; RUM={'units/hour':2,'mcg/kg/min':3}
def zsc(code,attr,x):
    d=NUMV[(code,attr)]; return float(np.clip((x-d['median'])/d['iqr'],-10,10))
def fbin(code,attr,x):
    return 1+int(np.searchsorted(np.array(NUMV[(code,attr)]['fine_bin_edges']),x,side='left'))
def build_bundles(ds):
    ev=[]
    for r in ds.itertuples():
        t=round(r.hour*60)
        if r.event in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            ev.append((t,dict(code=CODE_ID[r.event],kind=0,ekind=1,unit=UM['mg/dL' if 'glucose' in r.event else 'mmHg'],rate_unit=0,
                vb=fbin(r.event,'value',r.value),db=0,rb=0,num=(zsc(r.event,'value',r.value),1,0,0,0,0),act=False)))
        elif r.event in ('insulin','norepinephrine'):
            a=dict(code=CODE_ID[r.event],kind=0,ekind=2,unit=UM[r.dose_unit],rate_unit=RUM[r.rate_unit],vb=0,
                   db=fbin(r.event,'dose',r.dose),rb=fbin(r.event,'rate',r.rate),num=(0,0,zsc(r.event,'dose',r.dose),1,zsc(r.event,'rate',r.rate),1),act=True)
            ev.append((t,a)); b=dict(a); b['kind']=2; ev.append((round(r.end_hour*60),b))
        elif r.event=='hospital_admission':
            ev.append((t,dict(code=CODE_ID[r.event],kind=0,ekind=3,unit=0,rate_unit=0,vb=0,db=0,rb=0,num=(0,0,0,0,0,0),act=False)))
        elif r.event=='icu_admission':
            ev.append((t,dict(code=CODE_ID[r.event],kind=0,ekind=4,unit=0,rate_unit=0,vb=0,db=0,rb=0,num=(0,0,0,0,0,0),act=False)))
    ev.sort(key=lambda x:x[0])
    out=[]
    for t,a in ev:
        if out and out[-1][0]==t: out[-1][1].append(a)
        else: out.append((t,[a]))
    return [(t/60.0,al) for t,al in out]
tu=pd.read_csv('/data/tuning.csv')
npz=np.load('/work/fwd_2/states_original.npz'); X=npz['x'].astype(np.float64); sid_arr=npz['stay_id']; hr_arr=npz['hour']
stays=list(dict.fromkeys(sid_arr.tolist()))
row_of={s:np.where(sid_arr==s)[0] for s in stays}
W0=getW(load_original, NAMES)
'''
open('/work/setup.py','w').write(setup); exec(setup)
BUN={s:build_bundles(tu[tu.stay_id==s]) for s in stays}
bad=[]
for s in stays:
    tb=np.array([t for t,_ in BUN[s]]); th=hr_arr[row_of[s]]
    if len(tb)!=len(th) or np.abs(tb-th).max()>1e-3: bad.append((s,len(tb),len(th)))
print('stays',len(stays),'bad',len(bad),bad[:5])
s=stays[0]; print([ (round(t,4),[(a['code'],a['kind']) for a in al]) for t,al in BUN[s][-4:]], hr_arr[row_of[s]][-4:])
print('n_atoms match', all((np.array([len(al) for _,al in BUN[s]])==npz['n_atoms'][row_of[s]]).all() for s in stays))
```