My bundles are one short for every stay. The tool has an extra final bundle at hour 25.1667, the stay's end. The data README says admission rows carry the stay-end time in `end_hour`. So the model also gets END atoms for the hospital and/or ICU admission at the end of the stay.

Next I add an END atom (kind 2) at `end_hour` for each admission row that has one. I use the tool's atom count for that last bundle to decide whether both admission rows contribute. Then I recheck bundle times and atom counts for all 100 stays. A full match confirms the bundle construction.

```python
print('last bundle n_atoms:', npz['n_atoms'][row_of[stays[0]]][-3:], ' admission rows:', tu[(tu.stay_id==stays[0])&(tu.event.isin(['hospital_admission','icu_admission']))][['event','hour','end_hour']].values.tolist())
fix = r'''
def build_bundles(ds, adm_end=('hospital_admission','icu_admission')):
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
        elif r.event in ('hospital_admission','icu_admission'):
            a=dict(code=CODE_ID[r.event],kind=0,ekind=3 if r.event=='hospital_admission' else 4,unit=0,rate_unit=0,vb=0,db=0,rb=0,num=(0,0,0,0,0,0),act=False)
            ev.append((t,a))
            if r.event in adm_end and not pd.isna(r.end_hour):
                b=dict(a); b['kind']=2; ev.append((round(r.end_hour*60),b))
    ev.sort(key=lambda x:x[0])
    out=[]
    for t,a in ev:
        if out and out[-1][0]==t: out[-1][1].append(a)
        else: out.append((t,[a]))
    return [(t/60.0,al) for t,al in out]
'''
exec(fix)
s0=open('/work/setup.py').read(); i0=s0.index('def build_bundles'); i1=s0.index("tu=pd.read_csv")
open('/work/setup.py','w').write(s0[:i0]+fix.strip()+'\n'+s0[i1:])
for adm in [('hospital_admission','icu_admission'),('hospital_admission',),('icu_admission',)]:
    BUN={s:build_bundles(tu[tu.stay_id==s],adm) for s in stays}
    okt=all(len(BUN[s])==len(row_of[s]) and np.abs(np.array([t for t,_ in BUN[s]])-hr_arr[row_of[s]]).max()<1e-3 for s in stays)
    okn=okt and all((np.array([len(al) for _,al in BUN[s]])==npz['n_atoms'][row_of[s]]).all() for s in stays)
    print(adm,'times ok',okt,'n_atoms ok',okn)
```