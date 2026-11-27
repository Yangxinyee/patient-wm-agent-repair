Age is in seconds, so the last reading's hour is minute/60 minus age/3600. The dose_edit CSV has one row per stay, hour and edit, but only for hours with the relevant reading and with a `reached` flag. The readout is therefore the model's predicted-mean shift at recorded GF (or MA) bundles after the first edited infusion.

Next I build the exact counterfactual teacher-forced data for the 100 tuning stays:
- Rebind mod_events so it also adds the true shift S_t (the kernel applied to the dose change) to reading values. The last-value features in the reading table get the same shift.
- Run the replica from the edit start.
- Variants per drug: multipliers m ∈ {0, 0.5, 1.5, 2, 3} from hour 12, and {0, 0.5, 2, 3} from hours 6 and 16.
- Per reading, store the factual and counterfactual predicted means, the true shift, the head input `with_code`, and the per-draw logits.

As a check, the rebound mod_events must reproduce the old dose-only states exactly when no shift is set. The teacher-forced response ratio (mean change in the model's prediction ÷ true shift) should come out well below 1, consistent with the earlier finding that the model under-uses the dose.

```python
import time
GLU=('glucose_finger','glucose_lab'); MAPC=('map_arterial','map_cuff')
if 'mod_events_orig' not in globals(): mod_events_orig=mod_events
SHIFT={}
def mod_events_new(evs,edits):
    o=[]
    for (ev,kind,row) in evs:
        f=edits.get(ev,1.0)
        if f!=1.0: row=row._replace(rate=row.rate*f,dose=row.dose*f)
        if SHIFT and kind=='world':
            sh=SHIFT.get((ev,int(round(row.hour))),0.0)
            if sh!=0.0: row=row._replace(value=row.value+sh)
        o.append((ev,kind,row))
    return o
s=2020000000
O1,H1,_=run_cf(s,{'insulin':2.0},720,Xs); globals()['mod_events']=mod_events_new; O2,H2,_=run_cf(s,{'insulin':2.0},720,Xs)
print('override diff', float((O1-O2).abs().max()))
def shifts(s,drug,m,h0):
    K=KI if drug=='insulin' else KN; dd=drug_hist(s,drug)
    return {t:sum(K[t-j-1]*(m-1)*dd.get(j,0.0) for j in range(h0,t) if t-j-1<len(K)) for t in range(h0+1,26)}
VAR=[(d,12,m) for d in ('insulin','norepinephrine') for m in (0,0.5,1.5,2,3)]+[(d,h,m) for d in ('insulin','norepinephrine') for h in (6,16) for m in (0,0.5,2,3)]
def cf_stay(s):
    rt=RT[RT.stay_id==s].sort_values('b').reset_index(drop=True)
    eps=torch.tensor(np.random.default_rng(int(s)%1000003).standard_normal((len(rt),4,32)))
    hr=np.round(rt.minute.values/60).astype(int); lh=np.where(rt.has_last.values>0,np.round((rt.minute.values*60-rt.age.values)/3600),-99).astype(int)
    ev=rt.event.values; res=[]; base={}
    def pack(tag,drug,h0,m,rc,O,H,shn,keep):
        q,wc,P,mn=heads_on_states(O,H,s,rc,eps); f,lb,cid=last_feats(rc); lg=logits_draws(wc,lb,cid)
        mn=np.asarray(mn,dtype=float)
        md=pd.DataFrame({'stay':s,'h0':h0,'drug':drug,'m':m,'event':ev,'hour':hr,'S':shn,'mean_cf':mn,'obs':rc.observed.values,'b':rc.b.values})[keep]
        return mn,md,wc.mean(1).numpy()[keep],lg.numpy()[keep]
    for (drug,h0,m) in VAR:
        if h0 not in base:
            SHIFT.clear(); O,H,_=run_cf(s,{},h0*60,Xs)
            mn,md,w_,l_=pack('F','none',h0,1.0,rt,O,H,np.zeros(len(rt)),np.ones(len(rt),bool)); base[h0]=mn
            if h0==12: md['mean_f']=mn; res.append((md,w_,l_))
        dd=drug_hist(s,drug)
        if not any(v>0 for j,v in dd.items() if j>=h0): continue
        cs=GLU if drug=='insulin' else MAPC; St=shifts(s,drug,m,h0)
        SHIFT.clear(); SHIFT.update({(c,t):v for t,v in St.items() for c in cs}); O,H,_=run_cf(s,{drug:m},h0*60,Xs); SHIFT.clear()
        inc=np.isin(ev,cs); shn=np.array([St.get(h,0.0) for h in hr])*inc; shl=np.array([St.get(h,0.0) for h in lh])*inc
        rc=rt.copy(); rc['observed']=rc.observed.values+shn; rc['last_val']=rc.last_val.values+shl
        mn,md,w_,l_=pack('C',drug,h0,m,rc,O,H,shn,hr>h0); md['mean_f']=base[h0][hr>h0]; res.append((md,w_,l_))
    return res
CFD=[]; t0=time.time(); done=[]
for s in sorted(BUN.keys()):
    if time.time()-t0>700: break
    CFD+=cf_stay(s); done.append(s)
print('stays',len(done),'chunks',len(CFD),'sec %.0f'%(time.time()-t0))
M=pd.concat([c[0] for c in CFD],ignore_index=True)
g=M[(M.drug=='insulin')&(M.h0==12)&(M.event=='glucose_finger')]
for m in (0,0.5,2,3):
    x=g[g.m==m]; print('ins m',m,'n',len(x),'S %.2f model %.2f'%(x.S.mean(),(x.mean_cf-x.mean_f).mean()))
x=M[(M.drug=='norepinephrine')&(M.h0==12)&(M.m==2)&(M.event=='map_arterial')]; print('ne m2 n',len(x),'S %.2f model %.2f'%(x.S.mean(),(x.mean_cf-x.mean_f).mean()))
```