Run 262 failed at its second line. `inspect.getsource` cannot see functions that were defined with `exec`, so nothing else ran. I'm rerunning the same analysis, but reading the `bview` source from /work/wm2.py instead.

The plan is unchanged:
- **Collect:** reading-time views, model errors, and the view shifts caused by x2 and x1/2 dose edits at each drug bundle.
- **Fit:** for each drug, a decoder of the last dose's residual from the reading-time view. It is constrained to respond 1:1 to real dose edits of that drug and not at all to edits of the other drug.

What would support the head-side adapter:
- a unit response to edits with a small sd across bundles;
- a model-error slope on the decoded residual near the earlier −1.5 for insulin.

What would refute it: a decoded residual that tracks the data residual poorly once the constraint is imposed.

```python
import numpy as np, torch, time
src_=open('/work/wm2.py').read(); i_=src_.find('def bview'); print(src_[i_:i_+900])
t0=time.time(); W0c[0]=Wb
jof={(d['s'],d['k']):j for j,d in enumerate(DD)}
def tf_collect(W,B_,s):
    times=[b[0] for b in B_]
    out=torch.tensor(X0row[:192],dtype=T)[None]; h=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    ea,ua=split_fix(B_[0][1]); Ms=[set_enc(W,ea,'encode')]; ts=[times[0]]; reg={}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']]=(times[0],a['num'][0],a['vb'])
    READ=[]; PERT=[]; ld=None
    for k in range(1,len(B_)):
        t=times[k]; dt=(t-times[k-1])*3600.0; atoms=B_[k][1]; ea,ua=split_fix(atoms)
        meas=[a for a in ea if a['code'] in MEASC and a['kind']==0]
        if meas:
            views=bview(W,out,h,dt,zs=torch.zeros(1,1,32,dtype=T)); errs={}
            for a in meas:
                c=a['code']; p=bvalue_probs(W,views,c,*regp(reg,c,t,1))[0]; md,iq=MED[c]
                errs[c]=a['num'][0]*iq+md-float((p[1:]*torch.tensor(CENd[c],dtype=T)).sum())
            READ.append(dict(s=s,k=k,x=views[0,0].numpy().copy(),err=errs,ld=(ld if ld is not None and t-ld[1]<1.0 else None)))
            for a in meas: reg[a['code']]=(t,a['num'][0],a['vb'])
        e=set_enc(W,ea,'encode'); u=set_enc(W,ua,'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt=torch.stack(Ms)[None]; ages=torch.tensor([(t-tj)*3600.0 for tj in ts],dtype=T)
        if ua:
            ld=(k,t)
            if k+1<len(B_) and any(a['code'] in MEASC and a['kind']==0 for a in B_[k+1][1]):
                codes=set(a['code'] for a in ua if a['act'] and a['kind']==0); rates,_=drug_rates(atoms)
                conds=[(c,f) for c in (8,9) if c in codes for f in (2.0,0.5)]
                Up=torch.stack([u]+[u_for(atoms,t,cd,-1.0) for cd in conds],0); Bt=Up.shape[0]
                o2,h2,_=bstep(W,out.repeat(Bt,1),h.repeat(Bt,1,1),dt,e[None].repeat(Bt,1),Up,Mt.repeat(Bt,1,1),ages)
                tn=times[k+1]; vw=bview(W,o2,h2,(tn-t)*3600.0,zs=torch.zeros(Bt,1,32,dtype=T))
                mm=torch.stack([(bvalue_probs(W,vw,c,*regp(reg,c,tn,Bt))[:,1:]*torch.tensor(CENd[c],dtype=T)).sum(1) for c in ALLC],1)
                PERT.append(dict(s=s,k=k,conds=conds,dd=[rates[c]*(f-1) for c,f in conds],dx=(vw[1:,0]-vw[0:1,0]).numpy(),dm=(mm[1:]-mm[0:1]).numpy()))
        out,h,_=bstep(W,out,h,dt,e[None],u[None],Mt,ages)
    return READ,PERT
READ=[]; PERT=[]
for s,B_ in TRB.items():
    a_,b_=tf_collect(Wb,B_,s); READ+=a_; PERT+=b_
print('reads',len(READ),'pert',len(PERT),'t',round(time.time()-t0))
def rres(rd,drug):
    if rd['ld'] is None: return 0.0
    j=jof.get((rd['s'],rd['ld'][0]))
    if j is None: return 0.0
    return (ri[j]-pi_i[j]) if (drug==8 and hi[j]) else ((rn[j]-pi_n[j]) if (drug==9 and hn[j]) else 0.0)
GC=[CODE_ID['glucose_finger'],CODE_ID['glucose_lab']]; MC=[CODE_ID['map_arterial'],CODE_ID['map_cuff']]
DECS={}
for drug,cset in [(8,GC),(9,MC)]:
    rows=[r for r in READ if any(c in r['err'] for c in cset)]
    X=np.array([r['x'] for r in rows],dtype=np.float64); y=np.array([rres(r,drug) for r in rows])
    E_=np.array([np.mean([r['err'][c] for c in cset if c in r['err']]) for r in rows])
    Cx=[];Cy=[]
    for p in PERT:
        for (c,f),dd,dx in zip(p['conds'],p['dd'],p['dx']):
            if abs(dd)>1e-6: Cx.append(dx); Cy.append(dd if c==drug else 0.0)
    Cx=np.array(Cx,dtype=np.float64); Cy=np.array(Cy); own=np.abs(Cy)>0
    mx=X.mean(0); Xc=X-mx
    for wc,lam in [(1.0,1e-2),(10.0,1e-2),(100.0,1e-2)]:
        sw=np.sqrt(wc*len(X)/len(Cx)); A_=np.vstack([Xc,sw*Cx]); b_=np.r_[y-y.mean(),sw*Cy]
        a=np.linalg.solve(A_.T@A_+lam*len(X)*np.eye(192),A_.T@b_); rh=Xc@a+y.mean(); resp=Cx@a
        cb=np.polyfit(rh,E_,1)[0]; cr=np.polyfit(y,E_,1)[0]
        print('drug',drug,'wc',wc,'R2',round(1-np.var(y-rh)/np.var(y),3),'resp/dd own',round(np.mean(resp[own]/Cy[own]),2),'sd',round(np.std(resp[own]/Cy[own]),2),'cross sd',round(np.std(resp[~own]),3) if (~own).any() else None,'err~rhat',round(cb,2),'err~r',round(cr,2))
        DECS[(drug,wc)]=(a,mx,y.mean())
```