Run 295 failed. The cause is in my diagnostics, not the plan itself:
- **Misaligned targets.** BUN has 2,181 drug bundles but `s1data` has 4,759 rows, so the residual targets were not aligned with the calls (correlation 0.03).
- **Untrustworthy dose response.** The response measured through `tfnll` with edited bundle atoms was tiny and inconsistent across points (|J| = 0.04, common-direction fit 0.24). `tfnll` probably does not re-encode the edited atoms.
- **Result.** The unit-response constraint then forced a decoder with huge factual noise (sd 3.2). The readouts became symmetric noise and the TF NLL jumped from 1.556 to 1.80.
- **What did work.** The glucose-only and MAP-only view directions behave as intended: about 0.98 shift on the target codes and about 0.00 on the others.

Fixes in this run:
1. Measure the dose response through the rollout's own dose-edit path, which already reproduces the tool's readouts. I run `rollout` with readings never hidden, which makes it teacher-forced, under recorded, insulin ×2 and norepinephrine ×2 doses. I capture the states from `bstep` and rebuild every view base exactly.
2. Align `s1data` to BUN by nearest-neighbour matching of the drug-step contexts.
3. Fit decoders that trade residual tracking against response to the dose edit. I scan the weight and pick the one with the lowest noise per unit of response.
4. Evaluate two gains with `evalW` and `tfnll`. If a candidate stays within 0.004 nats, save it provisionally.

How I will read the outcome:
- **Confirms:** the capture-based response is consistent (fit well above 0.24), the noise/response ratio is near 1 or below, and the readouts move toward −14 / +7 / +4.9 / −2.4 at an acceptable NLL.
- **Refutes:** noise/response stays far above 1, which would mean the head cannot see enough dose information.

```python
import torch, numpy as np, time, math
t0=time.time()
Rl=open('/work/wm2_rollout.py').read().split('\n'); print(' / '.join(l.strip()[:70] for l in Rl[1:14] if l.strip())[:600])
def ridge(X,y,lam):
    m=X.mean(0); Xc=X-m; ym=y.mean(); A=Xc.T@Xc; A+=lam*np.trace(A)/A.shape[0]*np.eye(A.shape[0]); return np.linalg.solve(A,Xc.T@(y-ym)),m,ym
try:
    _bs=bstep; ST=[]; CAPR={}
    def bs_cap(*a,**kw):
        r=_bs(*a,**kw); ST.append((r[0].detach().clone(),r[1].detach().clone())); return r
    globals()['bstep']=bs_cap
    try:
        for s in STAYS:
            ST.clear(); rollout(W_cur,BUN[s],[None,(8,2.0),(9,2.0)],npaths=1,seed=0,t_edit=-100.0,t_hide=1e9); CAPR[s]=list(ST)
    finally: globals()['bstep']=_bs
    zb=W_cur['z_head.bias'].detach()
    o0=torch.tensor(X0row[:192],dtype=T)[None]; h0=torch.tensor(X0row[192:960],dtype=T).reshape(1,4,192)
    BS={}; nbad=0
    for s in STAYS:
        B_=BUN[s]; tm=[b[0] for b in B_]; st=CAPR[s]; d={}
        if len(st)!=len(B_)-1: nbad+=1; continue
        for k in range(1,len(B_)):
            o,h=(o0,h0) if k==1 else st[k-2]
            v=(_bv(W_cur,o,h,max((tm[k]-tm[k-1])*3600.0,1.0),zs=torch.zeros(o.shape[0],4,32,dtype=T))[:,0,:]-zb).double().numpy()
            d[k]=np.repeat(v,3,0) if v.shape[0]==1 else v
        BS[s]=d
    DK=[(s,k) for s in BS for k,(t,at) in enumerate(BUN[s]) if k>0 and any(a['act'] for a in at)]
    Cb=np.array([BS[s][k][0] for s,k in DK])
    d1=np.load('/work/s1data.npz'); C1=d1['C'].astype(np.float64); ri=d1['ri'].astype(float); rn=d1['rn'].astype(float); hi=d1['hi'].astype(float); hn=d1['hn'].astype(float)
    sq=(Cb**2).sum(1)[:,None]+(C1**2).sum(1)[None,:]-2*Cb@C1.T; nnI=sq.argmin(1); nd=np.sqrt(np.maximum(sq[np.arange(len(DK)),nnI],0))
    ok=nd<1e-2*np.sqrt((Cb**2).sum(1)); print('badstays',nbad,'drug',len(DK),'matched',int(ok.sum()),'med dist %.2e'%np.median(nd),round(time.time()-t0))
    ii=hi>0; wi,mi_,yi=ridge(C1[ii],ri[ii],1e-2); pi_i=(C1-mi_)@wi+yi
    nn_=hn>0; wn,mn_,yn=ridge(C1[nn_],rn[nn_],1e-2); pi_n=(C1-mn_)@wn+yn
    PRV={(s,k):(((ri[j]-pi_i[j])*(hi[j]>0),ri[j]*(hi[j]>0),(rn[j]-pi_n[j])*(hn[j]>0),rn[j]*(hn[j]>0)) if o else None) for (s,k),j,o in zip(DK,nnI,ok)}
    rows=[]
    for s in BS:
        B_=BUN[s]
        for k in range(1,len(B_)):
            codes={a['code'] for a in B_[k][1] if a['code'] in MEASC and a['kind']==0}
            if not codes: continue
            pv=PRV.get((s,k-1),(0.,0.,0.,0.))
            if pv is None: continue
            b=BS[s][k]; rows.append((b[0],b[1]-b[0],b[2]-b[0])+tuple(pv)+(bool(codes&set(glu)),bool(codes&set(mpc))))
    X=np.array([r[0] for r in rows]); DI=np.array([r[1] for r in rows]); DN=np.array([r[2] for r in rows]); A_=np.array([r[3:7] for r in rows],dtype=float)
    Gm=np.array([r[7] for r in rows]); Mm=np.array([r[8] for r in rows])
    def jfit(D,d):
        sel=d>0; J=(D[sel]*d[sel,None]).sum(0)/(d[sel]**2).sum(); return np.linalg.norm(J),1-((D[sel]-np.outer(d[sel],J))**2).sum()/(D[sel]**2).sum()
    print('rows',len(rows),'Ji |J| %.3f fit %.2f'%jfit(DI[Gm],A_[Gm,1]),'Jn |J| %.3f fit %.2f'%jfit(DN[Mm],A_[Mm,3]))
    def dec(Xs_,t,D,dd,w,lam=1e-3):
        sel=dd>0; D=D[sel]; dd=dd[sel]; m=Xs_.mean(0); Xc=Xs_-m; A=Xc.T@Xc; sc=np.trace(A)/192; r_=len(t)/len(dd)
        a=np.linalg.solve(A+w*r_*(D.T@D)+lam*sc*np.eye(192),Xc.T@t+w*r_*(D.T@dd)); s_=Xc@a
        resp=float((D@a)@dd/(dd@dd)); return a,m,resp,float(s_.std()),float(np.corrcoef(s_,t)[0,1])
    best={}
    for nm,Xs_,t,D,dd in [('ins',X[Gm],A_[Gm,0],DI[Gm],A_[Gm,1]),('ne',X[Mm],A_[Mm,2],DN[Mm],A_[Mm,3])]:
        tab=[]
        for w in [0,0.03,0.1,0.3,1,3,10]:
            a,m,resp,sd,cr=dec(Xs_,t,D,dd,w); tab.append((sd/max(resp,1e-6) if resp>0.15 else 1e9,w,a,m,resp,sd,cr))
        print(nm,' '.join('w%g:r%.2f/sd%.2f/c%.2f'%(x[1],x[4],x[5],x[6]) for x in tab)); best[nm]=min(tab,key=lambda x:x[0])
    Q8=dict(Q7); Q8.update(a_i=best['ins'][2],m_i=best['ins'][3],a_n=best['ne'][2],m_n=best['ne'][3])
    rI,rN=best['ins'][4],best['ne'][4]; print('chosen w',best['ins'][1],best['ne'][1],'resp',round(rI,3),round(rN,3))
    tb=tfnll(W_cur,STAYS)[0]; RES=[]
    for f in [1.0,1.5]:
        bi=-1.9/rI*f; bn=3.0/rN*f; Wt=mkWz(bi,bn,Q=Q8); r=evalW(Wt,STAYS); tf=tfnll(Wt,STAYS)[0]
        RES.append((bi,bn,r,tf)); print('bi %.2f bn %.2f'%(bi,bn),{k:v[0] for k,v in r.items()},'dNLL %.4f'%(tf-tb),round(time.time()-t0))
    cand=[x for x in RES if x[3]-tb<=0.004]
    if cand:
        bi,bn,r,tf=min(cand,key=lambda x:abs(x[2][1][0]-x[2][1][1])); Wt=mkWz(bi,bn,Q=Q8)
        for nme in ['z_prior.0.weight','z_prior.0.bias','z_prior.2.weight','z_prior.2.bias','z_head.weight']: save_edit(nme,Wt[nme].float().contiguous())
        print('SAVED',round(bi,2),round(bn,2),edited())
except Exception as e:
    import traceback; print('ERR',str(e)[:200]); print(traceback.format_exc()[-500:])
with open('/work/notes.md','a') as f:
    f.write("- Run 295 FAILED: s1data rows != BUN drug bundles (mapping wrong); J via tfnll+edited BUN unreliable; decoder noise huge -> tfnll 1.80. Discard.\n- Run 296: J via rollout(t_hide=1e9) bstep capture; s1data matched by NN on ctx; decoders dec(w) scan; Q8, mkWz(bi,bn,Q=Q8). Saved only if dNLL<=.004.\n")
```