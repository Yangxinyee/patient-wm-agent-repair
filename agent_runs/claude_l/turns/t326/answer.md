The replica's `bview` applies `z_prior` (a 192→192→64 MLP) to the base context and uses `z = mu + exp(logsig)*eps`. So a `z_prior`/`z_head` edit will be reproduced exactly by `evalW` and `tfnll`. `tfnll` calls the global `bview` once per reading bundle, with dt in seconds, so I can capture the view bases. `s1data` holds, per batch1 drug bundle, the context C, the raw doses (`ri`, `rn`) and the drug flags (`hi`, `hn`).

This run builds the whole adapter:
1. **Capture.** Record the view base at every reading prediction under three conditions: recorded doses, insulin ×2, and norepinephrine ×2. The differences give the base's response J to the dose given in the previous bundle.
2. **Residual target.** Compute each dose's residual from a ridge policy fitted on the drug-step context.
3. **Decoder.** Fit a ridge decoder of the previous-bundle residual. Constrain it to have unit response along J, so a dose change moves it one-for-one while its factual noise stays small.
4. **Output directions.** From head Jacobians, find a view direction that shifts only glucose predictions and one that shifts only MAP predictions.
5. **Wiring.** Make `z_prior` a deterministic linear pass-through, using GELU in its linear regime and log-sigma −30. Point `z_head` columns 0 and 1 along the two directions.
6. **Calibration.** Run two gain settings through `evalW` and `tfnll` so the next run can interpolate.

How I will read the outcome:
- **Supports the plan:** the readouts grow roughly in proportion to the gain while the TF NLL stays within about 0.002 of baseline.
- **Refutes it:** the readouts barely move, or the NLL degrades sharply.

```python
import torch, numpy as np, time, math, copy
t0=time.time(); STAYS=list(BUN.keys())
glu=[c for c in sorted(MEASC) if max(CENd[c])>200]; mpc=[c for c in sorted(MEASC) if c not in glu]
print('glu',glu,'map',mpc,'gelu',G_ is torch.nn.functional.gelu)
_bv=bview; _bp=bvalue_probs; zhb=W_cur['z_head.bias'].detach().double().numpy()
def tt(x): return torch.as_tensor(x).reshape(-1)[:1].clone()
def q7_cap(Wx, edit=None, jac=False):
    R={}
    for s in STAYS:
        calls=[]
        def bv(W_,out,h,dt,zs=None,gen=None,nz=4):
            r=_bv(W_,out,h,dt,zs=zs,gen=gen,nz=nz); calls.append([r[0,0,:].detach().double().numpy()-zhb,[]]); return r
        def bp(W_,views,code,has,age_s,zlast,vblast):
            p=_bp(W_,views,code,has,age_s,zlast,vblast); c=int(code)
            pm=float((p[0,1:]*torch.tensor(CENd[c],dtype=p.dtype)).sum())
            calls[-1][1].append((c,pm,(views[:1].clone(),tt(has),tt(age_s),tt(zlast),tt(vblast)) if jac else None)); return p
        orig=BUN[s]
        if edit is not None:
            ce,f=edit; BUN[s]=[(t,[edit_atom(copy.deepcopy(a),f) if (a['code']==ce and a['act']) else a for a in at]) for t,at in orig]
        globals()['bview']=bv; globals()['bvalue_probs']=bp
        try: tfnll(Wx,[s])
        finally: BUN[s]=orig; globals()['bview']=_bv; globals()['bvalue_probs']=_bp
        R[s]=calls
    return R
Q7={}
try:
    CO=q7_cap(W_cur,jac=True); CI=q7_cap(W_cur,edit=(8,2.0)); CN=q7_cap(W_cur,edit=(9,2.0))
    d1=np.load('/work/s1data.npz'); C1=d1['C'].astype(np.float64); ri=d1['ri']; rn=d1['rn']; hi=d1['hi']; hn=d1['hn']
    drows={}; n=0
    for s in STAYS:
        for k,(t,at) in enumerate(BUN[s]):
            if any(a['act'] for a in at): drows[(s,k)]=n; n+=1
    def ridge(X,y,lam):
        m=X.mean(0); Xc=X-m; ym=y.mean(); A=Xc.T@Xc; A+=lam*np.trace(A)/A.shape[0]*np.eye(A.shape[0]); return np.linalg.solve(A,Xc.T@(y-ym)),m,ym
    ii=hi>0; wi,mi,yi=ridge(C1[ii],ri[ii],1e-2); res_i=np.zeros(len(ri)); res_i[ii]=ri[ii]-((C1[ii]-mi)@wi+yi)
    nn=hn>0; wn,mn,yn=ridge(C1[nn],rn[nn],1e-2); res_n=np.zeros(len(rn)); res_n[nn]=rn[nn]-((C1[nn]-mn)@wn+yn)
    print('drug bundles',n,'s1',len(ri),'pihat R2 ins %.3f ne %.3f'%(1-res_i[ii].var()/ri[ii].var(),1-res_n[nn].var()/rn[nn].var()))
    R=[]; bad=0
    for s in STAYS:
        rk=[k for k,(t,at) in enumerate(BUN[s]) if any(a['code'] in MEASC and a['kind']==0 for a in at)]
        cl=CO[s]
        ks=rk if len(cl)==len(rk) else ([k for k in rk if k>0] if len(cl)==len([k for k in rk if k>0]) else None)
        if ks is None: bad+=1; continue
        for j,k in enumerate(ks):
            dr=drows.get((s,k-1)); codes=[c for c,_,_ in cl[j][1]]
            ti,di,tn,dn=(res_i[dr],ri[dr]*hi[dr],res_n[dr],rn[dr]*hn[dr]) if dr is not None else (0,0,0,0)
            R.append((cl[j][0],CI[s][j][0]-cl[j][0],CN[s][j][0]-cl[j][0],ti,di,tn,dn,any(c in glu for c in codes),any(c in mpc for c in codes)))
    X=np.array([r[0] for r in R]); DI=np.array([r[1] for r in R]); DN=np.array([r[2] for r in R]); A_=np.array([r[3:7] for r in R],dtype=float)
    G=np.array([r[7] for r in R]); M=np.array([r[8] for r in R]); print('calls',len(R),'bad',bad,'glu pts',G.sum(),'map pts',M.sum())
    def jfit(D,d):
        sel=d>0; J=(D[sel]*d[sel,None]).sum(0)/(d[sel]**2).sum(); fr=1-((D[sel]-np.outer(d[sel],J))**2).sum()/(D[sel]**2).sum(); return J,fr
    Ji,fri=jfit(DI,A_[:,1]); Jn,frn=jfit(DN,A_[:,3]); print('|Ji| %.3f fit %.2f |Jn| %.3f fit %.2f'%(np.linalg.norm(Ji),fri,np.linalg.norm(Jn),frn))
    def cdec(Xs_,t,J,lam=1e-2):
        m=Xs_.mean(0); Xc=Xs_-m; A=Xc.T@Xc; A+=lam*np.trace(A)/A.shape[0]*np.eye(A.shape[0])
        b0=np.linalg.solve(A,Xc.T@t); bJ=np.linalg.solve(A,J); mu=(1-J@b0)/(J@bJ); a=b0+mu*bJ; s=Xc@a
        return a,m,(round(float(J@b0),3),round(float(np.corrcoef(s,t)[0,1]),3),round(float(s.std()),3),round(float(t.std()),3))
    a_i,m_i,inf_i=cdec(X[G],A_[G,0],Ji); a_n,m_n,inf_n=cdec(X[M],A_[M,2],Jn); print('dec ins (J.b0,corr,sd_s,sd_t)',inf_i,'ne',inf_n)
    JR=[];JB=[]; cnt={c:0 for c in MEASC}
    for s in STAYS:
        for j,call in enumerate(CO[s]):
            if j%3: continue
            for (c,pm,args) in call[1]:
                if args is None or cnt[c]>=120: continue
                v0,hs,ag,zl,vb=args; P=v0.repeat(193,1,1).clone(); P[1:]+=1e-2*torch.eye(192,dtype=v0.dtype)[:,None,:]
                p=_bp(W_cur,P,c,hs.repeat(193),ag.repeat(193),zl.repeat(193),vb.repeat(193))
                pmv=(p[:,1:]*torch.tensor(CENd[c],dtype=p.dtype)).sum(1).double().numpy(); JR.append((pmv[1:]-pmv[0])/1e-2); JB.append(c); cnt[c]+=1
    JR=np.array(JR); JB=np.array(JB)
    def vdir(tc):
        b=np.isin(JB,tc).astype(float); A=JR.T@JR; A+=1e-3*np.trace(A)/192*np.eye(192); v=np.linalg.solve(A,JR.T@b); sh=JR@v
        return v,{int(c):(round(float(sh[JB==c].mean()),2),round(float(sh[JB==c].std()),2)) for c in sorted(set(JB))}
    vg,shg=vdir(glu); vm,shm=vdir(mpc); print('vglu',shg,'|v| %.2f'%np.linalg.norm(vg)); print('vmap',shm,'|v| %.2f'%np.linalg.norm(vm))
    Q7.update(a_i=a_i,m_i=m_i,a_n=a_n,m_n=m_n,vg=vg,vm=vm,Ji=Ji,Jn=Jn)
except Exception as e:
    import traceback; print('ERR',str(e)[:200]); print(traceback.format_exc()[-400:])
dtp=W_cur['z_prior.0.weight'].dtype
def mkWz(bi,bn,K=12.0,Q=Q7):
    Wt=dict(W_cur); W0w=torch.zeros(192,192,dtype=dtp); b0=torch.zeros(192,dtype=dtp)
    W0w[0]=torch.tensor(Q['a_i'],dtype=dtp); b0[0]=K-float(Q['a_i']@Q['m_i']); W0w[1]=torch.tensor(Q['a_n'],dtype=dtp); b0[1]=K-float(Q['a_n']@Q['m_n'])
    W2w=torch.zeros(64,192,dtype=dtp); b2=torch.zeros(64,dtype=dtp); W2w[0,0]=1; W2w[1,1]=1; b2[0]=-K; b2[1]=-K; b2[32:]=-30
    ZH=torch.zeros(192,32,dtype=dtp); ZH[:,0]=torch.tensor(bi*Q['vg'],dtype=dtp); ZH[:,1]=torch.tensor(bn*Q['vm'],dtype=dtp)
    Wt.update({'z_prior.0.weight':W0w,'z_prior.0.bias':b0,'z_prior.2.weight':W2w,'z_prior.2.bias':b2,'z_head.weight':ZH}); return Wt
try:
    tb=tfnll(W_cur,STAYS); print('tf base',tb, round(time.time()-t0))
    for bi,bn in [(-1.5,3.0),(-3.0,6.0)]:
        Wt=mkWz(bi,bn); r=evalW(Wt,STAYS); tf=tfnll(Wt,STAYS); print(bi,bn,r,tf,round(time.time()-t0))
except Exception as e:
    import traceback; print('ERR2',str(e)[:200]); print(traceback.format_exc()[-400:])
with open('/work/notes.md','a') as f:
    f.write("- Run 295: built z_prior/z_head adapter (mkWz(bi,bn), Q7 dict: decoders a_i/a_n constrained a.J=1 on post-drug view base, vg/vm glucose-only/MAP-only view dirs). See run output for evalW/tfnll at (-1.5,3),(-3,6).\n")
```