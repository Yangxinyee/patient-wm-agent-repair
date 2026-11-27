Run 296 showed the dose-response capture through the rollout path works, but two things went wrong:
- **Capture works.** Response fits are 0.61 for insulin and 0.37 for norepinephrine, against 0.24 before.
- **Bad training data.** `s1data` still did not match these stays (only 15 of 2,181), so every post-drug point was dropped. The decoder was trained almost only on points without a preceding drug bundle.
- **Result.** On real post-drug views it responded to the full dose, policy part included. That biased the factual predictions, and the TF NLL rose by +0.23.
- **The approach can move the readouts.** At those gains insulin ×2 went to −10.4 and −12.5, and norepinephrine ×2 to +4.0 and +4.8.
- **Norepinephrine looks log-like.** Its ×1/2 response is about as large as its ×2 response. The gain choice will have to balance the two.

`edit_atom` showed where the doses are: rate = `num[4]·iqr + median` from `NUMV`. So I can now take every dose directly from the bundles, with no `s1data`.

This run:
1. Build exact insulin and norepinephrine rates for all 2,181 drug bundles.
2. Fit a policy on the drug-step view, cross-fitted by stay, to get honest dose residuals.
3. Train the decoders on 70 stays and choose the hyperparameters on the 30 held-out stays.
4. Measure the held-out NLL cost at half and full gain for each channel, and the readouts at full gain.
5. Choose the gains that bring the readouts closest to the truth while keeping the held-out cost at or below 0.0035 nats.
6. Refit on all 100 stays and save.

How I will read the outcome:
- **Confirms:** a held-out cost small enough to allow sizeable gains.
- **Refutes:** even small gains are expensive, so the head cannot separate dose deviations from policy-driven doses cheaply.

```python
import torch, numpy as np, time
t0=time.time()
try:
    def rate_of(a):
        p=NUMV[(DRUGN[a['code']],'rate')]; return a['num'][4]*p['iqr']+p['median']
    SL=list(BS); DRUG={}
    for s in SL:
        for k,(t,at) in enumerate(BUN[s]):
            ua=[a for a in at if a['act'] and a['kind']==0 and a['code'] in (8,9)]
            if k==0 or not ua: continue
            r8=[rate_of(a) for a in ua if a['code']==8]; r9=[rate_of(a) for a in ua if a['code']==9]
            DRUG[(s,k)]=(sum(r8),float(len(r8)>0),sum(r9),float(len(r9)>0))
    keys=list(DRUG); Xd=np.array([BS[s][k][0] for s,k in keys]); Vd=np.array([DRUG[x] for x in keys]); fold={s:i%2 for i,s in enumerate(SL)}; Fd=np.array([fold[s] for s,k in keys])
    RS={}
    for col,hc,nm in [(0,1,'i'),(2,3,'n')]:
        r=np.zeros(len(keys))
        for f_ in (0,1):
            trn=(Fd!=f_)&(Vd[:,hc]>0); tst=(Fd==f_)&(Vd[:,hc]>0); w_,m_,y_=ridge(Xd[trn],Vd[trn,col],3e-2); r[tst]=Vd[tst,col]-((Xd[tst]-m_)@w_+y_)
        RS[nm]=r; sel=Vd[:,hc]>0; print(nm,'n',int(sel.sum()),'mean rate %.3f pihat CV R2 %.3f resid sd %.3f'%(Vd[sel,col].mean(),1-r[sel].var()/Vd[sel,col].var(),r[sel].std()))
    PV={x:(RS['i'][j],Vd[j,0],RS['n'][j],Vd[j,2]) for j,x in enumerate(keys)}
    rows=[]
    for s in SL:
        B_=BUN[s]
        for k in range(1,len(B_)):
            codes={a['code'] for a in B_[k][1] if a['code'] in MEASC and a['kind']==0}
            if not codes: continue
            pv=PV.get((s,k-1),(0.,0.,0.,0.)); b=BS[s][k]
            rows.append((s,b[0],b[1]-b[0],b[2]-b[0])+tuple(pv)+(bool(codes&set(glu)),bool(codes&set(mpc))))
    Srow=np.array([r[0] for r in rows]); X=np.array([r[1] for r in rows]); DI=np.array([r[2] for r in rows]); DN=np.array([r[3] for r in rows])
    A_=np.array([r[4:8] for r in rows],dtype=float); Gm=np.array([r[8] for r in rows]); Mm=np.array([r[9] for r in rows]); TRm=np.isin(Srow,SL[:70]); HOm=~TRm
    def dec2(Xs_,t,D,dd,w,lam):
        sel=dd>0; D=D[sel]; dd=dd[sel]; m=Xs_.mean(0); Xc=Xs_-m; A=Xc.T@Xc; sc=np.trace(A)/192; r_=len(t)/max(len(dd),1)
        return np.linalg.solve(A+w*r_*(D.T@D)+lam*sc*np.eye(192),Xc.T@t+w*r_*(D.T@dd)),m
    def ev(a,m,Xs_,t,D,dd):
        sel=dd>0; s_=(Xs_-m)@a; return float((D[sel]@a)@dd[sel]/(dd[sel]@dd[sel])),float(s_.std()),float(np.corrcoef(s_,t)[0,1])
    CH={'i':(Gm,0,1,DI),'n':(Mm,2,3,DN)}; HP={}
    for nm,(msk,tc,dc,D) in CH.items():
        tr=msk&TRm; ho=msk&HOm; tab=[]
        for lam in [1e-3,1e-2,1e-1]:
            for w in [0.1,0.3,1,3]:
                a,m=dec2(X[tr],A_[tr,tc],D[tr],A_[tr,dc],w,lam); resp,sd,cr=ev(a,m,X[ho],A_[ho,tc],D[ho],A_[ho,dc]); tab.append((sd/resp if resp>0.15 else 1e9,lam,w,resp,sd,cr))
        b=min(tab); HP[nm]=b; print(nm,'pts',int(msk.sum()),'post-drug',int((msk&(A_[:,dc]>0)).sum()),'best lam %g w %g HO resp %.2f sd %.3f corr %.2f ratio %.2f'%(b[1],b[2],b[3],b[4],b[5],b[0]))
    def fitQ(mr):
        Q=dict(Q7)
        for nm,(msk,tc,dc,D) in CH.items():
            sel=msk&mr; a,m=dec2(X[sel],A_[sel,tc],D[sel],A_[sel,dc],HP[nm][2],HP[nm][1]); Q['a_'+nm]=a; Q['m_'+nm]=m
        return Q
    Qtr=fitQ(TRm); bI=-1.9/HP['i'][3] if HP['i'][0]<1e8 else 0.0; bN=3.0/HP['n'][3] if HP['n'][0]<1e8 else 0.0
    HO=SL[70:]; tbh=tfnll(W_cur,HO)[0]; dn={}
    for nm,f in [('i',.5),('i',1.),('n',.5),('n',1.)]:
        dn[(nm,f)]=tfnll(mkWz(bI*f if nm=='i' else 0.0,bN*f if nm=='n' else 0.0,Q=Qtr),HO)[0]-tbh
    print('bI %.2f bN %.2f HO dNLL'%(bI,bN),{k:round(v,4) for k,v in dn.items()},round(time.time()-t0))
    r0=evalW(W_cur,SL); r1=evalW(mkWz(bI,bN,Q=Qtr),SL); R0={k:v[0] for k,v in r0.items()}; R1={k:v[0] for k,v in r1.items()}; TT={k:v[1] for k,v in r0.items()}
    print('R0',{k:round(v,2) for k,v in R0.items()},'R1',{k:round(v,2) for k,v in R1.items()},'T',{k:round(v,2) for k,v in TT.items()})
    qi=2*dn[('i',1.)]-4*dn[('i',.5)]; li=dn[('i',1.)]-qi; qn=2*dn[('n',1.)]-4*dn[('n',.5)]; ln_=dn[('n',1.)]-qn; best=None
    for fi in np.arange(0,1.51,0.05):
        for fn in np.arange(0,1.51,0.05):
            cost=qi*fi**2+li*fi+qn*fn**2+ln_*fn
            if cost>0.0035: continue
            Rf={k:R0[k]+(fi if k<=2 else fn)*(R1[k]-R0[k]) for k in R0}
            sc=2*abs(Rf[1]-TT[1])/abs(TT[1])+sum(abs(Rf[k]-TT[k])/abs(TT[k]) for k in (2,3,4))
            if best is None or sc<best[0]: best=(sc,fi,fn,cost,Rf)
    fi,fn=best[1],best[2]; print('choice fi %.2f fn %.2f HOcost %.4f'%(fi,fn,best[3]),{k:round(v,2) for k,v in best[4].items()})
    Qall=fitQ(np.ones(len(rows),bool)); Wf=mkWz(bI*fi,bN*fn,Q=Qall)
    for nme in ['z_prior.0.weight','z_prior.0.bias','z_prior.2.weight','z_prior.2.bias','z_head.weight']: save_edit(nme,Wf[nme].float().contiguous())
    np.savez('/work/q9.npz',a_i=Qall['a_i'],m_i=Qall['m_i'],a_n=Qall['a_n'],m_n=Qall['m_n'],vg=Qall['vg'],vm=Qall['vm'],bI=bI*fi,bN=bN*fn)
    rf=evalW(Wf,SL); tff=tfnll(Wf,SL)[0]; tb=tfnll(W_cur,SL)[0]
    print('FINAL saved',edited(),{k:round(v[0],2) for k,v in rf.items()},'in-sample dNLL %.4f'%(tff-tb),round(time.time()-t0))
    with open('/work/notes.md','a') as f:
        f.write(f"- Run 297: rates from atoms (num[4]*iqr+median); decoders trained w/ resid targets + Delta-response penalty (HP {HP['i'][1:4]}, {HP['n'][1:4]}); HO dNLL {dict((str(k),round(v,4)) for k,v in dn.items())}; R0 {R0} R1 {R1} T {TT}; chose fi {fi:.2f} fn {fn:.2f}; SAVED z_prior/z_head edit (gains bI {bI*fi:.2f} bN {bN*fn:.2f}); final offline {dict((k,round(v[0],2)) for k,v in rf.items())}, in-sample dNLL {tff-tb:.4f}. Params in /work/q9.npz.\n")
except Exception as e:
    import traceback; print('ERR',str(e)[:200]); print(traceback.format_exc()[-500:])
```