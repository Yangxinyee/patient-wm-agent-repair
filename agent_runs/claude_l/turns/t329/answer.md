Run 297 produced the first edit that works within the factual budget, but the gain is limited by noise:
- **The trade-off.** At full gain the readouts move most of the way to the truth: insulin ×2 −10.6 against −14.2, ×1/2 +6.5 against +7.1, norepinephrine ×2 +4.1 against +4.9. The held-out NLL cost would be 0.015 for insulin and 0.027 for norepinephrine, far over budget.
- **What I saved.** Gains at 0.2 (insulin) and 0.3 (norepinephrine). Offline readouts are −5.1 / +2.6 / +1.8 / −2.1, with an in-sample NLL cost of +0.0017.
- **Why it is noisy.** The model keeps no memory of earlier doses. The residual it can see has sd 0.62 against 0.35 for the true policy residual. The norepinephrine policy is almost invisible to it (CV R² 0.06). On top of that, the head-side decoder reads the current dose noisily (noise/response 1.37 for insulin).

One plausible source of cost is noise at the prediction points that follow no drug bundle. There the correction should be exactly zero.

Plan for this run:
1. Gate each channel with a linear detector for "the last bundle contained this drug". In `z_prior` the gate is built as a difference of two GELU units, `gelu(γ+s) − gelu(γ)`. It passes s when the gate is on and gives about 0 when it is off.
2. Fit the decoders on post-drug points only.
3. Compare held-out cost and readouts against the ungated version at a budget of 0.0045.
4. Save whichever version scores better.
5. Print the batch-2 stay IDs, so the tool can check the factual cost on stays never used for fitting.

How I will read the outcome:
- **Confirms:** the gates separate cleanly and the gated version gives larger readouts for the same held-out cost.
- **Refutes:** the gates overlap, or the cost does not fall.

```python
import json, numpy as np, torch, time
t0=time.time()
try:
    bj=json.load(open('/work/batches.json')); bl=list(bj.values()) if isinstance(bj,dict) else bj
    print('batches',len(bl),'b0==BUN',set(map(str,bl[0]))==set(map(str,SL))); print('B2',','.join(str(x) for x in bl[1]))
except Exception as e: print('bj err',str(e)[:100])
try:
    hasI=(A_[:,0]!=0)|(A_[:,1]>0); hasN=(A_[:,2]!=0)|(A_[:,3]>0); CHg={'i':(Gm,hasI,0,1,DI),'n':(Mm,hasN,2,3,DN)}
    def lda(Xs_,y,lam=1e-2):
        mP=Xs_[y].mean(0); mN=Xs_[~y].mean(0); Xc=np.vstack([Xs_[y]-mP,Xs_[~y]-mN]); S=Xc.T@Xc/len(Xc); S+=lam*np.trace(S)/192*np.eye(192); return np.linalg.solve(S,mP-mN)
    def gpar(Xs_,y,g):
        sc=Xs_@g; th=(np.median(sc[y])+np.median(sc[~y]))/2; gap=min(np.percentile(sc[y],2)-th,th-np.percentile(sc[~y],98))
        return th,(8.0/gap if gap>0 else 16.0/(np.median(sc[y])-np.median(sc[~y])))
    HPg={}
    for nm,(msk,hs,tc,dc,D) in CHg.items():
        tr=msk&TRm&hs; ho=msk&HOm&hs; tab=[]
        for lam in [1e-3,1e-2]:
            for w in [0.3,1,3]:
                a,m=dec2(X[tr],A_[tr,tc],D[tr],A_[tr,dc],w,lam); resp,sd,cr=ev(a,m,X[ho],A_[ho,tc],D[ho],A_[ho,dc]); tab.append((sd/resp if resp>0.15 else 1e9,lam,w,resp,sd))
        HPg[nm]=min(tab)[1:]; print(nm,'gated dec lam %g w %g HO resp %.2f sd %.3f'%HPg[nm])
    def fitG(mr):
        Q=dict(Q7)
        for nm,(msk,hs,tc,dc,D) in CHg.items():
            sel=msk&mr; g=lda(X[sel],hs[sel]); th,G=gpar(X[sel],hs[sel],g); Q['g_'+nm]=g; Q['th_'+nm]=th; Q['G_'+nm]=G
            sp=sel&hs; a,m=dec2(X[sp],A_[sp,tc],D[sp],A_[sp,dc],HPg[nm][1],HPg[nm][0]); Q['a_'+nm]=a; Q['m_'+nm]=m
        return Q
    def mkWg(bi,bn,Q):
        Wt=dict(W_cur); W0w=torch.zeros(192,192,dtype=dtp); b0=torch.zeros(192,dtype=dtp)
        for j,nm in enumerate(['i','n']):
            g=Q['g_'+nm]*Q['G_'+nm]; c=-Q['G_'+nm]*Q['th_'+nm]; a=Q['a_'+nm]
            W0w[2*j]=torch.tensor(g+a,dtype=dtp); b0[2*j]=c-float(a@Q['m_'+nm]); W0w[2*j+1]=torch.tensor(g,dtype=dtp); b0[2*j+1]=c
        W2w=torch.zeros(64,192,dtype=dtp); b2=torch.zeros(64,dtype=dtp); W2w[0,0]=1; W2w[0,1]=-1; W2w[1,2]=1; W2w[1,3]=-1; b2[32:]=-30
        ZH=torch.zeros(192,32,dtype=dtp); ZH[:,0]=torch.tensor(bi*Q['vg'],dtype=dtp); ZH[:,1]=torch.tensor(bn*Q['vm'],dtype=dtp)
        Wt.update({'z_prior.0.weight':W0w,'z_prior.0.bias':b0,'z_prior.2.weight':W2w,'z_prior.2.bias':b2,'z_head.weight':ZH}); return Wt
    Qg=fitG(TRm)
    for nm,(msk,hs,tc,dc,D) in CHg.items():
        ho=msk&HOm; gm=Qg['G_'+nm]*(X[ho]@Qg['g_'+nm]-Qg['th_'+nm]); print(nm,'gate HO: P>6 %.3f N<-6 %.3f'%((gm[hs[ho]]>6).mean(),(gm[~hs[ho]]<-6).mean()))
    bIg=-1.9/HPg['i'][2]; bNg=3.0/HPg['n'][2]; dng={}
    for nm,f in [('i',.5),('i',1.),('n',.5),('n',1.)]:
        dng[(nm,f)]=tfnll(mkWg(bIg*f if nm=='i' else 0.0,bNg*f if nm=='n' else 0.0,Qg),HO)[0]-tbh
    R1g={k:v[0] for k,v in evalW(mkWg(bIg,bNg,Qg),SL).items()}
    print('gated bI %.2f bN %.2f HO dNLL'%(bIg,bNg),{k:round(v,4) for k,v in dng.items()},'R1g',{k:round(v,2) for k,v in R1g.items()},round(time.time()-t0))
    def choose(dn_,R1_,budget):
        qi=2*dn_[('i',1.)]-4*dn_[('i',.5)]; li=dn_[('i',1.)]-qi; qn=2*dn_[('n',1.)]-4*dn_[('n',.5)]; ln_=dn_[('n',1.)]-qn; best=None
        for fi in np.arange(0,1.21,0.05):
            for fn in np.arange(0,1.21,0.05):
                cost=qi*fi**2+li*fi+qn*fn**2+ln_*fn
                if cost>budget: continue
                Rf={k:R0[k]+(fi if k<=2 else fn)*(R1_[k]-R0[k]) for k in R0}
                sc=2*abs(Rf[1]-TT[1])/abs(TT[1])+sum(abs(Rf[k]-TT[k])/abs(TT[k]) for k in (2,3,4))
                if best is None or sc<best[0]: best=(sc,fi,fn,cost,Rf)
        return best
    bu=choose(dn,R1,0.0045); bg=choose(dng,R1g,0.0045)
    print('ungated',round(bu[0],3),round(bu[1],2),round(bu[2],2),{k:round(v,2) for k,v in bu[4].items()}); print('gated',round(bg[0],3),round(bg[1],2),round(bg[2],2),{k:round(v,2) for k,v in bg[4].items()})
    if bg[0]<bu[0]:
        Qf=fitG(np.ones(len(X),bool)); Wf=mkWg(bIg*bg[1],bNg*bg[2],Qf); tag='gated'; gains=(bIg*bg[1],bNg*bg[2])
    else:
        Wf=mkWz(bI*bu[1],bN*bu[2],Q=Qall); Qf=Qall; tag='ungated'; gains=(bI*bu[1],bN*bu[2])
    for nme in ['z_prior.0.weight','z_prior.0.bias','z_prior.2.weight','z_prior.2.bias','z_head.weight']: save_edit(nme,Wf[nme].float().contiguous())
    np.savez('/work/q10.npz',**{k:np.asarray(v) for k,v in Qf.items()},gains=np.array(gains))
    rf=evalW(Wf,SL); tff=tfnll(Wf,SL)[0]; tb=tfnll(W_cur,SL)[0]
    print('SAVED',tag,'gains %.2f %.2f'%gains,{k:round(v[0],2) for k,v in rf.items()},'in-sample dNLL %.4f'%(tff-tb),round(time.time()-t0))
    with open('/work/notes.md','a') as f:
        f.write(f"- Run 298: gated variant (LDA gate post-drug, gelu(g+s)-gelu(g)); HO dNLL gated {dict((str(k),round(v,4)) for k,v in dng.items())}, R1g {R1g}. Budget .0045 HO. SAVED {tag} gains {gains[0]:.2f},{gains[1]:.2f}; offline {dict((k,round(v[0],2)) for k,v in rf.items())}; in-sample dNLL {tff-tb:.4f}. Params /work/q10.npz. Next: tool factual both on batch2 to check out-of-sample cost.\n")
except Exception as e:
    import traceback; print('ERR',str(e)[:200]); print(traceback.format_exc()[-500:])
```