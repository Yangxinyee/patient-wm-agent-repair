Run 232 gives mixed results.
- **Gate:** the drug-bundle indicator decodes almost exactly from u (error ≤ 0.003, exactly 0 on the empty encoding).
- **Rates:** a single linear read-out does poorly across the full 0–30 insulin grid (rms 1.2) and is mediocre for norepinephrine (rms 0.08). The action encoder is nonlinear in the rate: a clipped z-score plus bin embeddings, passed through MLPs.

What matters is accuracy on the rates that actually occur, factual and scaled by ×0, ×½, ×1.5, ×2 and ×3. So I'll refit the linear read-outs on that distribution, built from the 8,707 recorded drug bundles. That includes the ×0 case, where atoms stay present with rate 0.

The single-unit design is confirmed if rms errors are small next to the innovation spread (insulin 0.66, norepinephrine 0.23) and roughly unbiased within each multiplier. If errors are large or systematic by multiplier, I'll switch to a small multi-unit GELU read-out.

```python
rng_=np.random.default_rng(0); pick_=rng_.choice(len(dr),1500,replace=False)
rowsU=[];ta=[];tb=[];mm_=[]
for j in pick_:
    key_=(int(stA[dr[j]]),round(float(hrA[dr[j]]),3)); at=SB.get(key_,[])
    for mlt in [0,0.5,1,1.5,2,3]:
        for which in ['A','B']:
            if which=='B' and mlt==1: continue
            atoms=[];a_=0.0;b_=0.0
            for c,r in at:
                rr=r*mlt if ((c==8 and which=='A') or (c==9 and which=='B')) else r
                atoms.append(datom(c,rr,0)); a_+= rr if c==8 else 0; b_+= rr if c==9 else 0
            if atoms: rowsU.append(uenc(atoms)); ta.append(a_); tb.append(b_); mm_.append((which,mlt))
UR=np.stack(rowsU); ta=np.array(ta); tb=np.array(tb); Xu=np.vstack([UR,UE[None,:]]); wv=np.ones(len(UR)+1); wv[-1]=100.0
for nmv,t_ in [('A',ta),('B',tb)]:
    y_=np.concatenate([t_,[0.0]])
    for lam_ in [1e-5,1e-4,1e-3]:
        A_=np.vstack([Xu*wv[:,None],np.sqrt(lam_)*np.eye(192)]); p_=np.linalg.lstsq(A_,np.concatenate([y_*wv,np.zeros(192)]),rcond=None)[0]
        e_=UR@p_-t_; PD[(nmv+'r',lam_)]=p_
        bym=' '.join('x%s:%.3f/%.3f'%(m,e_[[mm==(nmv,m) for mm in mm_]].mean(),np.sqrt((e_[[mm==(nmv,m) for mm in mm_]]**2).mean())) for m in [0,0.5,1.5,2,3])
        print(nmv,lam_,'|p| %.0f rms %.3f max %.3f'%(np.linalg.norm(p_),np.sqrt((e_**2).mean()),np.abs(e_).max()),'| bias/rms by mult',bym)
```