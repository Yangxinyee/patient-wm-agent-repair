The implied true readouts on these 100 tuning stays are now complete. The norepinephrine shift builds from +1.7 at hour 13 to a plateau of about +5.2 by hour 17. Both placebo tests pass: the dose residual is uncorrelated with the pre-dose readings (r = 0.003 for insulin, 0.004 for norepinephrine, n of about 250k and 145k).

| edit | model now | implied truth |
|---|---|---|
| insulin x2 | −1.73 | −14.5 |
| insulin x1/2 | +0.91 | +7.3 |
| norepinephrine x2 | −0.08 | +4.6 |
| norepinephrine x1/2 | −0.48 | −2.3 |

Two assumptions remain behind these truths: linearity in dose (norepinephrine x2 pushes rates past the 1.0 cap seen in the data) and homogeneity across patient states. Next I will re-estimate the lag-1 and lag-2 effects in subgroups:
- insulin by base and by rate level;
- norepinephrine by previous-rate level and MAP band.

Constant per-unit effects support linear extrapolation. A response that shrinks at higher rates would mean saturation, and the x2 targets would need to come down.

```python
def gest_sub(Y, D, EPSX, contmask, K_, coefs_prev, obsmask, sub_s, ctrl_fn, wts=None):
    """lag-K_ estimate given previous lag coefs, restricted to s in sub_s"""
    Yt=[];E=[];A=[];C=[];Wt=[]
    for t in range(K_, H):
        s=t-K_
        obs = obsmask & ~np.isnan(Y[:,t])
        yt = Y[:,t] - sum(coefs_prev[j]*D[:,t-j] for j in range(1,K_))
        sel = obs & contmask[:,s] & sub_s[:,s]
        Yt.append(yt[sel]); E.append(EPSX[sel,s]); A.append(D[sel,s]); C.append(ctrl_fn(sel,s)); Wt.append(np.ones(sel.sum()) if wts is None else wts[sel,t])
    Yt=np.concatenate(Yt);E=np.concatenate(E);A=np.concatenate(A);C=np.concatenate(C);Wt=np.concatenate(Wt)
    Xc=np.c_[np.ones(len(Yt)),C]; sw=np.sqrt(Wt)
    coef=np.linalg.lstsq(Xc*sw[:,None], Yt*sw, rcond=None)[0]; r=Yt-Xc@coef
    b=np.sum(Wt*r*E)/np.sum(Wt*A*E); u=r-b*(A-A.mean()); se=np.sqrt(np.sum((Wt*E*u)**2))/abs(np.sum(Wt*A*E))
    return b, se, len(Yt)
ctrl_ins = lambda sel,s: np.c_[Gdec[sel,s], Gdec[sel,s]**2/100, (BASE[sel,s][:,None]==np.array([1,2,3,4,6])).astype(float)]
ctrl_ne = lambda sel,s: np.c_[np.nan_to_num(Mdec[sel,s],nan=70), np.isnan(Mdec[sel,s]), NP[sel,s], NP[sel,s]**2, np.nan_to_num(Mdec[sel,s],nan=70)**2/100]
allm = np.ones_like(GF,bool)
for nm_, sub in [('base1',BASE==1),('base2',BASE==2),('base3+',BASE>=3),('rate<=1.5',INS<=1.5),('rate 2-3',(INS>=2)&(INS<=3)),('rate>=3.5',INS>=3.5)]:
    out=[gest_sub(GF, INS, EPS, GIVEN, k, beta_ins, allm, sub, ctrl_ins, W) for k in [1,2]]
    print('INS',nm_, [(round(b,2),round(se,2),n) for b,se,n in out])
for nm_, sub in [('prev<0.3',NP<0.3),('prev 0.3-0.6',(NP>=0.3)&(NP<0.6)),('prev>=0.6',NP>=0.6),('map<65',Mdec<65),('map 65-75',(Mdec>=65)&(Mdec<75)),('map>=75',Mdec>=75)]:
    out=[gest_sub(MA, NOR, EPSN, cont, k, gam, art_stay[:,None]&allm, sub, ctrl_ne) for k in [1,2]]
    print('NE',nm_, [(round(b,2),round(se,2),n) for b,se,n in out])
```