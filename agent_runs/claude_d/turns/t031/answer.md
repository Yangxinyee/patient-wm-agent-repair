The helper failed on an array shape. The observation mask was a 2-D stay × hour array where a per-stay vector was needed. I am passing a 1-D mask instead (all stays for insulin, arterial-line stays for norepinephrine) and rerunning the same subgroup checks for linearity and homogeneity.

```python
def gest_sub(Y, D, EPSX, contmask, K_, coefs_prev, obsmask, sub_s, ctrl_fn, wts=None):
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
ones1 = np.ones(len(stays),bool)
for nm_, sub in [('base1',BASE==1),('base2',BASE==2),('base3+',BASE>=3),('rate<=1.5',INS<=1.5),('rate 2-3',(INS>=2)&(INS<=3)),('rate>=3.5',INS>=3.5)]:
    out=[gest_sub(GF, INS, EPS, GIVEN, k, beta_ins, ones1, sub, ctrl_ins, W) for k in [1,2]]
    print('INS',nm_, [(round(b,2),round(se,2),n) for b,se,n in out])
for nm_, sub in [('prev<0.3',NP<0.3),('prev 0.3-0.6',(NP>=0.3)&(NP<0.6)),('prev>=0.6',NP>=0.6),('map<65',Mdec<65),('map 65-75',(Mdec>=65)&(Mdec<75)),('map>=75',Mdec>=75)]:
    out=[gest_sub(MA, NOR, EPSN, cont, k, gam, art_stay, sub, ctrl_ne) for k in [1,2]]
    print('NE',nm_, [(round(b,2),round(se,2),n) for b,se,n in out])
```