Run 71 settles where amplified dose input goes. With the z-score inputs ×5, the state at insulin bundles moves only 2.85 units per unit z. That move is nearly orthogonal (cosine −0.04) to the "natural dose direction" from run 70. So that direction is mostly glucose context correlated with dose, not the drug's own signal. The model passes very little dose information into the state: input amplification would need about ×70 to reach the target, which is far out of range.

There is still a possible way through. The edited-minus-original state difference from pass 5 is a clean, unconfounded trace of the linear dose channel, s_e. If the model's natural dynamics rarely move along s_e, it is a selective carrier. I could then add gain along it without disturbing the factual fit. Next I will measure:
- the edit-induced change per unit z at drug bundles and how it decays at later bundles;
- the natural variation along s_e at reading and drug bundles, in both state changes and state levels;
- the same for norepinephrine, and the cosine between the insulin and norepinephrine traces.

A per-unit-z change that is comparable to or larger than the natural sd along s_e would confirm a usable channel. Natural sd much larger than the signal would refute it.

```python
def trace_dir(mask, zfun):
    m=mask&ok_
    zz_=zfun(m)
    X=np.column_stack([np.ones(m.sum()),zz_])
    B,_,_,_=np.linalg.lstsq(X,dYe[m],rcond=None)
    return B[1], m, zz_
s_i,m_i2,z_i=trace_dir(grp_['ins_only'],lambda m:(R_[m,0]-1.5)/1.0)
s_n,m_n2,z_n=trace_dir(grp_['ne_only'],lambda m:(R_[m,1]-0.46)/0.59)
Ym=Y_.reshape(-1,4,192).mean(1); dYm=dY.reshape(-1,4,192).mean(1); dYem=dYe.reshape(-1,4,192).mean(1)
for nm_,s_,m_,zz_ in [('ins',s_i,m_i2,z_i),('ne',s_n,m_n2,z_n)]:
    sm=s_.reshape(4,192).mean(0); sh=sm/np.linalg.norm(sm)
    sl=s_.reshape(4,192); nrm=np.linalg.norm(sl,axis=1)
    cosl=[sl[a]@sl[0]/nrm[a]/nrm[0] for a in range(4)]
    # fit quality: how much of dYe at these bundles is explained by z along sh
    proj=dYem[m_]@sh
    r2=np.corrcoef(proj,zz_)[0,1]**2
    print(nm_,'per-unit-z edit change (x4 extra): %.2f; slot cos to slot0'%np.linalg.norm(sm),np.round(cosl,2),'R2(proj~z) %.2f'%r2)
    print('   natural sd along s_hat: dY reading %.2f | dY drug %.2f | y level %.2f (mean %.2f) | natural z-path write per unit z %.2f'%((dYm[mr&ok_]@sh).std(),(dYm[(hi_|hn_)&ok_]@sh).std(),(Ym@sh).std(),(Ym@sh).mean(),np.linalg.norm(sm)/4))
    # persistence: edit-induced shift along sh over subsequent bundles (lag in bundles)
    idxs=np.where(m_)[0]; lagv=[]
    for L in range(1,7):
        vals=[];zs=[]
        for i0,zv in zip(idxs,zz_):
            j=i0
            for _ in range(L):
                nx=np.where(prev_==j)[0]
                if len(nx)==0: j=-1;break
                j=nx[0]
            if j>=0: vals.append(Ym[j]@sh*0+ (Ye_[j].reshape(4,192).mean(0)-Ym[j])@sh); zs.append(zv)
        vals=np.array(vals);zs=np.array(zs)
        lagv.append(np.polyfit(zs,vals,1)[0] if len(zs)>10 else np.nan)
    print('   level shift along s_hat per unit z at bundle lags 0..6:',np.round([np.polyfit(zz_,(Ye_[m_].reshape(-1,4,192).mean(1)-Ym[m_])@sh,1)[0]]+lagv,2))
print('cos(s_ins, s_ne) slot-mean: %.3f'%(s_i.reshape(4,192).mean(0)@s_n.reshape(4,192).mean(0)/np.linalg.norm(s_i.reshape(4,192).mean(0))/np.linalg.norm(s_n.reshape(4,192).mean(0))))
```