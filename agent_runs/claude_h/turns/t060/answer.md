The lab check is only weakly corroborating:

| Window | Finger | Lab | Stacked (finger + lab) |
|---|---|---|---|
| Hours ≥ 12 | +1.89 ± 0.69 | +0.97 ± 0.89 | +1.74 ± 0.75 |
| Hours < 12 | — | — | +0.17 ± 0.78 |
| All hours | +0.90 ± 0.51 | +0.30 ± 0.62 | +0.80 ± 0.51 |

Lab has the same sign as finger in the late window but about half the size. The early and late windows do not differ significantly (1.45σ). So a constant-effect reading, +0.8 ± 0.5 (1.6σ), is the fair summary, and it is weak.

I will keep zeros. With a large probability that the effect is exactly zero, zero minimises expected absolute error and is the most likely value. It also cannot be meaningfully worse than the original model, which sits at about +0.05 to +0.07 for insulin ×2.

One scenario could still hide a large effect: the recorded jitter or innovations might not reflect what was actually infused. In that case my dose instruments would show nothing even if the drugs worked. The sliding-scale thresholds and the norepinephrine start rule give independent tests. At each insulin threshold (150/180/220/260/300) the base dose jumps; norepinephrine starts when MAP falls below about 64.5. A realistic effect would leave a jump of several mg/dL or mmHg in later readings at these cut-offs.

- **Rules out large hidden effects:** local-linear jump estimates near 0 with standard errors of about 1 or less.
- **Reverses my conclusion:** a clear jump at the cut-offs.

```python
t0=time.time()
def rd_pooled(rowsS,rowsT,run,cuts,outcome,K=6,bw=15,label=''):
    res=[]
    for k in range(1,K+1):
        Xs=[];Ys=[];Ss=[]
        for c in cuts:
            m=(run>=c-bw)&(run<c+bw)
            s_=rowsS[m]; t_=rowsT[m]; tt=t_+k; ok=tt<=25
            y=np.full(m.sum(),np.nan); y[ok]=outcome[s_[ok],tt[ok]]-np.where(np.isnan(outcome[s_[ok],t_[ok]]),np.nan,outcome[s_[ok],t_[ok]])
            x=run[m]-c; D=(x>=0).astype(float); okk=~np.isnan(y)
            cd=np.zeros((okk.sum(),len(cuts))); cd[:,cuts.index(c)]=1
            Xs.append(np.column_stack([cd, D[okk], x[okk], x[okk]*D[okk]])); Ys.append(y[okk]); Ss.append(s_[okk])
        Xm=np.vstack([np.column_stack([x[:,:len(cuts)],x[:,len(cuts):]]) for x in Xs]); Y=np.concatenate(Ys); S_=np.concatenate(Ss)
        b,_,_,_=np.linalg.lstsq(Xm,Y,rcond=None); u=Xm*(Y-Xm@b)[:,None]
        cl=pd.DataFrame(u).groupby(S_).sum().to_numpy(); XtXi=np.linalg.inv(Xm.T@Xm); V=XtXi@(cl.T@cl)@XtXi
        j=len(cuts); res.append('k%d %.2f(%.2f)'%(k,b[j],np.sqrt(V[j,j])))
    print(label,' '.join(res))
# insulin thresholds on finger glucose (policy uses finger if present); rows with finger at t
sF,tF=np.nonzero(~np.isnan(GF)&(np.arange(26)[None,:]>=1)&(np.arange(26)[None,:]<=24))
gF=GF[sF,tF]
dose_jump={150:1,180:1,220:1,260:1,300:2}
print('mean insulin just above/below each cut (+-5):',{c:(round(INS0[sF,tF][(gF>=c)&(gF<c+5)].mean(),2),round(INS0[sF,tF][(gF>=c-5)&(gF<c)].mean(),2)) for c in dose_jump})
rd_pooled(sF,tF,gF,[150,180,220,260,300],GF,label='INSULIN RD (finger change)')
# NE start RD: rows not on NE at t-1, MAP reading at t (policy MAP: arterial else cuff)
MAPpol=np.where(np.isnan(GA),GC,GA)
sN0,tN0=np.nonzero((lag(NE0,1)==0)&~np.isnan(MAPpol)&(np.arange(26)[None,:]>=1)&(np.arange(26)[None,:]<=24))
mN=MAPpol[sN0,tN0]
for c in (64.5,):
    print('NE start prob just below/above %.1f: %.3f / %.3f; mean start rate below: %.3f'%(c,(NE0[sN0,tN0]>0)[(mN>=c-3)&(mN<c)].mean(),(NE0[sN0,tN0]>0)[(mN>=c)&(mN<c+3)].mean(),NE0[sN0,tN0][(mN>=c-3)&(mN<c)&(NE0[sN0,tN0]>0)].mean()))
rd_pooled(sN0,tN0,mN,[64.5],MAPc,K=6,bw=8,label='NE START RD (MAP change, sign: above-below)')
print('time',round(time.time()-t0,1))
```