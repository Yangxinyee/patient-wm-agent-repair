The model's factual errors do carry the missing dose effects:
- **Insulin:** regressing observed minus predicted glucose on the insulin residuals at lags 1–4 gives β = (−1.72, −1.66, −0.43, −0.44), se about 0.37.
- **Norepinephrine:** for MAP, β = (2.7, 2.7, 1.5, 2.5), se about 1.

The problem is that q, the only input the value head has, encodes those residuals weakly. A ridge probe on with_code recovers the lag-1 insulin residual with cross-validated R² 0.21, lag 2 only 0.04, and lag-1 norepinephrine 0.10. A shrunk probe like that responds to a real dose change by only about its R² times the change. So the counterfactual gain would have to be about 5×. At 5× the probe noise would cost roughly 0.003–0.004 nats per reading for each drug, which is too close to the 0.005 budget.

What matters for the dose questions is how q moves per unit of dose with the readings held fixed. So I'll fit an encoding model: regress q on the residuals at lags 1–3, plus the readings-implied doses and the last value as controls. From it I'll build a generalised-least-squares decoder, which responds one-for-one to the dose direction and has zero response to the control directions. Its noise, cross-validated by stay, tells me the best factual cost achievable.

- A decoder R² well above 0.2 would make a low-cost, near-unbiased correction feasible.
- A decoder R² near 0.2 means a linear readout of q can't do better, and I'd need to accept an attenuated probe and compensate its bias through the code×last-bin table.

```python
def lagarr(E,l):
    t=hr_r-l; ok=(t>=0)&(t<=24); out=np.zeros(len(t)); out[ok]=E[si_[ok],t[ok]]; return out
def gls_study(msk_, eps_, dose_, lab):
    R=[lagarr(eps_,l) for l in (1,2,3)]; I=[lagarr(dose_-eps_,l) for l in (1,2,3)]
    X=np.column_stack([np.ones(len(hr_r))]+R+I+[zl,has_.astype(float)])[msk_]
    Qm=WCA[msk_]; grp_s=M4.stay_id.values[msk_]; y=X[:,1]
    out={}
    for alpha in [0.1,1.0,10.0]:
        pr=np.zeros(len(y)); att=[]
        for tri,tei in GroupKFold(5).split(Qm,y,grp_s):
            B,*_=np.linalg.lstsq(X[tri],Qm[tri],rcond=None); Res=Qm[tri]-X[tri]@B
            S=Res.T@Res/len(tri); S+=alpha*np.trace(S)/192*np.eye(192)
            Bs=B[1:]; Si=np.linalg.inv(S); Wd=np.linalg.solve(Bs@Si@Bs.T, Bs@Si)
            pr[tei]=(Qm[tei]-B[0])@Wd[0]
        out[alpha]=1-((pr-y)**2).mean()/y.var()
    m=Ridge(alpha=10.0).fit(Qm,y); B,*_=np.linalg.lstsq(X,Qm,rcond=None)
    print(lab,'GLS decoder CV R2 by alpha',{k:round(v,3) for k,v in out.items()},
          '| ridge probe response to encoding dir: lag1 %.3f lag2 %.3f'%(m.coef_@B[1],m.coef_@B[2]), '| |v1| %.3f'%np.linalg.norm(B[1]))
gls_study(glu, eps_ins, d_all, 'INS/glucose')
gls_study(mapm, eps_ne, ne_all, 'NE/MAP')
```