Run 220 shows the stored dataset. It has 5,214 factual rows and about 63,000 counterfactual rows for insulin and norepinephrine, covering starts at hours 6, 12 and 16 and multipliers 0 to 3. Each row carries a 192-number head-feature vector (`WC`) and a target tilt (`TH`).

Next I will measure how much dose signal the head features can carry at best. I will pair each insulin x2 (hour 12) glucose row with its factual twin and take the feature difference D. The best linear readout is then Σ_f⁻¹·mean(D), with Σ_f the factual feature covariance. I will compute it out-of-fold by stay at several ridge strengths and report its signal-to-noise ratio: the mean counterfactual response divided by the factual spread.

A rough calculation gives the threshold. Suppose the full correction is a tilt of about −1 on counterfactual rows, and the factual spread must stay below about 0.24 to fit the 0.005 limit. Then the ratio must exceed about 4. A ratio above 4 means the head approach can work and its earlier failure was overfitting. A ratio near 1–2 means the head features cannot do it.

```python
isGl=M['event'].str.startswith('glucose').values
F=M[(M.drug=='none')&isGl].reset_index().rename(columns={'index':'iF'})
C=M[(M.drug=='insulin')&(M.h0==12)&(M.m==2.0)&isGl].reset_index().rename(columns={'index':'iC'})
J=C.merge(F[['stay','b','event','iF']],on=['stay','b','event'],how='inner')
J=J[J.hour>=1]; print(len(F),len(C),len(J))
from sklearn.model_selection import GroupKFold
iC=J.iC.values; iFm=J.iF.values; Dm=WC[iC]-WC[iFm]; Xf=WC[F.iF.values]; gF=F.stay.values; gJ=J.stay.values
for lam in [1e-3,1e-2,1e-1,1.0]:
    rs=[]
    for tr,te in GroupKFold(5).split(Xf,groups=gF):
        trs=set(gF[tr]); mJ=np.isin(gJ,list(trs))
        mu=Xf[tr].mean(0); Sf=np.cov((Xf[tr]-mu).T); w=np.linalg.solve(Sf+lam*np.trace(Sf)/192*np.eye(192),Dm[mJ].mean(0))
        rs.append(((Dm[~mJ]@w).mean(), (Xf[te]@w-mu@w).std(), (Dm[mJ]@w).mean(), (Xf[tr]@w-mu@w).std()))
    r=np.array(rs).mean(0); print(lam,'oof SNR %.2f (cf %.3f sd %.3f) | in SNR %.2f'%(r[0]/r[1],r[0],r[1],r[2]/r[3]))
```