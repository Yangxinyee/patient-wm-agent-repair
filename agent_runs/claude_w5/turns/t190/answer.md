The centring term is better taken from the model's own state:
- **Insulin:** a ridge readout of the pre-drug state predicts the rate with held-out R² 0.78. Glucose bands alone give 0.69, with band means of about 0.07, 0.98, 1.95, 2.81 and 3.36 u/h.
- **Norepinephrine:** the state predicts the rate with held-out R² 0.53.

I will take ĉ from the state readout for both drugs (λ = 10).

Next, the gate: a single readout r of the last block FFN's hidden units, with no intercept, so that r·hidden gives:
- **at decision bundles:** dose minus ĉ, for the recorded dose and for each x2 and x1/2 variant;
- **at all other bundles:** about 0.

I will fit insulin and norepinephrine jointly by ridge on the 70 fitting stays, over a few penalties. On the 30 held-out stays I will report:
- R² of the target at decision bundles, recorded and edited doses;
- the leakage RMS elsewhere.

High R² with leakage well below the residual-dose spread would confirm that the gate can be built from this layer.

```python
isdec=lambda d: abs((d['t']%1)-1/12)<1e-3 or ('hid_i2' in d)
def sfeat(d):
    r=RK[d['sid']][d['k']]; return torch.cat([r['out0'],r['h0'].mean(0)]).numpy()
Dd=[d for d in D if isdec(d)]
Xa=(np.stack([sfeat(d) for d in Dd])-mu)/sd
fitd=np.array([d['sid'] in FIT for d in Dd])
RIa=np.array([d['ri'] for d in Dd]); RNa=np.array([d['rn'] for d in Dd])
mI=Ridge(alpha=10).fit(Xa[fitd],RIa[fitd]); mN=Ridge(alpha=10).fit(Xa[fitd],RNa[fitd])
for d,ci,cn in zip(Dd,mI.predict(Xa),mN.predict(Xa)): d['cI']=ci; d['cN']=cn
rows=[]
for d in D:
    f=d['sid'] in FIT
    if isdec(d):
        ri,rn,ci,cn=d['ri'],d['rn'],d['cI'],d['cN']
        rows.append((d['hid'],ri-ci,rn-cn,f,1,'fac'))
        if 'hid_i2' in d:
            rows+= [(d['hid_i2'],2*ri-ci,rn-cn,f,1,'i2'),(d['hid_ih'],.5*ri-ci,rn-cn,f,1,'ih'),(d['hid_n2'],ri-ci,2*rn-cn,f,1,'n2'),(d['hid_nh'],ri-ci,.5*rn-cn,f,1,'nh')]
    else: rows.append((d['hid'],0.,0.,f,0,'non'))
Xh=torch.cat([r[0] for r in rows]).double()          # (Nrows*4,768)
Yh=torch.tensor([[r[1],r[2]] for r in rows for _ in range(4)],dtype=torch.float64)
Fm=torch.tensor([r[3] for r in rows for _ in range(4)]); Dm=torch.tensor([r[4] for r in rows for _ in range(4)])
Tg=np.array([r[5] for r in rows for _ in range(4)])
print('rows',Xh.shape,'hid mean abs %.3f'%float(Xh.abs().mean()))
XtX=Xh[Fm].T@Xh[Fm]; XtY=Xh[Fm].T@Yh[Fm]
for lam in [1e-2,1e-1,1,10]:
    Rr=torch.linalg.solve(XtX+lam*torch.eye(768,dtype=torch.float64),XtY); Pv=Xh@Rr
    v=~Fm; dv=v&(Dm==1); nv=v&(Dm==0)
    r2s=[1-float(((Pv[dv,j]-Yh[dv,j])**2).mean()/Yh[dv,j].var()) for j in (0,1)]
    leak=[float(Pv[nv,j].pow(2).mean().sqrt()) for j in (0,1)]
    print('lam',lam,'val R2 I %.3f N %.3f'%tuple(r2s),'| leak rms I %.3f N %.3f'%tuple(leak))
```