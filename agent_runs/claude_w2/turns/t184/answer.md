Two clear findings.

First, the difference-based oracle (the exact dose change per arm, filter G′) reproduces my truth within about 15%: x2 −18.9 vs −16.5, x½ +9.0 vs +8.3, x0 +15.5 vs +16.5, x2 from hour 16 −15.3 vs −14.2. So the absorption and innovation logic is about right.

Second, the deviation-based version (dose − p(drawn glucose)) does worse. It overshoots (x2 −22.7, x½ +10.6, x0 +18.1), because p falls as the drawn glucose falls. It also slightly worsens factual glucose NLL (+0.0009) instead of improving it. So the model does not contain the policy-feedback compensation I assumed. This is consistent with drawn shifts persisting without reverting.

That raises a key question: do the model's factual residuals depend on recent doses at all? Next I regress teacher-forced residuals (2000 training stays, saved in tf2k_orig.parquet) on dose lags, policy lags and glucose lags.
- A strong lag-1/2 dose coefficient (≈ −1.7) means the model misses the dose effect on factual data, so a properly compensated correction should gain NLL.
- Near-zero coefficients mean the factual information is already absorbed elsewhere, and the correction must be built to vanish on factual histories.

```python
tf2=pd.read_parquet('/work/tf2k_orig.parquet'); print(tf2.shape, tf2.columns.tolist()); print(tf2.head(2).to_string())
ids2=np.array(sorted(tf2.stay_id.unique())); sub=tr[tr.stay_id.isin(ids2)]
s_of={s:i for i,s in enumerate(ids2)}
# hourly insulin dose grid and policy glucose grid
ins=sub[sub.event=='insulin']; Dg=np.zeros((len(ids2),32)); np.add.at(Dg,(ins.stay_id.map(s_of).values,np.floor(ins.hour.values+1e-6).astype(int)+3),ins.rate.values)
gg_=sub[sub.event.isin(['glucose_finger','glucose_lab'])&(np.abs(sub.hour-np.round(sub.hour))<1e-6)]
Gq=np.full((len(ids2),32),np.nan)
for ev in ['glucose_lab','glucose_finger']:   # finger overrides lab
    x_=gg_[gg_.event==ev]; Gq[x_.stay_id.map(s_of).values,np.round(x_.hour.values).astype(int)+3]=x_.value.values
Pq=np.where(np.isnan(Gq),0.0,pfv(np.nan_to_num(Gq)))
f=tf2[tf2.code.isin(GLC)&tf2.value.notna()].copy(); f['si']=f.stay_id.map(s_of).values; f['tau']=np.round(f.t).astype(int)
f=f[(f.tau>=2)]
L_=10; Xd=np.zeros((len(f),L_)); Xp=np.zeros((len(f),L_)); Xg=np.zeros((len(f),3))
for k in range(1,L_+1):
    ix=np.clip(f.tau.values-k+3,0,31); Xd[:,k-1]=Dg[f.si.values,ix]; Xp[:,k-1]=Pq[f.si.values,ix]
for k in range(1,4):
    ix=np.clip(f.tau.values-k+3,0,31); Xg[:,k-1]=np.nan_to_num(Gq[f.si.values,ix],nan=np.nanmean(Gq))
res_=f.value.values-f.pm.values
def ols(y,X):
    X1=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(X1,y,rcond=None)[0]; e=y-X1@b
    se=np.sqrt(np.diag(np.linalg.inv(X1.T@X1))*e.var()); return b[1:],se[1:]
b1,s1=ols(res_,Xd-Xp); print('resid ~ dev lags      :',np.round(b1,2),' se',np.round(s1,2))
b2,s2=ols(res_,np.column_stack([Xd,Xp,Xg])); print('resid ~ d lags       :',np.round(b2[:L_],2)); print('        p lags       :',np.round(b2[L_:2*L_],2)); print('        g lags       :',np.round(b2[2*L_:],3))
b3,s3=ols(f.value.values,np.column_stack([Xd,Xp,Xg])); print('value ~ d lags (data):',np.round(b3[:L_],2)); 
b4,s4=ols(f.pm.values,np.column_stack([Xd,Xp,Xg])); print('pm ~ d lags (model)  :',np.round(b4[:L_],2))
print('n',len(f),' resid sd',res_.std())
```