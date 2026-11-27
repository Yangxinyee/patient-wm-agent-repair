Stored head inputs for 154k teacher-forced readings across the five insulin arms (600 stays), in 14 s.

Next is the feasibility test. For glucose readings I'll compute the head's feature vector exactly as the value MLP sees it (query at the prior-mean z, plus the last-value register terms). The decoding targets are the insulin rates at lags 1–3 under each arm and the policy dose at lags 1–2. Each target gets a ridge decoder fitted on half of the stays and evaluated on the other half. The key number is the slope of decoded-vs-true dose change between edited and factual arms. Near 1 makes the head-level correction viable; strongly attenuated means the state does not carry the dose cleanly enough.

```python
A_I=RI['A']; nAI=len(armsI); S_I=len(RI['stays'])
codeI=A_I.code.values[hjI]; tI=A_I.t.values[hjI]; siI=hnI//nAI; armI=hnI%nAI
# hourly factual insulin rate & glucose per stay
H0=-3; NH=32
Dm=np.zeros((S_I,NH)); 
ia=A_I[(A_I.code==CODE['insulin'])&(A_I.kind==0)]
Dm[ia.si.values, np.floor(ia.t.values).astype(int)-H0]=ia.rate.values
Gm=np.full((S_I,NH),np.nan)
for c in [CODE['glucose_lab'],CODE['glucose_finger']]:   # gf overrides gl
    ga=A_I[(A_I.code==c)&(A_I.kind==0)]
    Gm[ga.si.values, np.round(ga.t.values).astype(int)-H0]=ga.value.values
Gff=pd.DataFrame(Gm.T).ffill().values.T; POLm=pfun(np.nan_to_num(Gff,nan=0.0))
fac=np.array([a.get('ins',1.0) for a in armsI]); frm=np.array([a.get('ins_from',12.0) for a in armsI])
hh=np.round(tI).astype(int)
def dose_lag(k):
    h_=hh-k; ok=(h_-H0>=0)
    d=np.where(ok,Dm[siI,np.clip(h_-H0,0,NH-1)],0.0)
    return d*np.where(h_>=frm[armI],fac[armI],1.0), d
def pol_lag(k):
    h_=hh-k; return np.where(h_-H0>=0,POLm[siI,np.clip(h_-H0,0,NH-1)],0.0)
glu_rows=np.isin(codeI,[CODE['glucose_finger'],CODE['glucose_lab']])
# head feature u
with torch.no_grad():
    view=baseI+F.linear(zmuI,W['z_head.weight'],W['z_head.bias'])
    qI=LN(view+W['type_emb.weight'][1],W,'query_norm')
    uI=(qI+F.linear(f8I,W['last_numeric.weight'],W['last_numeric.bias'])+W['embed.bins.value.weight'][lbI]).numpy()
pmI=np.array([resI['pm'][int(n*10**7+j)] for n,j in zip(hnI,hjI)])
pm0=np.array([resI['pm'].get(int((s*nAI)*10**7+j),np.nan) for s,j in zip(siI,hjI)])
from sklearn.linear_model import Ridge
trn=(siI%2==0); tst=~trn
m=glu_rows
Xd=np.c_[uI,(codeI==CODE['glucose_lab']).astype(float)]
for nm_,tg in [('d1',dose_lag(1)[0]),('d2',dose_lag(2)[0]),('d3',dose_lag(3)[0]),('pol1',pol_lag(1)),('pol2',pol_lag(2))]:
    rg=Ridge(alpha=1.0).fit(Xd[m&trn],tg[m&trn]); pr_=rg.predict(Xd[m&tst])
    r2=1-np.mean((tg[m&tst]-pr_)**2)/np.var(tg[m&tst])
    # counterfactual slope: decoded(arm)-decoded(fact) vs true diff, on test rows of perturbed arms
    dec=rg.predict(Xd); 
    key0=pd.Series(dec[armI==0],index=pd.MultiIndex.from_arrays([siI[armI==0],hjI[armI==0]]))
    dec0=key0.reindex(pd.MultiIndex.from_arrays([siI,hjI])).values
    tg0=pd.Series(tg[armI==0],index=key0.index).reindex(pd.MultiIndex.from_arrays([siI,hjI])).values
    mm_=m&tst&(armI>0)&(np.abs(tg-tg0)>1e-9)
    sl=np.polyfit(tg[mm_]-tg0[mm_],dec[mm_]-dec0[mm_],1)[0] if mm_.sum()>10 else np.nan
    print(nm_,'R2',round(r2,3),'cf slope',round(sl,3),'n_cf',mm_.sum())
```