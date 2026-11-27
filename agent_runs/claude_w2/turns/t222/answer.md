With correct lags the picture is clear and consistent across finger and lab glucose.

**Lag alignment checks out.** The lag-0 dose (infusion started 5 minutes after the reading) only predicts the residual through its policy part. Its random part has net coefficient +0.27 ≈ 0, as a placebo should.

**Insulin.** The model misses the effect of the dose deviation:

| | lag 1 | lag 2 | lag 3 | lag 4 | lag 5 |
|---|---|---|---|---|---|
| e coefficient, finger glucose (±0.17) | −1.27 | −1.35 | −0.63 | +0.60 | +0.83 |
| e coefficient, lab glucose | −1.34 | −1.01 | −0.66 | +0.59 | +0.85 |
| policy-lag coefficient, dose parametrization | +0.59 | +1.19 | +0.31 | −0.40 | −0.18 |

The policy-lag coefficients are partly the mirror image of the dose coefficients. So the model has absorbed roughly 60% of the policy-dose effect into its glucose-level dynamics, which supports a dose-plus-policy-compensation correction.

**Norepinephrine.** The model misses about +2.7 (arterial) and +3.0 (cuff) mmHg per mcg/kg/min at lag 1. The expected-rate term is not negative: +1.0 ± 1.0, where a pure-deviation story would predict −2.7. So the model misses the effect of the dose level, and the correction there needs no policy compensation.

Next:
1. Check that the policy function PF reproduces gg's policy mean, so my rollout correction uses the same p(g).
2. Get the norepinephrine regression on lags 1–5 without the bad lag-0 control.
3. Write a full-correction oracle for the local rollout engine. It adds Σa_k·d_{t−k} + Σc_k·p(g_{t−k}) + const to glucose in both arms, and Σa^N_k·n_{t−k} to MAP, using the drawn glucose for p.

If this factual-optimal correction then reproduces the true dose effects in rollouts, I will implement it in weights.

```python
chk=gg[['glu_now','ins0','e_ins']].dropna(); polm=chk.ins0-chk.e_ins
for nmf in ['PF','pfun']:
    f=globals().get(nmf)
    try:
        pv_=np.asarray(f(chk.glu_now.values),dtype=float); print(nmf,'maxdiff',np.abs(pv_-polm.values).max(),'mean',np.abs(pv_-polm.values).mean())
    except Exception as ex: print(nmf,'err',ex)
print('ids_o' in globals(), [k for k in globals() if k.startswith('ids') or k.startswith('Ro')])
for nm,cc in [('ma',7),('mc',5)]:
    Mi=M3[(M3.code==cc)&(M3.h>=2)]
    t1,n1,s1=ols_tab(Mi,'res',['n%d'%k for k in [1,2,3,4,5]])
    t2,_,_=ols_tab(Mi,'res',['n%d'%k for k in [1,2,3]])
    print(nm,n1,round(s1,2)); print(pd.concat([t1.reset_index(),t2.reset_index()],axis=1).to_string())
code_txt = r'''
def make_lin_oracle(R,DGi=None,a_i=None,c_i=None,k_i=0.0,pf=None,DGn=None,a_n=None,k_n=0.0,t_hide=13.0,gain_i=1.0,gain_n=1.0):
    A=R['A']; code=A.code.values; tt=A.t.values; val=A.value.values; loc=A['loc'].values; S_=len(R['stays'])
    GFI=np.full((S_,32),-1); GLI=np.full((S_,32),-1)
    for cc,arr in [(CODE['glucose_finger'],GFI),(CODE['glucose_lab'],GLI)]:
        m=np.where((code==cc)&(np.abs(tt-np.round(tt))<1e-6))[0]; arr[A.si.values[m],np.round(tt[m]).astype(int)+3]=m
    def gval(n,j,drawn):
        v=val[j].astype(float).copy(); hid=tt[j]>=t_hide
        if hid.any(): b=drawn[n[hid],loc[j[hid]]]; v[hid]=CENT[code[j[hid]],np.maximum(b,1)-1]
        return v
    def plags(n,si,tau,drawn,L):
        Pm=np.zeros((len(n),L))
        for kk in range(1,L+1):
            ix=tau-kk+3; ok=ix>=0; ixc=np.clip(ix,0,31)
            jf=GFI[si,ixc]; jl=GLI[si,ixc]; g=np.full(len(n),np.nan)
            m1=ok&(jf>=0); g[m1]=gval(n[m1],jf[m1],drawn); m2=ok&(jf<0)&(jl>=0); g[m2]=gval(n[m2],jl[m2],drawn)
            Pm[:,kk-1]=np.where(np.isnan(g),0.0,pf(np.nan_to_num(g)))
        return Pm
    def hook(k,hn_,hj,P,ctx):
        P=P.clone()
        for codes_,drug in [(GLC,'i'),(MPC,'n')]:
            DG=DGi if drug=='i' else DGn
            if DG is None: continue
            m=np.isin(code[hj],codes_)
            if not m.any(): continue
            n=hn_[m]; si=ctx['seq_si'][n]; a=ctx['seq_arm'][n]; tau=np.round(tt[hj[m]]).astype(int)
            if drug=='i':
                d=lag_abs(DG,a,si,tau,len(a_i)); r=d@np.asarray(a_i)+k_i
                if c_i is not None: r=r+plags(n,si,tau,ctx['drawn'],len(c_i))@np.asarray(c_i)
                r=gain_i*r
            else:
                d=lag_abs(DG,a,si,tau,len(a_n)); r=gain_n*(d@np.asarray(a_n)+k_n)
            ii=torch.as_tensor(np.where(m)[0])
            P[ii]=tilt_P(P[ii],torch.as_tensor(CENT[code[hj[m]]],dtype=torch.float32),r)
        return P
    hook.plags=plags; return hook
'''
open('/work/linor.py','w').write(code_txt); exec(code_txt); print('ok')
```