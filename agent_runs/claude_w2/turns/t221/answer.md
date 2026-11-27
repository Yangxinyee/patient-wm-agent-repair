The gg columns named ins_lk / nor_lk (and probably gf_lk etc.) are leads, not lags. At h=6 the row shows ins_l1 = 1.0 for the infusion that starts at hour 7. The grid was built for "dose at h → outcome at h+k" impulse analyses, which is why it worked for estimating Kins earlier. In run 214 it meant I regressed residuals on future doses while the policy columns were true lags, hence the contradiction.

I now build the lags explicitly by within-stay shifts:
- d_{h−k} = ins0 shifted by k.
- policy mean = ins0 − e_ins (e_ins is the deviation from the smooth policy mean; ins_base is the rounded band rate).
- for norepinephrine, expected rate = nor0 − e_nor, also shifted.

Expectation under my "absorbed policy" hypothesis for insulin: dose coefficients a ≈ (−1.7, −1.4, −0.4, …) and policy coefficients c ≈ −a. Then the deviation parametrization should show policy coefficients near 0.

For norepinephrine, I want to know whether the model is missing the effect of the rate *level* (expected-rate coefficients ≈ dose coefficients) or only the innovation. A lag-0 dose term (the infusion started 5 minutes after the reading) serves as a placebo: it should be ≈ 0.

```python
code_txt = r'''
def build_resid_tab2(tf_, gg_, K=6):
    G=gg_.sort_values(['stay_id','h'])[['stay_id','h','ins0','e_ins','nor0','e_nor','glu_now','map_pol']].copy()
    G['polm']=G.ins0-G.e_ins; G['nexp']=G.nor0-G.e_nor
    gb=G.groupby('stay_id')
    for k in range(0,K+1):
        G['d%d'%k]=gb.ins0.shift(k); G['p%d'%k]=gb.polm.shift(k); G['e%d'%k]=gb.e_ins.shift(k)
        G['n%d'%k]=gb.nor0.shift(k); G['x%d'%k]=gb.nexp.shift(k); G['u%d'%k]=gb.e_nor.shift(k)
    T=tf_[np.abs(tf_.t-np.round(tf_.t))<1e-6].copy(); T['h']=np.round(T.t).astype(int)
    M=T.merge(G,on=['stay_id','h'],how='left'); M['res']=M.value-M.pm
    return M
'''
open('/work/resid.py','a').write(code_txt); exec(code_txt)
M3=build_resid_tab2(tf2,gg)
L=[1,2,3,4,5]
for nm,cc in [('gf',6),('gl',10)]:
    Mi=M3[(M3.code==cc)&(M3.h>=2)]
    t1,n1,s1=ols_tab(Mi,'res',['d%d'%k for k in L]+['p%d'%k for k in L])
    t2,n2,s2=ols_tab(Mi,'res',['e%d'%k for k in L]+['p%d'%k for k in L])
    t3,_,_=ols_tab(Mi,'res',['d0','e0']+['e%d'%k for k in L])
    print(nm,'n',n1,'sd',round(s1,2)); print(pd.concat([t1.reset_index(),t2.reset_index(),t3.reset_index()],axis=1).to_string())
for nm,cc in [('ma',7),('mc',5)]:
    Mi=M3[(M3.code==cc)&(M3.h>=2)]
    t1,n1,s1=ols_tab(Mi,'res',['n%d'%k for k in L]+['x%d'%k for k in L])
    t2,n2,s2=ols_tab(Mi,'res',['n0']+['n%d'%k for k in L])
    print(nm,'n',n1,'sd',round(s1,2)); print(pd.concat([t1.reset_index(),t2.reset_index()],axis=1).to_string())
```