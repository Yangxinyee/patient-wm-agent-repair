Run 213 confirmed the inputs I need:
- **tf2:** teacher-forced predicted means (pm) of the original model for 94,387 readings in 2,000 train stays.
- **gg:** the stay-hour grid, with dose lags ins_l0..l12 / nor_l0..l12, the policy base rate ins_base, the insulin deviation e_ins and the norepinephrine innovation e_nor.
- **Codes:** glucose_finger 6, glucose_lab 10, map_arterial 7, map_cuff 5. Bin centres are 72..271 (finger glucose) and 33..101 (arterial MAP).

Next I regress the model's teacher-forced residual (reading − pm) on dose lags 1–4 and on policy-expected dose lags 1–4. This measures what the model is missing in factual prediction.
- **Insulin:** the hypothesis is that the model absorbed the policy dose into glucose-level dynamics. Then the coefficients on the doses (a_k ≈ −1.7, −1.4, −0.4) will be roughly the negative of those on the policy lags (c_k ≈ +1.7, +1.4, …). In the equivalent deviation parametrization, the policy coefficients will be ≈ 0.
- **Refutation:** if instead the policy coefficients are ≈ 0 next to the dose coefficients, the model under-uses dose *level*, and a level-based channel would be right.
- **Norepinephrine:** I run the same check, with expected rate = rate − innovation.

```python
code_txt = r'''
def build_resid_tab(tf_, gg_):
    G=gg_.sort_values(['stay_id','h']).copy()
    gb=G.groupby('stay_id')
    G['nexp']=G['nor_l0']-G['e_nor']
    for k in range(1,6):
        G['pol_l%d'%k]=gb.ins_base.shift(k); G['ei_l%d'%k]=gb.e_ins.shift(k)
        G['en_l%d'%k]=gb.e_nor.shift(k); G['nexp_l%d'%k]=gb.nexp.shift(k)
    T=tf_[np.abs(tf_.t-np.round(tf_.t))<1e-6].copy(); T['h']=np.round(T.t).astype(int)
    M=T.merge(G,on=['stay_id','h'],how='left'); M['res']=M.value-M.pm
    return M
def ols_tab(M,ycol,xcols):
    d=M[[ycol]+xcols].dropna(); X=np.c_[np.ones(len(d)),d[xcols].values]; y=d[ycol].values
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b; s2=r@r/(len(y)-X.shape[1])
    se=np.sqrt(np.diag(np.linalg.inv(X.T@X))*s2)
    return pd.DataFrame({'b':b,'se':se},index=['const']+xcols).round(3), len(y), np.sqrt(s2)
'''
open('/work/resid.py','w').write(code_txt); exec(code_txt)
chk=gg[['ins_l0','ins_base','e_ins']].dropna(); print('e_ins check', np.abs(chk.ins_l0-chk.ins_base-chk.e_ins).max())
M2=build_resid_tab(tf2,gg)
L=[1,2,3,4]
for nm,cc in [('gf',6),('gl',10)]:
    Mi=M2[(M2.code==cc)&(M2.h>=1)]
    t1,n1,s1=ols_tab(Mi,'res',['ins_l%d'%k for k in L]+['pol_l%d'%k for k in L])
    t2,n2,s2=ols_tab(Mi,'res',['ei_l%d'%k for k in L]+['pol_l%d'%k for k in L])
    print(nm,'n',n1,'sd',round(s1,2)); print(pd.concat([t1.reset_index(),t2.reset_index()],axis=1).to_string())
for nm,cc in [('ma',7),('mc',5)]:
    Mi=M2[(M2.code==cc)&(M2.h>=1)]
    t1,n1,s1=ols_tab(Mi,'res',['nor_l%d'%k for k in L]+['nexp_l%d'%k for k in L])
    t2,n2,s2=ols_tab(Mi,'res',['nor_l%d'%k for k in L])
    print(nm,'n',n1,'sd',round(s1,2)); print(pd.concat([t1.reset_index(),t2.reset_index()],axis=1).to_string())
```