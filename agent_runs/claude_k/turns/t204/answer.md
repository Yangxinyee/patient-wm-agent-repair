The rebuild works: the z posterior mean is reproduced to a mean absolute error of 0.05, about half its SD. That is the same accuracy as before.

**Insulin.** A linear ridge decoder on ctx predicts the rate with cross-validated R² 0.71. On "on" hours its residual SD is 0.72, against 0.67 for the policy benchmark (rate − base(band(glucose))). So ctx carries nearly all the information that sets the expected dose. The implied decode error is about 0.27 units/h, which I can tolerate. A linear insulin innovation term in the writer is feasible.

**Norepinephrine.** R² is only 0.59 (RMSE 0.23). I cannot yet tell whether that is near the policy's own limit or far below it. The policy is a multiplicative random walk with about 0.3 log noise, so even perfect knowledge of the previous rate and MAP leaves sizeable unpredictable variance.

This run compares, for both drugs:
- a flexible policy benchmark (gradient boosting on the true policy inputs: previous rate, the policy MAP or glucose, previous on/off), and ctx combined with those features;
- how well the previous norepinephrine rate itself decodes from ctx.

If the norepinephrine benchmark R² is close to 0.59, a linear ctx decoder is enough for norepinephrine too. If the benchmark is much higher (around 0.9), the writer lacks precise access to the previous rate, and I need an auxiliary carried channel for it.

```python
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
def cvpred(X,y,model_fn):
    p=np.zeros_like(y)
    for tr_,te_ in gkf.split(X,y,SID):
        m=model_fn().fit(X[tr_],y[tr_]); p[te_]=m.predict(X[te_])
    return p
def r2(y,p): return 1-((y-p)**2).mean()/y.var()
prevcol=np.maximum(hcol-1,0)
Dne_prev=np.where(hcol>=1, ne_all[ri,prevcol], 0.0)
Dins_prev=np.where(hcol>=1, ins_all[ri,prevcol], 0.0)
mp_=mbest[ri,gcol]
print('nan frac g,map', np.isnan(g_).mean(), np.isnan(mp_).mean())
Fne=np.c_[Dne_prev, np.nan_to_num(mp_,nan=70), hcol]
Fin=np.c_[np.nan_to_num(g_,nan=150), Dins_prev, hcol]
hgb=lambda: HGB(max_iter=300,learning_rate=0.05)
pN_pol=cvpred(Fne,Dne,hgb); pI_pol=cvpred(Fin,Dins,hgb)
print('NE  policy-feature R2 %.3f rmse %.3f'%(r2(Dne,pN_pol), np.sqrt(((Dne-pN_pol)**2).mean())))
print('INS policy-feature R2 %.3f rmse %.3f'%(r2(Dins,pI_pol), np.sqrt(((Dins-pI_pol)**2).mean())))
_,_,pN_ctx=cvr2(CT,Dne,0.1); _,_,pI_ctx=cvr2(CT,Dins,0.1)
print('NE prev-rate decodable from ctx: R2 %.3f'%cvr2(CT,Dne_prev,0.1)[0], '; corr(policy pred, ctx pred) %.3f'%np.corrcoef(pN_pol,pN_ctx)[0,1])
print('INS: corr(policy pred, ctx pred) %.3f'%np.corrcoef(pI_pol,pI_ctx)[0,1])
# ctx + policy features
pN_both=cvpred(np.c_[Fne,pN_ctx],Dne,hgb); pI_both=cvpred(np.c_[Fin,pI_ctx],Dins,hgb)
print('NE ctx+pol R2 %.3f ; INS ctx+pol R2 %.3f'%(r2(Dne,pN_both), r2(Dins,pI_both)))
# gap between ctx linear and policy, by drug
print('NE resid sd: ctx %.3f pol %.3f | INS resid sd: ctx %.3f pol %.3f'%(np.std(Dne-pN_ctx),np.std(Dne-pN_pol),np.std(Dins-pI_ctx),np.std(Dins-pI_pol)))
print('NE frac at cap %.3f, frac zero %.3f, frac start (prev0,now>0) %.3f'%((Dne>=0.999).mean(),(Dne==0).mean(),((Dne_prev==0)&(Dne>0)).mean()))
```