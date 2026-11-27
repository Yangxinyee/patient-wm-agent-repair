Decodability from the head input `WC0`, which is better than from the hidden layer:

| Target (rows) | CV R² |
|---|---|
| Insulin surprise, lag 1 (glucose) | 0.32 |
| Insulin surprise, lag 2 (glucose) | 0.18 |
| Insulin surprise, lag 3 (glucose) | 0.06 |
| Insulin rate level (glucose) | 0.84 |
| NE shock, lag 1 (MAP) | 0.03 |
| NE rate level (MAP) | 0.76 |
| NE rate change (MAP) | 0.16 |

So the head input knows the dose level well but the dose surprise only noisily, and NE shocks hardly at all.

This suggests a specific cause of the weak rollout response. If the head already predicts optimally given its noisy encoding, then its response to a dose change is attenuated by that encoding's reliability. For insulin this would give roughly −2 × 0.32 ≈ −0.6 per u/h at lag 1, which matches the −0.45 to −0.7 seen. The factual fit would then be almost unaffected by the missing effect.

To test this and get edit directions, this run does four things:
1. Redo the residual regressions with policy controls (expected dose levels).
2. Check whether the residual is linearly predictable from `WC0` at all. If not, the head is already factually optimal and any edit is a pure trade-off.
3. Estimate the causal direction ∂WC/∂dose at lags 1 and 2 by regressing `WC0` on the exogenous surprises. Do the same for NE.
4. Compute the exact head Jacobian. That gives the model's own one-step response, Jacobian times direction, which can be compared with −0.71 per u/h (true minus missing).

If that number is near −0.7, the attenuation picture holds. I will then build decoders with unit response along these directions and scale them.

```python
import numpy as np
from scipy.special import erf as _erf
gm_=RD2.event.isin(['glucose_finger','glucose_lab']).values; mm_=~gm_
def ols2(y,X):
    X=np.column_stack([np.ones(len(X)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b
    cov=np.linalg.inv(X.T@X)*((r**2).sum()/(len(y)-X.shape[1])); return b,np.sqrt(np.diag(cov))
E_=['eps_ins_1','eps_ins_2','eps_ins_3']; B_=['ins_exp_1','ins_exp_2','ins_exp_3']
okg=gm_&RD2[E_+B_].notna().all(1).values
Xg=RD2.loc[okg,E_+B_].values; rg_=RD2.loc[okg,'res'].values
b,se=ols2(rg_,Xg); print('G res~eps+base',np.round(b[1:4],2),np.round(se[1:4],2),okg.sum())
# NE controls: lag levels
Xm=np.column_stack([Xn_[:,:3], RD2.loc[mm_,['ne_1','ne_2','ne_3']].fillna(0).values])
b,se=ols2(RD2.loc[mm_,'res'].values,Xm); print('M res~NEI+lev',np.round(b[1:],2),np.round(se[1:],2))
# Jacobian of mean wrt WC
pre=A0; phi=np.exp(-pre**2/2)/np.sqrt(2*np.pi); gp=0.5*(1+_erf(pre/np.sqrt(2)))+pre*phi
C17=np.column_stack([np.zeros(len(P0)),HI['cen']]); m0_=(P0*C17).sum(1)
J=(((P0*(C17-m0_[:,None]))@W2_)*gp)@W1_
# causal WC directions
Wg=WC0[okg]; Bg=np.linalg.lstsq(np.column_stack([np.ones(okg.sum()),Xg]),Wg,rcond=None)[0]
gam1,gam2,gam3=Bg[1],Bg[2],Bg[3]
Jg=J[okg]
print('model direct resp lag1/2/3 (mg/dL per u/h):',[(Jg@g).mean().round(2) for g in (gam1,gam2,gam3)])
Wm=WC0[mm_]; Bm=np.linalg.lstsq(np.column_stack([np.ones(mm_.sum()),Xm]),Wm,rcond=None)[0]
Jm=J[mm_]
print('model MAP resp NEI1/2/3, lev1:',[(Jm@Bm[k]).mean().round(2) for k in (1,2,3,4)])
print('|gam1| %.3f |gam2| %.3f |WCsd| %.3f'%(np.linalg.norm(gam1),np.linalg.norm(gam2),np.linalg.norm(Wg.std(0))))
# is residual linearly predictable from WC?
print('res~WC CV r2 G',cvr2_(WC0[gm_],RD2.res.values[gm_],RD2.stay_id.values[gm_]),'M',cvr2_(WC0[mm_],RD2.res.values[mm_],RD2.stay_id.values[mm_]))
```