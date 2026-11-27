The rebuild works. Variant "left bins, age in hours" reproduces `mA2` exactly (MAE 0.000) and the tool's predicted means with MAE 1.26, and the transition-bias rows match exactly. So I now have a transparent, verified `with_code` for all 5,521 readings:

with_code = q + code embedding + last_numeric(A2·[1, has, log1p(age_h)/10, z_last]) + last-value-bin embedding.

Next is the cost check for the paired-unit plan, which needs 4 hidden units of the value head. For each of the 192 units I'll estimate the factual NLL increase if the unit is repurposed. Its activation is replaced by its mean within (code, last bin), and that mean is absorbed into the code×last-bin transition table, which is a legitimate code-specific bias. I'll also check that my rebuilt per-reading NLL agrees with the tool's (about 1.520 overall).

The plan is viable if at least 4 units each cost well under 0.001 nats. If every unit is costly, I'll switch to a design that adds a rank-1 term to existing units instead of repurposing.

```python
WCA,TRA,lbA=build_wc('left','h')
obA=binof(M4.value.values.astype(float),ev_,'left')
preA=WCA@W1.T+b1; actA=gelu(preA); lgA_=actA@W2.T+b2+TRA
def lsm_(z): z=z-z.max(1,keepdims=True); return z-np.log(np.exp(z).sum(1,keepdims=True))
LPA=lsm_(lgA_); nllA=-LPA[np.arange(len(obA)),obA]
print('recon NLL %.4f tool %.4f  corr %.3f'%(nllA.mean(), M4.nll.mean(), np.corrcoef(nllA,M4.nll)[0,1]))
grpk=code_*17+lbA; ug_,ginv=np.unique(grpk,return_inverse=True)
cost=np.zeros(192)
for j in range(192):
    a=actA[:,j]; m=np.bincount(ginv,weights=a)/np.bincount(ginv)
    d=(m[ginv]-a)[:,None]*W2[:,j][None,:]
    cost[j]=(-lsm_(lgA_+d)[np.arange(len(obA)),obA]).mean()-nllA.mean()
o_=np.argsort(cost); print('cheapest units', [(int(j),round(cost[j],5)) for j in o_[:12]]); print('median cost', np.median(cost))
```