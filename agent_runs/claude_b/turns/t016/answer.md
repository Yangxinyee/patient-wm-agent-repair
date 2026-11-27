The measurement structure is now clear.

- **Glucose.** Finger and lab readings measure the same underlying glucose without bias (mean difference −0.02, slope about 1). Finger noise has SD about 9–10 mg/dL, while lab noise is small, about 3. True glucose moves by about 5.6 mg/dL per hour (lab-to-lab change SD 7.3 at lag 1 and 9.2 at lag 2).
- **MAP.** The cuff reads about 3 mmHg higher than the arterial line and is noisier (SD about 6 versus 3). True MAP is very persistent, with hourly innovations of about 2 mmHg.

So a single finger reading is a noisy proxy for true glucose. That creates apparent mean reversion, which my estimates must not mistake for a drug effect.

Next I will estimate the next-hour insulin effect by regression adjustment, and check that the dose noise is not stay-specific. Clinician style shared within a stay could correlate with patient type and bias the adjustment. Concretely:

1. Measure the within-stay correlation of the multiplicative dose noise.
2. Regress the next finger change on dose, adjusting flexibly for the current reading and history, with standard errors clustered by stay.
3. Run the regression two ways: with tier dummies, so only within-tier noise identifies the effect, and without, so the threshold jumps also contribute.

Agreement with the discontinuity estimate (about −0.3 per u/h) would support a small true one-hour effect.

```python
def ols(y, X, cl=None, names=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    XtX = X.T@X; b = np.linalg.solve(XtX, X.T@y); e = y - X@b
    iX = np.linalg.inv(XtX)
    if cl is None:
        V = iX * (e@e)/(len(y)-X.shape[1])
    else:
        cl = pd.factorize(np.asarray(cl))[0]
        S = np.zeros((cl.max()+1, X.shape[1])); np.add.at(S, cl, X*e[:,None])
        V = iX @ (S.T@S) @ iX
    se = np.sqrt(np.diag(V))
    return b, se
def tier(g):
    return np.select([g<150,g<180,g<220,g<260,g<300],[0,1,2,3,4],5)
base = {0:0,1:1.05,2:2.1,3:3.13,4:4.19,5:6.2}
Q = P[P.glucose_finger.notna()].copy()
Q['tier'] = tier(Q.glucose_finger.values)
Q['stay'] = Q.index.get_level_values(0); Q['hr'] = Q.index.get_level_values(1)
Ig = Q[(Q.ins>0)&(Q.tier>0)].copy()
Ig['lr'] = np.log(Ig.ins / Ig.tier.map(base))
# within-stay correlation of dose noise
Ig['lr_prev'] = Ig.groupby('stay').lr.shift(1)
m = Ig.lr_prev.notna()
print("dose noise sd", Ig.lr.std().round(3), " lag-1 corr within stay", np.corrcoef(Ig.lr[m], Ig.lr_prev[m])[0,1].round(3))
sm_ = Ig.groupby('stay').lr.agg(['mean','size']); 
print("var of stay means vs expected iid:", sm_['mean'][sm_['size']>=5].var().round(4), (Ig.lr.var()/sm_['size'][sm_['size']>=5]).mean().round(4))
# regression adjustment: dG1 on dose, flexible in R_t
D = Q[Q.dG1.notna() & (Q.hr>=0)].copy()
gb = np.clip(D.glucose_finger.values, 60, 400)
knots = np.arange(60, 401, 10)
Xs = [np.ones(len(D))] + [np.maximum(gb-k,0) for k in knots[:-1]] + [gb]
X1 = np.column_stack(Xs + [D.ins.values])
b,se = ols(D.dG1.values, X1, D.stay.values)
print("no tier dummies: dose coef", b[-1].round(3), "+-", se[-1].round(3))
T = pd.get_dummies(D.tier).values[:,1:].astype(float)
X2 = np.column_stack(Xs + [T, D.ins.values])
b,se = ols(D.dG1.values, X2, D.stay.values)
print("with tier dummies: dose coef", b[-1].round(3), "+-", se[-1].round(3))
```