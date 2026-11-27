statsmodels is not installed, so I will write a small least-squares helper with numpy. The question is unchanged:
1. Does the dose residual relate to a same-hour lab glucose, given the finger reading?
2. How does future glucose respond to the random part of the dose, given the current reading?

I will include base-step indicators and a smooth function of the current glucose as controls. Within a step, the residual is the only reason the dose varies.

```python
def ols(y, X, names=None):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X@beta
    s2 = r@r/(len(y)-X.shape[1])
    cov = s2*np.linalg.inv(X.T@X)
    se = np.sqrt(np.diag(cov))
    return beta, se

b = mi2[mi2.glucose_lab.notna()]
# residual vs (lab - finger) given finger
Xc = np.column_stack([b.glucose_finger, b.glucose_finger**2/1e3, b.glucose_lab])
be, se = ols(b.res.values, Xc)
print("res ~ finger, finger^2, lab: coef lab", be[3], "se", se[3], "n", len(b))
# 2. future finger glucose vs residual, controlling for G_h (finger) and base step dummies
def ctrl(d):
    G = d.glucose_finger.values
    cols = [G, G**2/1e3, G**3/1e6]
    for bb in [2,3,4,6]:
        cols.append((d.base.values==bb).astype(float))
    return np.column_stack(cols)
for k in range(1,7):
    d = mi2[mi2[f'Gf_{k}'].notna()]
    X = np.column_stack([d.res.values, ctrl(d)])
    be, se = ols(d[f'Gf_{k}'].values, X)
    # also with log rate scaled as dose: use insulin rate directly
    X2 = np.column_stack([d.insulin.values, ctrl(d)])
    be2, se2 = ols(d[f'Gf_{k}'].values, X2)
    print(f"k={k} n={len(d)}  coef(res)={be[1]:.2f}±{se[1]:.2f}   coef(rate U/h)={be2[1]:.2f}±{se2[1]:.2f}")
```