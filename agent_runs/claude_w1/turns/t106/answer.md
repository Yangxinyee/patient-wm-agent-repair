The columns are now clear. The string is the reading code (2 values), the int is the stay id (1,500 training stays), and the floats are hour, previous value, current value and the drug dose at lags 1–3 under randomly perturbed doses. The 192-vector is the head input. Because the perturbation decouples doses from readings, this set directly tests how strongly the head input encodes past doses.

I'll fit a cross-validated ridge decoder from the vector to each dose lag, separately per code. A high held-out R² (above ~0.8) at lag 1 means a head-level tilt driven by a linear dose readout is feasible. A low R² means the dose signal must first be strengthened upstream.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict as cvp
codes = np.array(ids); print({c: int((codes==c).sum()) for c in set(ids)})
for c in sorted(set(ids)):
    m = (codes == c) & np.isfinite(F[:,1])
    Xc = V[m]; 
    for j, nm in [(3,'lag1'),(4,'lag2'),(5,'lag3'),(1,'prev'),(2,'cur')]:
        y = F[m, j]
        pr = cvp(RidgeCV(alphas=[0.1,1,10,100]), Xc, y, cv=5)
        r2 = 1 - np.var(y-pr)/np.var(y)
        print(c[:14], nm, 'R2 %.3f' % r2, 'sd %.2f' % y.std(), 'resid sd %.2f' % (y-pr).std())
```