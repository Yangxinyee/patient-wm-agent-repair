Run 87's printout was swamped by solver warnings. The one row that came through (α=30, h) still showed R² of about 0 at all lags when every bundle is pooled. I'll rerun with warnings suppressed and split the rows: action bundles only, versus the reading bundle that follows. That shows where in the step sequence the surprise information exists and how fast it disappears.

```python
import warnings; warnings.filterwarnings('ignore')
hi=z3['has_insulin'].astype(bool)
nxt=np.r_[hi[:-1]&(sid3[1:]==sid3[:-1]),False]  # row before is insulin -> next row index
idx_after=np.where(np.r_[False,nxt[:-1]])[0]
def cvr2(rows,sl,j,alpha):
    Xr=X[rows][:,sl]; y=Y[rows,j]; pr=np.zeros(len(rows))
    for a,b in kf.split(Xr): pr[b]=Ridge(alpha=alpha).fit(Xr[a],y[a]).predict(Xr[b])
    return 1-np.mean((y-pr)**2)/np.var(y)
for nm,rows in [('ins_bundles',np.where(hi)[0]),('next_bundle',idx_after)]:
    for alpha in [30,300]:
        print(nm,len(rows),alpha,[round(cvr2(rows,slice(192,960),j,alpha),2) for j in range(4)])
```