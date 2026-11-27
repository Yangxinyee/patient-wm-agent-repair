Run 90 found the bug. The next bundle comes 55 minutes after the insulin bundle (or 60), yet the computed dose hours differed in 100% of cases. The insulin bundle's time sits at hh:05, and with float32 hours, subtracting 5/60 lands just below the integer, so `floor` assigned the previous hour's dose. My "off 0" rows were therefore pointed at the previous dose, and run 88's R² of 0.43 at the following bundle was the correctly aligned one. I'll redo the table using integer minutes. The prediction: the surprise of dose t should be readable right after its own bundle, then fade.

```python
mins=np.round(hr3.astype(np.float64)*60).astype(int); T0=np.floor((mins-5)/60).astype(int)
def r2cv(rows,y,sl=slice(192,960),alpha=30):
    Xr=X[rows][:,sl]; pr=np.zeros(len(y))
    for a,b in kf.split(Xr): pr[b]=Ridge(alpha=alpha).fit(Xr[a],y[a]).predict(Xr[b])
    return round(1-np.mean((y-pr)**2)/np.var(y),2)
src=np.where(hi)[0]
for off in [0,1,2,3,4,6]:
    ok=[i for i in src if i+off<len(X) and sid3[i+off]==sid3[i]]; rows=np.array(ok)+off
    t=T0[ok]; s=sid3[ok]
    yr=np.array([RB[a][0][b] for a,b in zip(s,t)]); yb=np.array([RB[a][1][b] for a,b in zip(s,t)]); ye=yr-yb
    print('off',off,len(rows),'R2 rate/base/eps', r2cv(rows,yr), r2cv(rows,yb), r2cv(rows,ye), ' out-only eps', r2cv(rows,ye,slice(0,192)))
```