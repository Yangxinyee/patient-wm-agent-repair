The gate is not the cause. On glucose_finger rows after an insulin bundle its term sits far in the "on" region (about +120 to +195) at every dose. After non-insulin bundles it is far "off" (about −260 to −450). The readout term also tracks dev1 with a similar slope (about 0.45–0.65) in every dose band up to 3 u/h. So small doses are not simply invisible on recorded data.

The rollout shows the x1/2 shortfall is worst in the first hours: x2/x1/2 ratio 3.5 at 0–2 hours, against 2.2 at 8+ hours. That points to the readout responding less to dose decreases than to increases, i.e. a curved response to dev1.

To test that, I will tabulate the readout term by bins of dev1 on the same rows. If the curve is flatter for negative deviations, an extra negative-side gain would fix x1/2 without touching x2. If the curve is straight, the asymmetry comes from the rollout dynamics instead.

```python
mk_=gf&A_.hi.values
T2=pd.DataFrame({'gd':gdif[:,mk_].mean(0),'dev1':A_.dev1.values[mk_],'r':prev_rate[mk_]})
T2['db']=pd.cut(T2.dev1,[-4,-1.5,-1,-.6,-.3,-.1,.1,.3,.6,1,1.5,3,8])
print(T2.groupby('db',observed=True).agg(n=('gd','size'),gd=('gd','mean'),r=('r','mean')).round(3).to_string())
x=T2.dev1.values; y=T2.gd.values
Xp=np.c_[np.ones_like(x),np.maximum(x,0),np.minimum(x,0)]
cf=np.linalg.lstsq(Xp,y,rcond=None)[0]; print('slope pos',cf[1].round(3),'neg',cf[2].round(3))
print('dev1 sd',x.std().round(3),'frac dev1<-0.5',(x<-.5).mean().round(3),'frac >1',(x>1).mean().round(3))
```