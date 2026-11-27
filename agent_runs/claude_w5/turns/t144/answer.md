Only one always-on flag (column 0), with age as log(1+age in s)/10 in column 3 and z_last_value in column 6, gives the best fit so far: 1.4 mg/dL on finger glucose and 0.58 mmHg on arterial MAP. That is close but probably not exact.

Next I will look at where the remaining error sits:
- its quantiles;
- whether the reading has a previous value;
- the gap to the reading (5 min vs about 1 h);
- the hour of the stay.

If the error is spread evenly and small, it is the noise from the tool's 4 z draws and the head is effectively correct. If it is concentrated in one group, that group shows which detail is still wrong.

```python
a_,b_,m64 = eval_full([0],3,6,ages['s'],nsamp=64)
print('z64:', a_, b_)
a2,b2,m0 = eval_full([0],3,6,ages['s'])
err = (m64-pm_t).numpy(); R_['err']=err; R_['m64']=m64.numpy()
R_['dtb'] = pd.cut(R_.dt, [0,400,3400,3700,1e9], labels=['<=400s','~55m','~60m','>62m'])
for c in ['glucose_finger','map_arterial']:
    d = R_[R_.code==c]
    print(c, 'abs err q50/q90/q99', np.round(np.quantile(np.abs(d.err),[.5,.9,.99]),2), 'mean signed', round(d.err.mean(),3))
    print('  by hl', d.groupby('hl').err.apply(lambda x: (round(x.abs().mean(),2), len(x))).to_dict())
    print('  by dt', d.groupby('dtb', observed=True).err.apply(lambda x: (round(x.abs().mean(),2), len(x))).to_dict())
    print('  by hour<=1', d.groupby(d.t<=1).err.apply(lambda x: round(x.abs().mean(),2)).to_dict())
```