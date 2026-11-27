Type index 1 with gap input log(dt)/10 is clearly right, but the fit is not yet exact: RMS 7.3 mg/dL on glucose and 2.2 mmHg on MAP. The remaining candidates are the bin-edge tie rule, age units, the register column layout, the status index, and z sampling versus the prior mean. Below I vary these one or two at a time from that base. The aim is to find which change takes the error close to zero, or at least to the level expected from 4 random z draws.

```python
base = dict(typ=1,gap_scale=10.0)
tests = {'side_right':dict(side='right'), 'age_min':dict(age_div=60.0), 'age_h':dict(age_div=3600.0), 'status1':dict(status_idx=1),
         'lc0125':dict(lc=(0,1,2,5)), 'lc0127':dict(lc=(0,1,2,7)), 'lc1023':dict(lc=(1,0,2,3)), 'lc0213':dict(lc=(0,2,1,3)),'lc0132':dict(lc=(0,1,3,2)),
         'nz16':dict(nz=16)}
score(predict(FR,**base),'base')
for k,v in tests.items(): score(predict(FR,**base,**v),k)
```