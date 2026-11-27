The saved z Jacobians are tiny. The mean absolute effect is 0.01 to 0.03 per unit z on next readings (largest z dim 0.07), so the existing z_in columns barely move predictions at drug bundles. Any z-based injection needs a new, much stronger token direction, and the pre-norm prelude may saturate it.

The helpers I need already exist:
- `step_local(k, ztok_add=...)` steps bundle k with an addition to the z token.
- `ctx_from` and `value_mean` turn the new state into a glucose prediction.
- `u_for(k, fI, fN)` gives edited drug encodings.

Next I'll check the return format of `step_local` and time one call, then build a prediction for one insulin bundle. That shows whether a sensitivity scan over all 192 token directions is affordable. If one step plus head takes about 10 ms or less, a full scan at around 30 bundles fits easily in one run.

```python
import time
ks_ins = [k for k in range(len(INFO)-1) if 'insulin' in INFO[k] and INFO[k+1]['sid']==INFO[k]['sid'] and 'glucose_finger' in INFO[k+1]]
print('ins bundles w/ next gf:', len(ks_ins))
k = ks_ins[5]; t0=time.time(); r = step_local(k); t1=time.time()
print(type(r), len(r) if isinstance(r,(tuple,list)) else '', [tuple(x.shape) for x in r] if isinstance(r,(tuple,list)) else getattr(r,'shape',None), 'sec %.4f'%(t1-t0))
print('INFO k,k+1:', INFO[k], INFO[k+1])
print('dts type', type(dts), getattr(dts,'shape',None), 'tmin', getattr(tmin,'shape',None))
```