`value_mean` returns a (mean, probs) pair and expects ctx as (N, 192). So I'll pass `cv[None]` with numpy arrays for last value and age, and take element [0] of the result. Same test as before: how strongly do z-token additions of norm 1 to 30 move the next glucose and MAP predictions?

```python
def next_pred(k, ztok_add=None, codes=('glucose_finger','map_arterial')):
    c, z, h, o = step_local(k, ztok_add=ztok_add) if ztok_add is not None else step_local(k)
    dt1 = dts[k+1]; hf = h*torch.exp(-dt1*torch.exp(fa)); cv = ctx_from(o, hf, dt1).double().reshape(1,192)
    return np.array([float(value_mean(cv, [cd], np.array([LV[cd][k+1]]), np.array([(INFO[k+1]['t']-LT[cd][k+1])*60.0]))[0][0]) for cd in codes])
b = next_pred(k); print('base', b.round(2), 'actual', INFO[k+1].get('glucose_finger'), INFO[k+1].get('map_arterial'))
for s_ in [1,3,10,30]: print(s_, (next_pred(k, ztok_add=s_*dv) - b).round(3))
```