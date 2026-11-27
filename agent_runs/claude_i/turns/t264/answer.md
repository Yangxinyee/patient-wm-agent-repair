`value_mean` wants the last-value and age arguments as lists, one entry per code, not scalars. I've now spent three runs on interface errors, so this time I'll print the `value_mean` source from head2.py (about 1k characters) along with the fixed test. If the response test still fails, I'll know the exact signature for the next run.

```python
s = open('/work/head2.py').read(); i = s.find('def value_mean'); print(s[i:i+1100])
try:
    def next_pred(k, ztok_add=None, codes=('glucose_finger','map_arterial')):
        c, z, h, o = step_local(k, ztok_add=ztok_add) if ztok_add is not None else step_local(k)
        dt1 = dts[k+1]; hf = h*torch.exp(-dt1*torch.exp(fa)); cv = ctx_from(o, hf, dt1).double()
        return np.array([float(np.asarray(value_mean(cv, [cd], [LV[cd][k+1]], [(INFO[k+1]['t']-LT[cd][k+1])*60.0])).ravel()[0]) for cd in codes])
    b = next_pred(k); print('base', b.round(2), 'actual', INFO[k+1].get('glucose_finger'), INFO[k+1].get('map_arterial'))
    for s_ in [1,3,10,30]: print(s_, (next_pred(k, ztok_add=s_*dv) - b).round(3))
except Exception as ex: print('ERR', repr(ex)[:300])
```