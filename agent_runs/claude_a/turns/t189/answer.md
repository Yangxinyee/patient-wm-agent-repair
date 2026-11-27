`tf_loop` failed on 'z' in `keep`. z is always stored, and the function looks up the other `keep` names among its internal results. I'll retry with keep = (aw, E, hf, hn) and run the same comparison.

```python
r=tf_loop(PO, ds, keep=('aw','E','hf','hn'))
st=r[2]; print('store keys', list(st.keys()), {k:tuple(v.shape) for k,v in st.items()})
pr=prelude(PO, st['hf'], ds['e'], ds['u'], st['z'])
print('E err %.3g (scale %.3g)'%(float((pr['E']-st['E']).abs().max()), float(st['E'].abs().mean())))
print('att err %.3g'%float((pr['att']-st['aw']).abs().max()))
```