The fwd_1 summary gives the original model's readouts on 100 tuning stays: insulin x2 −0.53, insulin x1/2 +0.09, norepinephrine x2 −1.33, norepinephrine x1/2 −0.48. My record-based targets are −13.7, +6.9, +5.2 and −2.6. So the model almost ignores insulin and gives norepinephrine the wrong sign. The edit has to add nearly the whole insulin effect and reverse the norepinephrine one.

Next I will save this to the notebook. I will also look at two saved results tables, `res_tab` and `T_comb`, which probably hold my earlier comparison of ways to read the drug rates out of the prelude attention, so I don't redo that work.

```python
add="""- fwd_1 (orig, tuning sel1, 100 stays): ins x2 -0.53, half +0.094 (n 484 stay-h); ne x2 -1.331, half -0.482 (n 238).
  Targets (records-IRF, sel1): -13.71 / +6.86 / +5.17 / -2.58 -> edit must supply ~all of the insulin effect and flip norepi.
"""
open('/work/notes.md','a').write(add)
for nm in ['res_tab','T_comb','best','scs','cands','OPTS']:
    v=globals().get(nm)
    if isinstance(v,dict):
        print(nm, {str(k)[:30]: (np.round(np.asarray(x,float),3).tolist() if isinstance(x,(float,int,np.floating)) or (hasattr(x,'shape') and np.size(x)<8) else (str(x)[:80])) for k,x in v.items()})
```