On the two 300-stay train sets (about 14k readings each), ablating dims [4, 2, 67] costs +0.0007 and −0.0012, which is essentially free. On sel2 the same edit costs +0.0089. Either sel2 differs systematically from ordinary stays (it was chosen for the forward checks, possibly stays with drug exposure), or `tf_nll` has a stale cache that mishandles edited weights. I'll rebuild sel2 with the same general evaluator (`build_ds` + `tf_eval`) and score the original and ablated models. If `tf_eval` reproduces the +0.009 on sel2, those stays really are sensitive and I need to know why before reserving dims. If it gives about zero, `tf_nll` is unreliable for edits and I'll stop using it.

```python
ds_s2=build_ds(tr, S['sel2'], 0)
a=tf_eval(PO,ds_s2,False); b=tf_eval(Pb,ds_s2,False)
a0=a[0] if isinstance(a,(tuple,list)) else a; b_=b[0] if isinstance(b,(tuple,list)) else b
print('sel2 via tf_eval n=%d base %.5f d=%+.5f'%(len(a0), float(a0.mean()), float((b_-a0).mean())))
c=tf_eval(P1,ds_s2,False); c_=c[0] if isinstance(c,(tuple,list)) else c
print('P1 via tf_eval d=%+.5f'%float((c_-a0).mean()))
print('ages global?', 'ages' in globals(), getattr(globals().get('ages'),'shape',None))
```