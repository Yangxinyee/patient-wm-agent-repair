`tr` is the training record (2.0 M rows). /work/sel.json holds both stay selections: sel1 (tuning stays used in forward 1) and sel2 (train stays used in forwards 2–3).

Now I'll test the second evaluator, `tf_eval(P, build_ds(tr, stays, 0))`, on 20 sel2 stays with the original, `PS` and `P_`.
- **Sound** if `PS` and `P_` give different NLLs and different states.
- **Leaky too** if they match; then I'll write a fresh evaluator from the architecture.

```python
sel2=sel['sel2']; sel1=sel['sel1']; print(len(sel1), len(sel2))
st=sel2[:20]
t0=time.time(); ds=build_ds(tr, st, 0); t1=time.time()
print('build_ds %.1fs'%(t1-t0), type(ds).__name__, (list(ds.keys())[:15] if isinstance(ds,dict) else len(ds)))
outs={}
for nm,PP in [('PO',PO),('PS',PS),('P_',P_)]:
    r=tf_eval(PP, ds, True); outs[nm]=r
    print(nm, type(r).__name__, len(r), 'nll %.5f'%float(r[0].mean()), [tuple(x.shape) for x in r if torch.is_tensor(x)][:5])
print('PS-P_ nll diff %.3g ; probs maxdiff %.3g'%(float((outs['PS'][0]-outs['P_'][0]).abs().max()), float((outs['PS'][1]-outs['P_'][1]).abs().max())))
```