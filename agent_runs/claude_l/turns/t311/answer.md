Run 283 found the weight dicts. `W0` is the original model. `W1` has `z_head` zeroed plus older edits to `z_in`/`z_post`, probably an earlier version of the extra-token injection. Several other `W*` dicts exist; `W0z` is probably the original with `z_head` zeroed, which is the right base. The crash came from a dict with tuple keys (`VD`) that my filter caught by mistake. It is harmless, and the builder trial never ran.

Next I will list which tensors each saved weight dict changes relative to the original, so I know which one is which. Then I will run the builder trial on the `W0z` base: offline readouts and teacher-forced NLL on 40 batch-1 stays, timed. I'll also evaluate the existing `Wt`, which is probably the last candidate built before the restart. One result would show the builder works and moves the readouts toward my estimated effects:
- insulin ×2 well below the original's −3.3, toward about −14;
- insulin ×½ toward about +7;
- norepinephrine ×2 toward about +5, and ×½ toward about −2.5;
- with only a small NLL change.
Unchanged readouts or a large NLL increase would mean the gains or the injected directions need work.

```python
import time
def diffs(Wd): return [n for n in NAMES if not torch.equal(Wd[n].float(), load_original(n))]
for k in ['W0z','W1','W_cur','Wb','Wc','We','Wg','Wk','Ws0','Wt','Wx']:
    try: print(k, diffs(globals()[k]))
    except Exception as e: print(k,'err',e)
print('best', repr(best)[:300]); print('edited now', edited())
stays=list(BUN.keys())[:40]
Wt2=None
for args_ in [(1.0,1.0),(np.ones(1),np.ones(1))]:
    try:
        t=time.time(); Wt2=mkW3(*args_, W0z); print('mkW3 ok %.1fs'%(time.time()-t), args_); break
    except Exception as e: print('mkW3 fail', args_, repr(e)[:200])
if Wt2 is not None:
    print('changed vs W0z:', [(n, round(float((Wt2[n].float()-W0z[n].float()).norm()),3)) for n in NAMES if not torch.equal(Wt2[n],W0z[n])])
    for nm,Wv in [('Wt2',Wt2),('Wt_old',Wt)]:
        t=time.time(); r=evalW(Wv,stays); print(nm,'evalW %.1fs'%(time.time()-t)); print(repr(r)[:900])
    t=time.time(); r0=tfnll(W0z,stays); r1=tfnll(Wt2,stays); r2=tfnll(Wt,stays)
    print('tfnll %.1fs'%(time.time()-t), repr(r0)[:200], '||', repr(r1)[:200], '||', repr(r2)[:200])
```