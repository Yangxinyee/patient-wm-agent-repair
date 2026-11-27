The kernel kept a lot from the earlier session, including functions named `true_shift` and `rows_truth` and impulse-response arrays (`IRs`, `IRL`, `IRA`, `IRC`, `KI`, `KN`). So I already built a ground-truth calculator before the restart. Rather than rewrite it, I'll print its source and the stored arrays and tables (`chk`, `chk2`, `RT`, `R`). Those tables likely hold my earlier true-vs-model comparison on the forward-pass-1 stays.

If they show a true insulin x2 shift far larger than the model's -2.75, the plan stands: strengthen the model's insulin effect. If the comparison is missing or the source can't be recovered, I'll recompute it next run.

```python
import inspect
for fn in ['true_shift','rows_truth','deconv','surprise_reg']:
    try:
        print(inspect.getsource(globals()[fn]))
    except Exception as e:
        print(fn, 'ERR', e)
np.set_printoptions(precision=3, suppress=True, linewidth=150)
for nm in ['KI','KN','IRs','IRL','IRA','IRC']:
    print(nm, np.asarray(globals()[nm]).shape, np.asarray(globals()[nm])[:14])
for nm in ['chk','chk2','RT','R']:
    print(nm); print(globals()[nm].head(12).to_string())
```