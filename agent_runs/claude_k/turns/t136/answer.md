I can't recover the code of runs 74–121, because the kernel keeps no input history. Its variables do show the shape of that work, though.

- **Forward passes 5–7** were states requests (D5/D6/D7, probably the 300 stays in new300.json).
- **Functions** in the kernel include `head_pred`, `head_means`, `probs_from`, `cov_tilt`, `drug_atoms`, `meas_atoms`, `err_route*` and `z_action_err`. So I was rebuilding the model's internal computation in numpy, including the posterior update with drug atoms.
- **The last check** compared that rebuild with tool outputs on 300 reading bundles (`errs[300]`) and found close agreement.

Next I'll re-read the architecture text, which is stored in the kernel as `arch`, since I need it to design the edit. I'll also read the summaries of forward passes 5–7 and see whether the source of my helper functions can be retrieved. If it can, I can reuse the numpy rebuild to check candidate edits offline and save forward passes. If not, I'll rebuild only the parts I need: the value head and the reading-state features.

```python
import inspect
print(len(arch)); print(arch)
for d in ['fwd_1','fwd_2','fwd_4','fwd_5','fwd_6','fwd_7']:
    print(d, open(f'/work/{d}/summary.json').read()[:500].replace('\n',' '))
for fn in [head_pred, head_means, probs_from, cov_tilt, nll_of, meas_atoms, drug_atoms, err_route4, setenc, ctx_from, zmean, LN, mlp_, eval_decoder, cv_probe]:
    try:
        src=inspect.getsource(fn); print('SRC OK', fn.__name__, len(src))
    except Exception as e:
        print(fn.__name__, inspect.signature(fn), 'nosrc', type(e).__name__)
```