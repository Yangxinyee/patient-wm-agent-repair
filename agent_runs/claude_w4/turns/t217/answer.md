Run 190 showed that the kernel still holds a large set of replica and edit-building functions, including `build_v3`, `fIu`, `fNu`, `gate_fit` and `prelude2`. The notebook stops at run 160, so the work in runs 161 to 189 was never written down. The last output (run 189) appears to be a check of how well a linear readout recovers the insulin and norepinephrine rate across dose variants. It was probably read from the input that a prelude FFN unit sees: it varies even at measurement-only bundles, where u_k is constant.

Before adding anything to the model I need to know what `build_v3` and the readout functions actually do. Their source text cannot be recovered, so I will inspect their code objects: argument names, local variables, referenced globals and constants. I will also list the kernel objects that were cut off. No forward pass is used.

What I expect to find:
- **If v2/v3 is a prelude-FFN injection:** `build_v3` should take gain parameters and unit indices and touch the `core.prelude.0` FFN tensors. In that case I can pick up from it.
- **If it is something else:** I will rebuild the v2 plan from the notebook.

```python
import inspect
def info(f, n=500):
    c = f.__code__
    try: sig = str(inspect.signature(f))
    except Exception: sig = '(?)'
    consts = [x for x in c.co_consts if not hasattr(x, 'co_code')]
    s = f"{f.__name__}{sig}\n  locals={c.co_varnames[c.co_argcount:c.co_nlocals]}\n  names={c.co_names}\n  consts={consts}"
    return s[:n]
print(info(build_v3, 3000)); print()
for nm in ['fIu','fNu','gate_fit','fitgate','fit_rate','rfit','prelude2','tok_with','tok_var','ffn_block','pred_readout','mindelta','persist','build_edit']:
    print(info(globals()[nm], 420)); print()
rest_arr, rest_oth = [], []
for k, v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType) or (callable(v) and hasattr(v, '__code__')): continue
    if k < 'add': continue
    if isinstance(v, np.ndarray): rest_arr.append(f"{k}{v.shape}")
    elif hasattr(v, 'shape'): rest_arr.append(f"{k}{tuple(v.shape)}*")
    elif isinstance(v, (int, float, str, bool)): rest_oth.append(f"{k}={str(v)[:25]}")
    elif isinstance(v, (list, tuple, dict)): rest_oth.append(f"{k}:{type(v).__name__}[{len(v)}]")
    else: rest_oth.append(f"{k}:{type(v).__name__}")
print("ARRAYS2:", ", ".join(sorted(rest_arr))[:2600])
print("OTHER2:", ", ".join(sorted(rest_oth))[:1800])
```