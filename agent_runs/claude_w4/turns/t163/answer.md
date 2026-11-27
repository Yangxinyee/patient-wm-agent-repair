Runs 123 to 139 built a CPU copy of a whole core step, using these functions:
- mha, ffn, age_emb, mha_m (multi-head attention over the memory M), build_mem;
- step_full, step_var, step_parts, coda_only;
- fitstat and step_cv (checking functions).

The results yt and ot, of size 2319×4×192 and 2319×192, are the true new slots and `out` at the insulin and norepinephrine bundles of the sel100 tuning stays. So run 139 compared ways of adding the age embedding to the memory cross-attention:
- adding age to both keys and values ("ageKV") reproduced the true post-step state almost exactly, with R² 0.998 for the slots and 0.994 for `out`;
- no age, or a LayerNorm on M, did much worse.

The step copy is therefore close to exact at drug bundles. It can predict how a weight edit in the prelude changes the state after a dose.

Before designing v2, I want two things:
- the source of the copy functions, if the kernel kept it, so I can reuse them correctly;
- the core.* tensor shapes, so I can see which weights sit between the prelude output E and the new state.

If the source cannot be recovered, I will rely on the tensor shapes and test the functions on known inputs.

```python
import inspect
for fn_name in ['step_full', 'step_var', 'step_cv', 'fitstat', 'build_mem', 'mha_m', 'prelude']:
    fn = globals().get(fn_name)
    try:
        src = inspect.getsource(fn)
        print(f'=== {fn_name} ({len(src)} chars)\n{src[:1500]}')
    except Exception as ex:
        print(f'=== {fn_name}: no source ({type(ex).__name__}); signature', inspect.signature(fn) if fn else None)
core = [(n, tuple(SHAPES[n])) for n in NAMES if n.startswith('core.')]
print(len(core), 'core tensors')
print('; '.join(f'{n}{s}' for n, s in core))
```