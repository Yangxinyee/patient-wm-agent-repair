The kernel still has the replica tools: prelude, prelude2, step_full, step_parts, surprise_dir and readout, plus the drug-bundle targets yt/ot. Next I'll print their signatures, so I can reuse them to design a prelude-level injection rather than rebuild anything.

```python
import inspect
for nm in ['prelude','prelude2','step_full','step_parts','step_var','coda_only','surprise_dir','readout','true_shift','truth','truth_on','ffn','mha','mha_m','build_mem','fitstat','step_cv','build_edit','u_bundle','ctx_of','latest']:
    f = globals()[nm]
    try: print(nm, str(inspect.signature(f))[:150])
    except Exception as ex: print(nm, 'ERR', ex)
```