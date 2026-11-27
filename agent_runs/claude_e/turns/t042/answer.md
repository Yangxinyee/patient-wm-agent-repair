The architecture document is back in context, and the kernel still holds my partial re-implementation (`atom_vec`, `set_encode`, `split_atoms`, `zpost_check2`, `act_fn` and others). The best match against the tool's states was 0.257, so something is still mis-specified. Before deciding between finishing the exact re-implementation and a more direct edit built from state regressions, I need two things. First, the tensor list. Second, whether I can recover the source of the functions I wrote. If `getsource` works, I can find the mismatch quickly. If not, I will write a compact re-implementation from the tensor list.

```python
import inspect, json
print({k:v for k,v in VOCAB['config'].items()} if 'config' in VOCAB else VOCAB.keys())
print("CFG",CFG); print("cfg",cfg); print("code_unit",code_unit); print("code_id",code_id)
print("num",{k:(v if not hasattr(v,'shape') else v.shape) for k,v in num.items()})
print("TENSORS:")
print("; ".join(f"{n}{tuple(SHAPES[n])}" for n in NAMES))
for fn in ['atom_vec','set_encode','split_atoms','zpost_check2','act_fn','stay_bundles','mlp','ln','fine_bin','zsc']:
    try:
        src = inspect.getsource(globals()[fn]); print(f"--- {fn} ---"); print(src[:1500])
    except Exception as ex:
        print(fn, "no source:", ex)
```