The retained objects include the successful posterior specification and the original model’s tensors. I’ll now inspect only the transition components and the action-embedding helper. This should let us reconstruct a complete local step; agreement with saved `h` and `out`, not just `z`, will be the validation criterion.

```python
import dis
import inspect
import io
import re

print("=== SUCCESSFUL POSTERIOR SPECIFICATION ===")
print(r96_best["spec"])
print("Best posterior RMSE:", float(np.sqrt(r96_best["mse"])))
print("VOCAB config:", VOCAB["config"])

print("\n=== LOCAL RECONSTRUCTION OBJECTS ===")
for r101_name, r101_obj in sorted(list(globals().items())):
    if not r101_name.startswith("rm_"):
        continue
    if isinstance(r101_obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(r101_name, tuple(r101_obj.shape))
    elif isinstance(r101_obj, dict):
        print(r101_name, "dict length", len(r101_obj))
    elif isinstance(r101_obj, (str, int, float, bool)):
        print(r101_name, repr(r101_obj)[:80])

print("\n=== TRANSITION TENSOR NAMES ===")
r101_transition_names = [
    n for n in NAMES
    if n.startswith(("core.prelude", "core.blocks", "core.coda", "core.c_out",
                     "core.b_in", "core.delta_in", "core.a_in", "core.slots",
                     "core.age"))
]
for r101_name in r101_transition_names:
    print(r101_name, SHAPES[r101_name])

print("\n=== ACTION HELPER GLOBAL REFERENCES ===")
print(rm_insulin_atoms.__code__.co_names)
try:
    r101_source = inspect.getsource(rm_insulin_atoms)
except Exception:
    r101_buffer = io.StringIO()
    dis.dis(rm_insulin_atoms, file=r101_buffer)
    r101_source = r101_buffer.getvalue()
print(r101_source[:3500])

print("\n=== HELPER REFERENCES ===")
for r101_fn in (rm_set, rm_mlp, rm_ln, r96_empty_world):
    print(r101_fn.__name__, r101_fn.__code__.co_names,
          [v for v in r101_fn.__code__.co_consts if isinstance(v, (str, int, float))])

with Path("/work/notes.md").open("a") as r101_f:
    r101_f.write(
        "\n## Transition reconstruction preparation; Python 101, forward 11/50\n"
        "- Retrieved successful posterior specification and transition tensor layout.\n"
        "- Full local-step validation is required before synthetic-dose regression.\n"
        "- No weight changes.\n"
    )
```
