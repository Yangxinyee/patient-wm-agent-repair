The retained state arrays and helper functions are available, so no additional forward pass is needed yet. I’ll check the helper’s feature order, activation, and state alignment. Correct alignment should make `out` agree with the normalized slot mean; correcting the head inputs should also reduce the systematic glucose bias.

```python
import dis, io, inspect
import numpy as np
import torch

print("=== RECONSTRUCTION HELPER ===")
try:
    print(inspect.getsource(reconstruct_head))
except (OSError, TypeError):
    _buf = io.StringIO()
    dis.dis(reconstruct_head, file=_buf)
    _text = _buf.getvalue()
    print(_text[:7400])
    print("DISASSEMBLY CHARACTERS:", len(_text))

print("\n=== SMALL HELPERS ===")
for _fn in (mlp_named, norm_named):
    print(_fn.__name__)
    print("constants:", _fn.__code__.co_consts)
    print("names:", _fn.__code__.co_names)

print("\n=== HEAD ROWS ===")
print(head_rows.columns.tolist())
print(head_rows.head(5).to_string(index=False))

print("\n=== NUMERIC SCALING ===")
for _key, _spec in numeric_specs.items():
    print(_key, {k: v for k, v in _spec.items() if k != "fine_bin_edges"})

print("\n=== STATE LAYOUT CHECK ===")
_hmean = torch.from_numpy(state_x[:, 192:960].reshape(-1, 4, 192).mean(axis=1))
_reout = norm_named(_hmean, "core.out_norm").numpy()
_delta = _reout - state_x[:, :192]
print("out vs normalized slot mean: RMS", float(np.sqrt(np.mean(_delta**2))),
      "max", float(np.max(np.abs(_delta))))
print("best settings:", best_head_settings)
print("reference summary:", head_reference.groupby("event").agg(
    n=("nll", "size"), nll=("nll", "mean"),
    predicted=("predicted_mean", "mean"), observed=("observed", "mean")
).round(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Helper recovery — Python 14; forward 4\n"
        "- Recovered retained head_rows, state_x, head_weights and reconstruct_head.\n"
        "- Checking feature construction and state layout before any weight change.\n"
    )
```
