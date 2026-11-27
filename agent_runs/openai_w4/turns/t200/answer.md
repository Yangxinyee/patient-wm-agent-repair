The recovered construction confirms that the current channel is insulin-only. The record-based estimates support a positive norepinephrine response, though their causal interpretation still depends on the recorded history capturing treatment selection. I’ll inspect the channel algebra next so the MAP repair can use a separate signal; substantial factual changes would argue against this approach.

```python
import inspect, numpy as np, torch
from pathlib import Path

print("CHANNEL CONSTRUCTION NOTES")
for marker175 in ["## Run 102", "## Run 104", "## Run 105", "## Run 110",
                  "## Run 114", "## Run 115", "## Run 116", "## Run 118",
                  "## Run 119", "## Run 160"]:
    p175 = notes174.find(marker175)
    if p175 >= 0:
        e175 = notes174.find("\n## ", p175 + 4)
        print(notes174[p175:e175 if e175 >= 0 else p175 + 1000][:1300])

def describe175(name, obj):
    if isinstance(obj, dict):
        print(name, "dict")
        for k, v in obj.items():
            if hasattr(v, "shape"):
                a = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
                print(" ", k, a.shape, "norm", round(float(np.linalg.norm(a)), 5))
            elif callable(v):
                print(" ", k, "callable")
            else:
                print(" ", k, str(v)[:130])
    elif callable(obj):
        print(name, inspect.signature(obj), obj.__code__.co_names)
    elif hasattr(obj, "shape"):
        print(name, tuple(obj.shape))
    else:
        print(name, str(obj)[:200])

for name175 in ["chosen_channel116", "reference_bank125", "repair_bank",
                "dml_residuals", "kernels137", "readout_estimates137",
                "sig104", "signature104", "lambda104",
                "capture_clean104", "capture_prelude102",
                "transition73", "replay_factual125", "probe_means95"]:
    if name175 in globals():
        describe175(name175, globals()[name175])

print("\nOTHER CONSTRUCTION NAMES")
names175 = sorted(k for k in list(globals())
                  if any(str(n) in k for n in range(100, 117)))
print(" ".join(names175))
print("\nCURRENT CHANNEL WEIGHT SHAPES")
for name175 in ["z_post.0.weight", "z_post.0.bias", "z_post.2.weight",
                "z_post.2.bias", "z_in.weight", "z_in.bias",
                "core.prelude.0.attn.q.weight", "core.prelude.0.attn.kv.weight",
                "core.prelude.0.attn.out.weight", "core.b_in.weight"]:
    print(name175, SHAPES[name175])

with Path("/work/notes.md").open("a") as f175:
    f175.write("\n\n## Run 175\n"
               "Inspected retained channel algebra and reference-bank interfaces. The current "
               "channel is insulin-only; a separate NE signal is being considered within the "
               "same reserved head. No weights changed; statistical identification assumptions "
               "are not replaced by mechanism diagnostics.\n")
```