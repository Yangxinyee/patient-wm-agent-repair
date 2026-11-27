The candidate now supplies an approximately dose-linear signal without discarding the learned posterior. The remaining question is how to route that signal to glucose and MAP predictions. I’ll inspect the validated sensitivities and reconstruction helpers; they should identify a correction direction we can test with the forward tool.

```python
import inspect
import numpy as np
from pathlib import Path

print("=== LOCAL FUNCTIONS ===")
funcs69 = {}
for n in sorted(list(globals())):
    v = globals()[n]
    if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__":
        funcs69[n] = v
        print(n, inspect.signature(v))
print("\n=== KEY HELPER DEPENDENCIES ===")
for n in ["r51_core", "r59_core", "repaired_core_local", "rr35_scaled_action_encoding"]:
    fn = globals().get(n)
    if fn is not None:
        print(n, "locals:", fn.__code__.co_varnames,
              "\n dependencies:", fn.__code__.co_names)

print("\n=== RESPONSE CANDIDATES ===")
for n in ["D65_candidates", "D56_candidates", "post66_rate_models"]:
    d = globals().get(n)
    if isinstance(d, dict):
        for k, v in d.items():
            if hasattr(v, "coef_"):
                print(n, k, "coef", np.asarray(v.coef_).shape,
                      "norms", np.linalg.norm(np.asarray(v.coef_), axis=-1))
            elif hasattr(v, "shape"):
                a = np.asarray(v)
                print(n, k, a.shape, "column norms", np.linalg.norm(a, axis=0))
            else:
                print(n, k, type(v).__name__)
print("\n=== RECONSTRUCTION DATA ===")
for n in sorted(list(globals())):
    if n.startswith(("r51_", "r56_", "r59_", "r35_", "r23_")):
        v = globals()[n]
        if hasattr(v, "shape"):
            print(n, tuple(v.shape))
        elif isinstance(v, dict):
            print(n, "keys", list(v)[:12])

print("\n=== LATER IDENTIFICATION NOTES ===")
notes = Path("/work/notes.md").read_text()
for para in notes.split("\n"):
    if any(s in para.lower() for s in
           ["aggregate", "target shift", "true shift", "linear dose", "rollout target",
            "kernel", "impulse", "bootstrap", "nonlinearity", "half-life"]):
        print(para[:1100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 69: response-route recovery\n"
            "- Explicit posterior carrier is available; inspecting validated full-path and prior-head sensitivities.\n"
            "- Next step is a small centered candidate and an external factual/rollout test, not further unconstrained fitting.\n"
            "- Python runs: 69; forward passes: 4; no submitted edits.\n")
```