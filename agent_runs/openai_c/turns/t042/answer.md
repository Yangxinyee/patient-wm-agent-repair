The recovered notes show strong confounding in raw dose associations and a negative insulin effect around dose thresholds. The output cut off before the final causal estimates. I’ll retrieve those and the saved designs to determine whether near-zero norepinephrine response is a failure or an appropriate result.

```python
import re

nb = Path("/work/notes.md").read_text()
start = nb.find("## Run 4")
end = nb.find("## Run 7", start)
section = nb[start:end if end >= 0 else start + 6500]
print("=== IDENTIFICATION AND EARLY EDITS ===")
for line in section.splitlines():
    if len(line) > 2200:
        print(line[:800] + "\n[long line middle omitted]\n" + line[-1400:])
    else:
        print(line)

print("=== VARIABLE NAMES ===")
saved_names = [
    k for k in list(globals())
    if not k.startswith("_") and k not in {"In", "Out", "get_ipython", "exit", "quit"}
]
print(", ".join(saved_names))

print("=== SAVED CANDIDATES ===")
if "repair_candidates" in globals():
    for k, v in repair_candidates.items():
        print(k, type(v).__name__, "length", len(v))
print("=== DESIGN VARIABLES ===")
for k in ["pilot_basis", "pilot_design", "dose_estimators", "rd_results",
          "causal_results", "surrogates", "panel", "train", "tuning"]:
    if k in globals():
        v = globals()[k]
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys())[:15] if isinstance(v, dict) else "")
print("=== OPERATION IMPLEMENTATIONS ===")
import inspect
for k in ["replay_ops", "apply_shear", "make_shear", "fit_rd"]:
    if k in globals() and callable(globals()[k]):
        try:
            print(k, inspect.getsource(globals()[k])[:1800])
        except (OSError, TypeError):
            print(k, "source unavailable; arguments:", inspect.signature(globals()[k]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 27\n- Run 26 output truncated; retrieving causal evidence and saved variable names compactly.\n- Edits unchanged; forward passes: 15.\n")
```