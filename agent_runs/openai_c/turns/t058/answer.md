The architecture gives two distinct latent pathways: posterior `z` updates the recurrent state through `z_in`, while predictions use prior samples through `z_head`. I’ll recover the record-based effect estimates and earlier edit results before choosing which pathway to test. This should prevent treating a dose-correlated state direction as a proven causal response.

```python
import inspect
import pandas as pd

notes_text = Path("/work/notes.md").read_text()
print("EARLY NOTEBOOK\n", notes_text[:7200])

print("\nDATA FRAMES")
for key, value in sorted(list(globals().items())):
    if isinstance(value, pd.DataFrame):
        print(key, value.shape, list(value.columns)[:16])

print("\nANALYSIS HELPERS")
for key, value in sorted(list(globals().items())):
    if inspect.isfunction(value) and getattr(value, "__module__", "") == "__main__":
        try:
            print(key, inspect.signature(value))
        except Exception:
            pass

print("\nCAUSAL ESTIMATES")
for key in ["dynamic_moments", "dml_results", "baseline_rollouts", "dose_models",
            "persistence", "repair_signal", "repair_white", "head_surrogates"]:
    value = globals().get(key)
    if isinstance(value, dict):
        for subkey, item in value.items():
            if isinstance(item, dict):
                compact = {
                    k: v for k, v in item.items()
                    if np.isscalar(v) or isinstance(v, str)
                }
                print(key, subkey, "fields", list(item), "scalars", compact)
            elif np.isscalar(item):
                print(key, subkey, item)
            elif isinstance(item, (list, tuple)) and len(item) < 10:
                print(key, subkey, item)
            elif isinstance(item, np.ndarray):
                print(key, subkey, item.shape,
                      item.tolist() if item.size <= 12 else "")
            else:
                print(key, subkey, type(item).__name__)

print("\nLATENT EDIT TENSOR SHAPES")
for name in NAMES:
    if name.startswith(("z_in.", "z_head.", "encode_action.")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 38: causal and pathway audit\n"
            "- Separating posterior z_in state updates from prior z_head readouts.\n"
            "- Recovering identification estimates and prior ablations; no edits changed. Forward passes: 19.\n")
```