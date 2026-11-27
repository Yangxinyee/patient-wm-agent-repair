The records support immediate effects of roughly −2 glucose units per insulin-rate unit and +3 MAP units per norepinephrine-rate unit; the original rollout barely responds to either drug. That suggests an under-responsive treatment pathway, but it does not yet establish the 12-hour effects. I’ll recover the identification diagnostics and action reconstruction code. Agreement across independent estimates would strengthen that conclusion; disagreement would weaken it.

```python
import inspect
from pathlib import Path

print("=== RETAINED FUNCTIONS ===")
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, inspect.signature(v))

print("\n=== LATER RETAINED OBJECTS ===")
lines = []
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if isinstance(v, pd.DataFrame):
        lines.append(f"{k}: DF {v.shape} {list(v.columns)[:15]}")
    elif isinstance(v, dict) and k not in {"SHAPES", "VOCAB"}:
        lines.append(f"{k}: dict keys {list(v)[:12]}")
print("\n".join(lines)[2500:6000])

print("\n=== CAUSAL SUMMARY ===")
print(dml_results.to_string(index=False))
print("residual structure:", {
    k: {kk: (tuple(vv.shape) if hasattr(vv, "shape") else type(vv).__name__)
        for kk, vv in v.items()}
    for k, v in dml_residuals.items()
})
print("\n=== NOTEBOOK RUNS 5–8 ===")
nt = Path("/work/notes.md").read_text()
start = nt.find("## Run 5")
end = nt.find("## Run 9")
print(nt[start:end][:3000])

print("\n=== RECENT RECONSTRUCTION SOURCE ===")
history = globals().get("In", [])
if not history:
    try:
        history = get_ipython().history_manager.input_hist_raw
    except Exception:
        history = []
if history:
    print(str(history[-4])[:3500])
else:
    print("No notebook input history available.")
print("Current edits:", edited())
```
