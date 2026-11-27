The architecture confirms a direct path from drug bundles into the latent state. Earlier record-based estimates suggest effects substantially different from the original rollouts, but those estimates still need auditing. The last output was truncated, so I’ll recover just the key objects and remaining findings. Then I can test a drug-specific correction without solving every observation-encoding detail.

```python
import inspect

_n29 = open("/work/notes.md").read()
_a29 = _n29.find("## Run 10")
_b29 = _n29.find("## Run 13")
print("=== RUNS 10–12 ===")
print(_n29[_a29:_b29] if _a29 >= 0 else "Section not found")

_g29 = list(globals().items())
print("=== FUNCTIONS ===")
for _k29, _v29 in sorted(_g29):
    if not _k29.startswith("_") and inspect.isfunction(_v29):
        try:
            print(_k29, str(inspect.signature(_v29)))
        except Exception:
            print(_k29)

print("=== TABLES AND LARGE ARRAYS ===")
for _k29, _v29 in sorted(_g29):
    if _k29.startswith("_"):
        continue
    if isinstance(_v29, pd.DataFrame):
        print(_k29, _v29.shape, list(_v29.columns))
    elif isinstance(_v29, (np.ndarray, torch.Tensor)) and _v29.ndim and _v29.shape[0] > 100:
        print(_k29, type(_v29).__name__, tuple(_v29.shape))

print("=== DICTIONARIES ===")
for _k29, _v29 in sorted(_g29):
    if not _k29.startswith("_") and isinstance(_v29, dict) and len(_v29) > 5:
        print(_k29, len(_v29), list(_v29)[:10])

with open("/work/notes.md", "a") as _f29:
    _f29.write(
        "\n## Run 29: compact recovery\n"
        "Architecture confirms the drug-specific action-to-posterior pathway. "
        "Earlier insulin RD and NE residual-dose estimates differ substantially "
        "from original rollouts, but remain provisional. Recovering retained "
        "statistical and encoding objects; no edits.\n"
        "Python runs: 29; forward passes: 3.\n"
    )
```