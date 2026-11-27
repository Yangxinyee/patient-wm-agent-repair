The candidate increases factual value NLL by 0.00190, within the limit and leaving about 0.00310 for a correction. I’ll now inspect the dose signal’s timing and the value-head features. A usable signal must track the preceding infusion, not merely correlate with patient severity.

```python
import inspect as _r138_inspect
from pathlib import Path as _R138Path

print("=== HEAD AND CHANNEL SETTINGS ===")
for _r138_name in ["head_spec", "head_layout_spec", "layout134", "private_head116",
                   "gain135", "headTrain0", "featuresTransport134",
                   "featuresOriginal134", "cacheTransport133", "baseline135"]:
    if _r138_name not in globals():
        continue
    _r138_obj = globals()[_r138_name]
    if isinstance(_r138_obj, dict):
        print(_r138_name, {
            k: (tuple(v.shape) if hasattr(v, "shape") else str(v)[:180])
            for k, v in _r138_obj.items()
        })
    else:
        print(_r138_name, _r138_obj)

print("=== SMALL CHANNEL CONSTANTS ===")
for _r138_name, _r138_obj in list(globals().items()):
    if _r138_name.startswith("_"):
        continue
    if any(t in _r138_name.lower() for t in ["125", "129", "135", "coord", "reserve"]):
        if isinstance(_r138_obj, (int, float, str, list, tuple)):
            print(_r138_name, repr(_r138_obj)[:250])
        elif hasattr(_r138_obj, "shape") and int(np.prod(_r138_obj.shape)) <= 16:
            print(_r138_name, _r138_obj)

print("=== FUNCTION DEFINITIONS ===")
for _r138_name in ["head_features_layout", "reading_cache", "rw125",
                   "channel_targets105", "mechanistic_shifts"]:
    _r138_fn = globals().get(_r138_name)
    if _r138_fn is None:
        continue
    print(_r138_name, _r138_inspect.signature(_r138_fn))
    try:
        print(_r138_inspect.getsource(_r138_fn)[:3500])
    except (OSError, TypeError):
        print("Source unavailable; referenced globals:", _r138_fn.__code__.co_names)
        print("Constants:", repr(_r138_fn.__code__.co_consts)[:900])

print("=== PANEL SCHEMA ===")
print(panel.index.names, panel.columns.tolist())
print(panel.head(3).to_string())
print("Saved state timing:", list(zip(
    _r137_e["stay_id"][:16].tolist(), _r137_e["hour"][:16].tolist(),
    _r137_e["has_insulin"][:16].tolist(), _r137_e["has_norepinephrine"][:16].tolist()
)))

with _R138Path("/work/notes.md").open("a") as _r138_f:
    _r138_f.write(
        "\n## Forward 15 / run 138: clean-channel factual validation\n"
        "- Clean-pair candidate value NLL on tuning A: 1.64123 versus original 1.63933; +0.00190.\n"
        "- Passes the +0.005 constraint on this sample; approximately 0.00310 budget remains.\n"
        "- Saved-state ordinary out RMS difference is 0.03842. Pair differences are small and require timing/rate auditing.\n"
        "- No causal value-head coupling installed yet; factual fit alone is not a causal validation.\n"
    )
```
