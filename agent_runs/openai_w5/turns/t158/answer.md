The earlier record analyses suggest negative insulin and positive norepinephrine effects, while the original rollouts barely respond. Those estimates still depend on confounding and response-form assumptions. I’ll inspect the current channel and its saved states; clean dose signals with small ordinary-state changes would justify factual testing.

```python
import numpy as _r137_np
from pathlib import Path as _R137Path

_r137_g = dict(globals())
print("=== RELEVANT PERSISTENT OBJECTS ===")
_r137_terms = ("pair", "priv", "chan", "transport", "135", "134", "panel",
               "tune", "causal", "dynamic", "effect", "layout", "head", "read")
for _r137_k in sorted(_r137_g):
    if not _r137_k.startswith("_") and any(t in _r137_k.lower() for t in _r137_terms):
        _r137_v = _r137_g[_r137_k]
        print(_r137_k, type(_r137_v).__name__, getattr(_r137_v, "shape", ""))
print("=== HELPER FUNCTIONS ===")
print(", ".join(sorted(
    k for k, v in _r137_g.items()
    if not k.startswith("_") and callable(v) and getattr(v, "__module__", "") == "__main__"
)))
print("=== FURTHER CAUSAL EVIDENCE ===")
print(_R137Path("/work/notes.md").read_text()[3500:8000])

_r137_o = _r137_np.load("/work/fwd_15/states_original.npz")
_r137_e = _r137_np.load("/work/fwd_15/states_edited.npz")
print("=== SAVED STATE ARRAYS ===")
print({k: _r137_e[k].shape for k in _r137_e.files})
_r137_coords = [61, 21, 15, 66]
_r137_other = [j for j in range(192) if j not in _r137_coords]
for _r137_label, _r137_a in [("original", _r137_o["x"]), ("edited", _r137_e["x"])]:
    print(_r137_label, "private out quantiles:",
          _r137_np.round(_r137_np.quantile(_r137_a[:, _r137_coords],
                                         [0, .1, .5, .9, 1], axis=0), 5))
print("Ordinary out RMS difference:",
      float(_r137_np.sqrt(_r137_np.mean(
          (_r137_e["x"][:, _r137_other] - _r137_o["x"][:, _r137_other]) ** 2))))
print("Tuning stay IDs for paired factual validation:")
if "panel_tune" in _r137_g:
    _r137_pt = _r137_g["panel_tune"]
    if "stay_id" in _r137_pt.columns:
        print(_r137_pt["stay_id"].drop_duplicates().head(100).tolist())
    else:
        print(_r137_pt.index.get_level_values(0).unique()[:100].tolist())
```
