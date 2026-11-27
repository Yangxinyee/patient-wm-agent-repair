The exact rollout improves insulin doubling to −9.507, but halving gives only +2.836, farther from the record-derived target of about +5.35. The gate may attenuate the correction when doses are reduced. I’ll preserve this candidate and inspect the gate construction; a dose-invariant activity gate could improve halving, while evidence that it already behaves invariantly would refute that explanation.

```python
import numpy as np, torch, inspect
validated_gated8_checkpoint = {n: load(n).clone() for n in edited()}
with open("/work/notes.md", "a") as f:
    f.write("""
## Activity-gated scale8 GPU results; Python 240, forwards 29/50
- Hash fd0233cc27 preserved as validated_gated8_checkpoint.
- First100 factual delta +0.00487; second100 +0.00431; reading-weighted first200 about +0.00459.
- Exact first100 rollout: insulin x2 -9.507, half +2.836; NE x2 -0.502, half -0.608.
- Main readout improves versus stronger6, but halving worsens. Still pending last200 factual safety.
- Investigating whether activity gating suppresses reduced-dose responses; no weights changed.
""")
print("=== CURRENT GATE CONFIGURATION ===")
for k,v in activity_gate_config.items():
    if hasattr(v, "shape"):
        a = np.asarray(v)
        print(k, a.shape, "norm", float(np.linalg.norm(a)),
              "range", (float(a.min()),float(a.max())))
    else:
        print(k, v)
print("\n=== GRAFT HELPER ===")
print(inspect.signature(make_selective_contrast_graft))
print("variables:", make_selective_contrast_graft.__code__.co_varnames)
print("globals:", make_selective_contrast_graft.__code__.co_names)
try:
    print(inspect.getsource(make_selective_contrast_graft))
except (OSError, TypeError):
    print("source unavailable; constants:", repr(make_selective_contrast_graft.__code__.co_consts)[:1600])
print("\n=== GATE/CACHE VARIABLES ===")
for k,v in sorted(dict(globals()).items()):
    if not k.startswith("_") and not callable(v) and any(s in k.lower() for s in
        ["gated", "finger_gate", "separator", "selective", "synth", "contrast", "activity", "sc_"]):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:10])
        elif isinstance(v, (int,float,bool)):
            print(k,v)
print("\n=== EARLY NOREPINEPHRINE EVIDENCE ===")
ne_sections = [b for b in notes_text.split("\n## ")
               if any(s in b.lower() for s in
                      ["norepinephrine identification", "ne identification", "ne policy",
                       "ne sensor", "instrument validity", "ne effect"])]
for b in ne_sections[:6]:
    print(b[:850])
```
