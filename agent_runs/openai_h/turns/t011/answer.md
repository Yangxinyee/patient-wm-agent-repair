The architecture shows that dose edits change numeric dose/rate inputs and their bin embeddings before the recurrent update. Removing only one route would leave dose sensitivity. I’ll inspect these weights and the vocabulary to construct a dose-invariant input that retains drug identity and timing. Exact zero rollout shifts, together with preserved factual NLL, would validate that construction.

```python
import json
print("=== RELEVANT TENSORS ===")
prefixes = ("embed.", "encode_action.", "core.z", "last_numeric.", "pending_")
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== VOCABULARY ===")
print("top-level keys:", list(VOCAB))
for key, value in VOCAB.items():
    if key in {"codes", "config"} or any(s in key.lower() for s in ("norm", "categor", "unit")):
        print(key, repr(value)[:4500])
for key in ("dose_bin_edges", "rate_bin_edges"):
    if key in VOCAB:
        print(key, repr(VOCAB[key])[:2000])

print("\n=== WORKING OBJECTS (compact) ===")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(value, "shape"):
        print(key, type(value).__name__, value.shape)
    elif isinstance(value, dict):
        print(key, "dict keys:", list(value.keys())[:12])
    elif callable(value):
        print(key, "callable")
    elif isinstance(value, (list, tuple)) and len(value) < 20:
        print(key, repr(value)[:180])
print("\nCurrent edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 8: identify dose-dependent input routes\n"
            "- Architecture routes dose/rate numeric values and bins through action encoding, posterior and recurrent state.\n"
            "- Plan: retain action identity/presence/timing, neutralize dose magnitudes, then test exact rollout invariance and factual NLL.\n"
            "- Inspecting relevant weights and vocabulary; no edits yet.\n"
            "- Python runs: 8. Forward passes: 2.\n")
```