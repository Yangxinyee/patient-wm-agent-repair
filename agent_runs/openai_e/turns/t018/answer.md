The posterior receives both dose information and patient context, making it a promising place to expose dose deviations without changing ordinary predictions much. The previous output was truncated, so I’ll retrieve only the relevant layer shapes and vocabulary. This approach becomes viable if we can reconstruct the posterior inputs and recover dose deviations on held-out stays.

```python
import json
from pathlib import Path

print("=== ACTION, POSTERIOR, AND RECURRENT LAYERS ===")
prefixes = (
    "encode_action.", "core.z_", "core.prelude", "core.blocks.",
    "core.coda", "core.c_out", "core.b_in", "core.a_in",
    "core.delta_in", "core.flow", "core.out_norm", "core.gap",
    "attr_heads.value_bin", "last_numeric", "query_norm"
)
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
for key, val in VOCAB.items():
    if isinstance(val, dict):
        print(key, "keys:", list(val)[:20])
    elif isinstance(val, list):
        print(key, "list length:", len(val), "first:", repr(val[:2])[:300])
    else:
        print(key, repr(val)[:500])

print("\n=== SMALL VOCAB TABLES ===")
for key in ("config", "codes", "categorical", "categoricals", "normalisation",
            "normalization", "numeric", "units", "event_kinds", "atom_kinds"):
    if key in VOCAB:
        print(key, json.dumps(VOCAB[key])[:2200])
print("NUM_SPEC:", repr(NUM_SPEC)[:2200])

print("\n=== STATE AND HEAD METADATA ===")
for key, val in S0.items():
    print(key, getattr(val, "shape", None), repr(val[:3])[:150] if hasattr(val, "__getitem__") else "")
print(head_meta.head(3).to_string(index=False))
print("head metadata columns:", list(head_meta.columns))
print("Remaining retained names:",
      ", ".join(k for k in sorted(globals()) if not k.startswith("_"))[-4000:])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 15: posterior-edit feasibility\n"
        "Posterior receives context and action jointly. Candidate repair: recover "
        "dose-policy innovations in its hidden features and route them through a "
        "low-rank posterior-mean edit, with non-action updates constrained unchanged. "
        "Need exact input reconstruction and held-out innovation recovery first.\n"
        "Python runs: 15. Forward requests: 3.\n"
    )
```