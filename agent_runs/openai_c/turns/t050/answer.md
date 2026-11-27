The records support nonzero effects, but the current model responses remain much smaller than the estimated targets. I’ll now retrieve only the factual score and edit-design details. If the head pilot consumes factual accuracy without improving dose sensitivity, I’ll remove it and test a different action pathway.

```python
import inspect
from pathlib import Path

print("FACTUAL PASS 16")
for p in sorted(Path("/work/fwd_16").glob("*")):
    if p.suffix == ".csv":
        f16 = pd.read_csv(p)
        print(p.name, f16.shape, f16.columns.tolist())
        print({c: float(f16[c].mean()) for c in f16.columns if "nll" in c.lower()})
    elif p.suffix == ".json":
        print(p.name, p.read_text()[:1400])

print("\nDESIGN STRUCTURES")
def small_description(obj):
    if isinstance(obj, np.ndarray):
        return {"shape": obj.shape, "norm": float(np.linalg.norm(obj))}
    if isinstance(obj, pd.DataFrame):
        return {"shape": obj.shape, "columns": obj.columns.tolist()}
    if isinstance(obj, dict):
        return {str(k): small_description(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return str(obj)[:180]
    return str(obj)[:180]

for key in ["dose_directions", "head_surrogates", "head_ne_design",
            "repair_candidates", "numeric_direction", "probe_ids", "validation_ids"]:
    if key in globals():
        print(key, str(small_description(globals()[key]))[:2600])
print("\nHELPER SIGNATURES")
for key in ["replay_ops", "freeze_ops", "apply_lowrank_right_transform",
            "apply_lowrank_left_transform", "orthogonal_rank1_factors"]:
    if key in globals():
        fun = globals()[key]
        print(key, inspect.signature(fun), "locals", fun.__code__.co_varnames[:12])
print("\nNUMERIC VOCAB KEYS")
print(list(VOCAB.keys()))
print("\nRELEVANT TENSORS")
print({k: SHAPES[k] for k in NAMES
       if k.startswith(("encode_action", "embed.numeric", "attr_heads.value_bin"))})
print("\nSAVED FILES")
print([str(p) for p in Path("/work/vectors").glob("*")][:30])
print("\nOPS SUMMARY", str(ops())[:300])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 33\n"
            "- Compact inspection of factual pass 16 and head edit design; no edits changed.\n"
            "- Python runs: 33; forward passes: 17.\n")
```
