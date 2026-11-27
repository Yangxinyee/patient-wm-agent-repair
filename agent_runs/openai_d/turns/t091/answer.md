The architecture confirms that drug doses reach reading predictions through the recurrent state, while the reading head also uses a last-reading register. The records show why association alone is misleading: insulin’s positive unadjusted association becomes negative after adjustment. I’ll recover the cached causal estimates and exact head helpers next. An edit remains viable only if its installed computation matches the local test and preserves factual NLL.

```python
import inspect
from pathlib import Path

print("=== CAUSAL AND ROLLOUT NOTE SECTIONS ===")
text81 = Path("/work/notes.md").read_text()
for marker, limit in [
    ("## Run 4\n", 2200),
    ("## Run 7", 1300),
    ("## Run 8", 1300),
]:
    pos = text81.find(marker)
    if pos >= 0:
        print(text81[pos:pos + limit])

print("\n=== FUNCTION NAMES ===")
func81 = {
    k: v for k, v in list(globals().items())
    if not k.startswith("_") and inspect.isfunction(v)
}
print(", ".join(sorted(func81)))

print("\n=== HEAD, EMBEDDING, AND LATENT TENSORS ===")
for name in NAMES:
    if (name.startswith(("attr_heads.value_bin", "z_prior", "z_head",
                         "query_norm", "last_numeric", "last_status", "gap"))
        or name in ("embed.code.weight", "type_emb.weight",
                    "embed.numeric.weight", "embed.numeric.bias",
                    "embed.bins.value.weight", "embed.bins.rate.weight",
                    "embed_transition.value_bin.weight", "core.out_norm.weight",
                    "core.out_norm.bias", "core.flow.a")):
        print(name, SHAPES[name])

print("\n=== CACHE NAMES ===")
cache81 = []
for name, obj in list(globals().items()):
    if name.startswith(("rep_", "dml", "causal", "kernel", "gate", "head")) and not callable(obj):
        shape = getattr(obj, "shape", None)
        suffix = str(tuple(shape)) if shape is not None else type(obj).__name__
        cache81.append(name + ":" + suffix)
print("; ".join(sorted(cache81)))

print("\n=== OTHER DATA DICTIONARIES ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, dict) and (
        name.startswith(("dml", "causal", "kernel", "V", "W", "data", "fit", "est"))
        or name in ("VOCAB", "states", "state_data")
    ):
        print(name, list(obj)[:16])

print("\n=== WORK FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
print("Current edits:", edited())
```