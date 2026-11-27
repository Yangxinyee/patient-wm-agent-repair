The available result still supports opposite drug effects: insulin lowers glucose and norepinephrine raises MAP after adjustment. No weights have changed. I’ll extract the saved rollout targets and decoder structures next. Those will let me test a small correction against both the causal estimates and the factual-NLL limit, rather than rely on the unadjusted associations.

```python
from pathlib import Path

text = Path("/work/notes.md").read_text()
start = text.find('  "norepinephrine": {', text.find("## Run 5"))
end = text.find("## Run 8")
print("=== REMAINING EARLY CAUSAL NOTES ===")
print(text[start:end][-4300:])

print("\n=== DML CACHE STRUCTURE ===")
for drug, cache in dml_cache.items():
    print(drug)
    for key, value in cache.items():
        if hasattr(value, "shape"):
            print(" ", key, tuple(value.shape))
        elif isinstance(value, dict):
            print(" ", key, "dict", list(value)[:8])
        else:
            print(" ", key, type(value).__name__)

print("\n=== SAVED CAUSAL SUMMARIES ===")
for drug, result in dml_results.items():
    print(drug)
    if isinstance(result, dict):
        for key, value in result.items():
            if key != "rows":
                print(" ", key, str(value)[:550])
    else:
        print(str(result)[:700])

print("\n=== REPAIR DECODERS ===")
print(str(repair_decoder_cv)[:1200])
for drug, fits in repair_decoder_fits.items():
    print(drug, type(fits).__name__,
          list(fits) if isinstance(fits, dict) else str(fits)[:180])

print("\n=== QUERY / REGISTER CACHE VARIABLES ===")
for key in sorted(list(globals())):
    if key.startswith(("rec_", "wr_")):
        value = globals()[key]
        if hasattr(value, "shape"):
            print(key, tuple(value.shape))
        elif isinstance(value, dict):
            print(key, "keys", list(value)[:8])

print("\n=== POSTERIOR / VALUE HEAD TENSORS ===")
for name in NAMES:
    if "z_post" in name or "attr_heads.value_bin" in name or "last_numeric" in name:
        print(name, SHAPES[name])
```
