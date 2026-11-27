Numeric gain alone remains the better tested fallback. Its response saturates, so increasing that gain further is unlikely to close the gap. I’ll inspect the remaining targeted-head experiments and saved state directions, looking for a candidate with a small factual footprint that has not already failed a forward test.

```python
print("=== LATER EARLY EXPERIMENTS ===")
notes = Path("/work/notes.md").read_text()
a = notes.find("## Forwards 11")
b = notes.find("## Python run 35")
part = notes[a:b] if a >= 0 and b > a else ""
for line in part.splitlines():
    print(line[:420] + (" ..." if len(line) > 420 else ""))

print("\n=== FUNCTION SIGNATURES ===")
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

print("\n=== DATA OBJECT NAMES ===")
items = []
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"notes", "part", "v"}:
        continue
    if isinstance(v, (np.ndarray, pd.DataFrame)):
        items.append(f"{k}:{v.shape}")
    elif isinstance(v, dict) and k not in {"VOCAB", "SHAPES", "__builtins__"}:
        items.append(f"{k}:dict({len(v)})")
print("\n".join(items))

print("\n=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in ["attr_heads.value", "query_norm", "last_numeric",
                                "embed_transition", "core.c_out", "core.out_norm",
                                "encode_action", "core.prelude.self_attn"]):
        print(name, SHAPES[name])
```
