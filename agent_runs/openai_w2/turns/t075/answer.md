Earlier tests show that simply amplifying drug embeddings does not reliably strengthen insulin responses. I’ll inspect the reconstructed prediction head and action-encoder functions next. The working hypothesis is that a targeted action-path edit, followed by factual recalibration, can succeed where broad scaling failed.

```python
print("--- RELEVANT EARLIER DIAGNOSTICS ---")
print(notes_text[23022:28287])
print("\n--- FUNCTION SIGNATURES ---")
for k, v in sorted(list(globals().items())):
    if not k.startswith("_") and callable(v) and hasattr(v, "__code__"):
        c = v.__code__
        print(k, c.co_varnames[:c.co_argcount])
print("\n--- PRELUDE AND VALUE-HEAD SHAPES ---")
for n in NAMES:
    if any(s in n for s in ("core.prelude.", "attr_heads.value", "last_numeric")):
        print(n, SHAPES[n])
print("\n--- FORWARD FILES ---")
for i in range(1, 13):
    p = f"/work/fwd_{i}"
    print(i, os.listdir(p) if os.path.exists(p) else "missing")
```
