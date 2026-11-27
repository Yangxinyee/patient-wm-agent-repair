All diagnostic edits are now removed. The output was truncated before the probe details, so it does not yet resolve the head mismatch. I’ll inspect those details in a smaller output; matching the probe means would confirm the register layout before any repair is installed.

```python
print("=== PROBE OUTPUT ===")
print(r87_probe.columns.tolist())
print(r87_probe.head(8).to_string(index=False))

print("\n=== RELEVANT TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in (
        "last_numeric", "last_status", "query_norm", "type_emb",
        "attr_heads.value_bin", "embed_transition.value_bin"
    )):
        print(n, SHAPES[n])

print("\n=== PROBE WEIGHT STRUCTURE ===")
for n in [
    "attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
    "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias"
]:
    a = r87_probe_weights[n].numpy()
    nz = np.argwhere(np.abs(a) > 1e-8)
    print(n, a.shape, "nonzero count:", len(nz))
    print([(tuple(ix), round(float(a[tuple(ix)]), 7)) for ix in nz[:20]])

print("\n=== VOCAB CONFIG ===")
print(VOCAB.get("config"))
print("Vocab keys:", list(VOCAB))

print("\n=== PERSISTENT VARIABLE NAMES ===")
print(" ".join(sorted(
    n for n in globals()
    if not n.startswith("_") and not n.startswith("r87")
)))

print("\n=== CAUSAL NOTEBOOK CONTINUATION ===")
p = r87_notes.find("## Run 4")
print(r87_notes[p:p + 3200])
```