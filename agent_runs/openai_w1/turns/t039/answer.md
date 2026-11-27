The tensor shapes allow a correction at the latent posterior, where context and actions meet, without changing the prediction heads directly. I’ll first check bundle construction and dose units. This approach is viable only if that input retains dose information that the later value-head features lost.

```python
print("=== NUMERIC NORMALIZATION ===")
for r32_item in VOCAB["numeric"]:
    print({k: v for k, v in r32_item.items() if k != "fine_bin_edges"})
print("Numeric note:", VOCAB["numeric_note"])
print("Event/atom kinds:", VOCAB["event_kinds"], VOCAB["atom_kinds"])

print("\n=== LATER RETAINED OBJECTS ===")
r32_seen = False
r32_lines = []
for r32_name, r32_obj in list(globals().items()):
    if r32_name == "head_test_predictions":
        r32_seen = True
    if not r32_seen or r32_name.startswith(("_", "r30_", "r31_", "r32_")):
        continue
    if callable(r32_obj):
        r32_desc = "callable"
    elif hasattr(r32_obj, "shape"):
        r32_desc = f"{type(r32_obj).__name__} {r32_obj.shape}"
    elif isinstance(r32_obj, dict):
        r32_desc = f"dict[{len(r32_obj)}] keys={list(r32_obj)[:8]}"
    elif isinstance(r32_obj, (list, tuple)):
        r32_desc = f"{type(r32_obj).__name__}[{len(r32_obj)}]"
    elif isinstance(r32_obj, (str, int, float, bool)):
        r32_desc = repr(r32_obj)[:80]
    else:
        continue
    r32_lines.append(f"{r32_name}: {r32_desc}")
print("\n".join(r32_lines)[:3700])

print("\n=== STATE EXPORT SAMPLE ===")
print("S keys:", list(S))
for r32_key in S:
    r32_arr = S[r32_key]
    if r32_key != "x":
        print(r32_key, np.asarray(r32_arr)[:30])
print("State vector first/last:", S["x"][5, :5], S["x"][5, -5:])

print("\n=== MATCHING RECORD SAMPLE ===")
r32_sid = int(S["stay_id"][5])
print(train[(train.stay_id == r32_sid) &
            (train.hour.between(3, 7))].to_string(index=False))

print("\n=== RATE PANEL SCALE ===")
print("hours:", hours)
for r32_drug in drugs:
    print(r32_drug, "shape", R[r32_drug].shape,
          "quantiles", np.nanquantile(R[r32_drug], [0, .5, .9, .99]),
          "first stay", R[r32_drug][0, :16])
print("\n=== EARLY CAUSAL NOTES ===")
r32_notes = Path("/work/notes.md").read_text()
r32_start = r32_notes.find("Run 3")
r32_end = r32_notes.find("## Run 7")
print(r32_notes[r32_start:r32_end][:2300])
```
