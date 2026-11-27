The value head is a two-layer network, so changing its final weights is possible, but we first need to know what dose information reaches it. I’ll inspect the remaining state tensors and saved predictions to reconstruct the head locally. Matching the forward tool’s predictions will validate that reconstruction; otherwise, edits based on it would be unreliable.

```python
print("=== REMAINING CORE AND HEAD SHAPES ===")
for name in NAMES:
    if name.startswith(("core.", "attr_heads.", "last_", "filter_", "pending_")):
        if not name.startswith(("core.prelude", "attr_heads.value", "last_numeric")):
            print(name, SHAPES[name])

print("\n=== NUMERIC VOCABULARY ===")
for entry in VOCAB["numeric"]:
    print(entry)
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("value centres:", VOCAB["value_bin_centres"])

print("\n=== PANEL AND CACHE STRUCTURE ===")
print("wide index:", wide.index.names, "columns:", list(wide.columns))
print("Xhist columns:", list(Xhist.columns))
print(wide.iloc[:5].to_string())
for drug in ("insulin", "norepinephrine"):
    cc = dml_cache[drug]
    print(drug, [(k, getattr(v, "shape", type(v).__name__)) for k, v in cc.items()])
    if "models" in cc:
        print("model types:", str(cc["models"])[:900])

print("\n=== SAVED FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
    for p in folder.glob("factual_*.csv"):
        df = __import__("pandas").read_csv(p)
        print(df.columns.tolist())
        print(df.head(5).to_string(index=False))
        print("rows", len(df))
print("state hours", states0["hour"][:16])
print("state IDs", np.unique(states0["stay_id"])[:5])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 8\nRecovered persistent analysis variables and exact tensor shapes. Value-bin head is Linear(192,192), activation, Linear(192,17). Action encoder is separate two-stage DeepSets MLP. Last_numeric has 8 inputs; exact reconstruction must be checked rather than assuming the architecture summary lists every feature.\n")
```
