The recovered notes confirm that raw dose–reading associations were misleading, and record-based analyses suggest stronger drug effects than the original model predicts. The output was truncated before the key validation results. I’ll retrieve those compactly to decide whether the prelude edit improves on the previous candidate.

```python
print("=== FACTUAL FORWARD 31 ===")
factual265 = {}
for model in ("original", "edited"):
    p = Path(f"/work/fwd_31/factual_{model}.csv")
    if p.exists():
        d = pd.read_csv(p)
        factual265[model] = d
        cols = [c for c in d.columns if "nll" in c.lower()]
        print(model, len(d), "columns:", d.columns.tolist())
        print("NLL means:", d[cols].mean(numeric_only=True).to_dict())
        ec = next((c for c in ("event", "code", "event_code", "event_name") if c in d), None)
        if ec:
            print(d.groupby(ec)[cols].mean(numeric_only=True).to_string())
    else:
        print(model, "available:", [p.name for p in Path("/work/fwd_31").iterdir()])
if len(factual265) == 2:
    common = [c for c in factual265["original"] if "nll" in c.lower()
              and pd.api.types.is_numeric_dtype(factual265["original"][c])]
    print("EDITED MINUS ORIGINAL:",
          {c: float(factual265["edited"][c].mean()-factual265["original"][c].mean())
           for c in common})

print("\n=== CANDIDATE NOTES ===")
for run in (219, 236):
    ms = list(re.finditer(rf"(?m)^##[^\n]*\b{run}\b[^\n]*", notes264))
    for m in ms[-1:]:
        end = notes264.find("\n##", m.end())
        print(notes264[m.start():end if end >= 0 else None][:1400])

print("\n=== LATEST RECORD-TARGET NOTES ===")
lines265 = [line for line in notes264.splitlines()
            if any(k in line.lower() for k in
                   ("target", "causal", "persistence", "rd ", "dynamic identification"))
            and not line.startswith("##")]
for line in lines265[-16:]:
    print(line[:260])

print("\n=== SAVED EDIT CONTRACTS ===")
for filename in ("joint236.npz", "nonlinear262.npz", "ne_nonlinear250.npz",
                 "prelude_direction255.npz"):
    with np.load("/work/"+filename) as a:
        print(filename, [(k, a[k].shape) for k in a.files])
        for k in a.files:
            v = a[k]
            if v.size <= 8:
                print(" ", k, v.tolist())
print("CURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 265: compact validation and target recovery\n"
            "Run264 output truncated before factual31 and saved candidate contracts. "
            "Recovering these compactly; no weights changed.\n"
            "Python runs: 265; forwards: 32.\n")
```