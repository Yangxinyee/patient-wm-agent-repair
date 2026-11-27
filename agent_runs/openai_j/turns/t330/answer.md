The records-derived kernel estimate is about −10.5/+5.3 for insulin doubling/halving and +3.8/−1.9 for norepinephrine on the first 100 stays, under its identification assumptions. Insulin is already close; norepinephrine doubling remains weak. I’ll check the recent validation results and safer calibration variants before changing weights. A lower-cost variant is useful only if its rollouts retain those gains.

```python
import dis

print("=== RECENT FORWARD OUTPUTS ===")
for k284 in range(35, 40):
    directory284 = Path(f"/work/fwd_{k284}")
    print(directory284.name, [p.name for p in directory284.iterdir()] if directory284.exists() else [])
    for p284 in sorted(directory284.glob("*.csv")):
        t284 = pd.read_csv(p284)
        print(p284.name, "shape", t284.shape, "columns", list(t284.columns))
        if "dose_edit" in p284.name:
            print(t284.head(4).to_string(index=False))
        elif "factual" in p284.name:
            nllcols284 = [c for c in t284.columns if "nll" in c.lower()]
            print("mean NLL fields", {c: float(t284[c].mean()) for c in nllcols284})

print("=== RECENT EXTRAPOLATION NOTES ===")
pos284 = notes282.find("## Run 273")
print(notes282[pos284:pos284+5400])

print("=== COMPACT BUILDERS ===")
print("make242 constants", make242.__code__.co_consts)
dis.dis(make242)
print("head177_make constants", head177_make.__code__.co_consts)
print("calibration norms", {k: float(np.linalg.norm(
    v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)))
    for k, v in r265_calibrators.items() if v is not None})
print("=== SOFT CALIBRATION FRONTIER ===")
t284 = pd.read_csv("/work/soft_hidden_frontier264.csv")
print(t284.shape, list(t284.columns))
print(t284.head(8).to_string(index=False))

print("=== AVAILABLE VALIDATION HELPERS ===")
for n284, o284 in list(globals().items()):
    if callable(o284) and any(tag284 in n284.lower()
        for tag284 in ["head177", "moment", "mixture", "cal170", "264", "265"]):
        print(n284, o284.__code__.co_varnames[:o284.__code__.co_argcount])

with open("/work/notes.md", "a") as f284:
    f284.write("\n## Run 284: candidate and extrapolation audit\n"
               "- Kernel targets are assumption-dependent estimates, not observed intervention truth.\n"
               "- Current insulin rollout is close to its kernel estimate; NE doubling remains weak.\n"
               "- Auditing recent validation and lower-cost calibration variants; no weights changed.\n"
               "- Counts: 284 Python runs, 38 forward passes.\n")
```
