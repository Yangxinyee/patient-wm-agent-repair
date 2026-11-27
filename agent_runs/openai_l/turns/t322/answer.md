The architecture confirms that dose edits must influence the recurrent state and survive feedback during rollout. The output was truncated before showing the candidate objects, so it did not resolve the reconstruction discrepancy. I’ll inspect those objects and the later identification evidence with shorter output; no weights will change yet.

```python
print("=== CANDIDATE STRUCTURES ===")
for name in ["weighted281", "grid269", "arch263", "backup281", "T0",
             "GF", "RF", "GDI", "GDN", "RDI", "RDN", "feature_scale"]:
    if name not in globals():
        continue
    obj = globals()[name]
    if isinstance(obj, dict):
        print(name, "dict", len(obj), "first keys", list(obj)[:6])
    else:
        print(name, describe282(obj))

for name, key in [
    ("weighted281", (0.3, 0.25, 8.0)),
    ("weighted281", (0.25, 0.25, 8.0)),
    ("grid269", (0.25, 0.25)),
]:
    obj = globals().get(name)
    if isinstance(obj, dict) and key in obj:
        print(name, key, describe282(obj[key]))

print("=== RELEVANT LIVE NAMES ===")
names283 = sorted(n for n in list(globals())
                  if any(s in n.lower() for s in
                         ["281", "269", "263", "paired", "target", "truth", "contrast"]))
print(str(names283)[:1800])

print("=== LATER IDENTIFICATION NOTES ===")
start283 = notes282.find("## Run 9")
print(notes282[start283:start283 + 3800])

print("=== AVAILABLE SOURCE HISTORY ===")
for name in ["In", "_ih"]:
    obj = globals().get(name)
    if isinstance(obj, list):
        print(name, "length", len(obj))
        matches = [s for s in obj if isinstance(s, str)
                   and "Baseline solution error" in s]
        if matches:
            print(matches[-1][-4500:])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 283 / bounded recovery\n"
            "No weights changed. Previous output was truncated before candidate structures; "
            "using bounded diagnostics to recover rebalanced-head objects and later "
            "causal-identification evidence.\n")
```