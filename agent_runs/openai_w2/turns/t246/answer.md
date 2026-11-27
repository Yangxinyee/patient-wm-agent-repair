The standardization check failed before saving, so the weights are unchanged. My assumed conversion was incorrect. I’ll inspect the cached feature design and existing conversion helpers rather than guess. Exact agreement with the saved design will justify installation; another mismatch will rule out that mapping.

```python
print("CURRENT WEIGHTS UNCHANGED",
      np.array_equal(load("core.b_in.weight").numpy(), fallback_B210)
      if "fallback_B210" in globals()
      else np.array_equal(load("core.b_in.weight").numpy(), candidate_B204))

print("\nOBJECTS ENDING IN 134 / 147")
for k, v in sorted(dict(globals()).items()):
    if k.endswith(("134", "147")):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif callable(v):
            print(k, "function", inspect.signature(v))
        elif isinstance(v, dict):
            print(k, "dict keys", list(v)[:6])

print("\nFUNCTION NAMES")
print(" ".join(sorted(
    k for k, v in dict(globals()).items()
    if inspect.isfunction(v) and not k.startswith("_")
)))

print("\nSAVED STANDARDIZATION")
print("mu147 first six", np.asarray(mu147)[:6])
print("sd147 first six / range", np.asarray(sd147)[:6],
      np.min(sd147), np.max(sd147))
print("X147 first six / intercept", np.asarray(X147)[0, :6],
      np.asarray(X147)[0, -1])

print("\nFEATURE-DESIGN CHECKS")
for source_name211 in ["F147", "Efull134"]:
    source211 = np.asarray(globals()[source_name211], dtype=np.float64).reshape(-1, 192)
    print(source_name211, source211.shape,
          "first six", source211[0, :6])
    if len(source211) == len(X147):
        for label211, trial211 in [
            ("raw", source211),
            ("divided_by_sd", source211 / sd147),
            ("centered_divided_by_sd", (source211 - mu147) / sd147),
            ("centered", source211 - mu147),
            ("multiplied_by_sd", source211 * sd147)
        ]:
            print(label211, "max / RMS difference",
                  np.max(np.abs(trial211 - X147[:, :192])),
                  np.sqrt(np.mean((trial211 - X147[:, :192])**2)))

print("\nFEATURE / CONVERSION NOTEBOOK SECTIONS")
for sec in notes_text.split("\n## "):
    heading = sec.splitlines()[0].lower()
    if any(s in heading for s in ["134", "147", "171"]):
        print(sec[:1100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run210–211: failed mapping, no installation\n"
            "Run210 stopped before save_edit because assumed X147 "
            "standardization of raw Efull134 failed (max error 9.20). "
            "Current edit remains candidate_B204. Recovering exact cached "
            "feature design/conversion rather than using an unvalidated map.\n")
```