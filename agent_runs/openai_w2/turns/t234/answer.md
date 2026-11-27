The architecture confirms that `core.b_in.weight` changes how normalized prelude features enter the recurrent state. The output was truncated before the independent-feature names, so I’ll retrieve those and check the old feature transformation. Exact agreement would justify combining the cohorts; a mismatch would require fixing the transformation first.

```python
print("CURRENT EDIT VERIFIED",
      edited(),
      np.array_equal(np.asarray(load("core.b_in.weight")), validated_B176))

print("\nRECENT RECONSTRUCTION VARIABLES")
for k in sorted(list(globals())):
    if k.startswith(("r197", "r198", "r199")):
        v = globals()[k]
        if hasattr(v, "shape"):
            print(k, tuple(v.shape))
        elif isinstance(v, (list, tuple, dict)):
            print(k, type(v).__name__, len(v))
        elif callable(v):
            print(k, str(inspect.signature(v)))
        else:
            print(k, str(v)[:120])

print("\nFACTUAL FEATURE TRANSFORMATION")
for k in ["F147", "Efull134", "prelude_E111", "Xcheck171"]:
    if k not in globals():
        continue
    a = np.asarray(globals()[k])
    print(k, a.shape, "mean/sd", float(a.mean()), float(a.std()))
    if a.shape == (4250, 4, 192):
        x = np.column_stack([
            np.ones(a.shape[0] * 4),
            (a.reshape(-1, 192) - mu147) / sd147
        ])
        print("  transformed versus X147: max/rms",
              float(np.max(np.abs(x - X147))),
              float(np.sqrt(np.mean((x - X147)**2))))

print("\nPROBE/FIT VARIABLES")
for k in ["key147", "key171", "keys171", "probe_keys171", "probe_masks171",
          "validated_gains151", "candidate_gains173", "C147", "penalty171",
          "Y147", "D147", "validated_W176"]:
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, {kk: (vv.shape if hasattr(vv, "shape") else str(vv)[:80])
                  for kk, vv in v.items()})
    elif hasattr(v, "shape"):
        a = np.asarray(v)
        print(k, a.shape, "mean/sd", float(a.mean()), float(a.std()))
    else:
        print(k, v)

print("\nNOTEBOOK: DECODER CONSTRUCTION")
notes201 = Path("/work/notes.md").read_text()
for heading in ["## Run 147", "## Runs 147", "## Runs 170", "## Run 170",
                "## Runs 188", "## Runs193"]:
    pos = notes201.find(heading)
    if pos >= 0:
        print(notes201[pos:pos+1400])
```
