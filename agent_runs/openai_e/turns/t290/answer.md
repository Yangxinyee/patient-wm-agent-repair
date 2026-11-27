The inspection stopped at a closed archive; no weights changed. The detector’s verified scaling remains valid. I’ll skip closed archives and recover the response-calibration inputs. This is a bookkeeping fix, not evidence for or against the edit; the decisive test remains improved rollouts within the factual-NLL limit.

```python
print("OPEN STATE ARCHIVES")
for k250, v250 in list(globals().items()):
    if k250.startswith("_"):
        continue
    if isinstance(v250, np.lib.npyio.NpzFile):
        if getattr(v250, "zip", None) is not None and "x" in v250.files:
            print(k250, list(v250.files)[:10], v250["x"].shape)
    elif isinstance(v250, dict) and "x" in v250:
        x250 = v250["x"]
        if hasattr(x250, "shape") and len(x250.shape) == 2:
            print(k250, list(v250.keys())[:10], x250.shape)

print("\nCORE ARRAYS")
for k250, v250 in list(globals().items()):
    if k250.startswith("core_") and hasattr(v250, "shape"):
        print(k250, v250.shape)
print("core_memories",
      {k: [tuple(a.shape) for a in v] for k, v in core_memories.items()})
print("\nTRAINING ARRAYS")
for k250, v250 in list(globals().items()):
    if any(k250.endswith(str(i)) for i in (189, 190, 191)) and not k250.startswith("_"):
        if hasattr(v250, "shape"):
            print(k250, v250.shape)
        elif isinstance(v250, dict):
            print(k250, "dict", list(v250.keys())[:8])
        elif isinstance(v250, (tuple, str)):
            print(k250, repr(v250)[:130])
print("\nREADOUT DEFAULTS")
print("src45", src45.shape, "dt45", dt45.shape)
for fn250 in (readout121, readout125, next_means45):
    print(fn250.__name__,
          [(type(a).__name__, getattr(a, "shape", None))
           for a in (fn250.__defaults__ or ())])
print("W0 b_in versus installed RMS",
      np.sqrt(np.mean((as_numpy194(W0["core.b_in.weight"]) -
                       np.asarray(load("core.b_in.weight")))**2)))

print("\nEXACT RECONSTRUCTION NOTES")
for tag250 in ("Run 133:", "Run 134:", "Run 135:"):
    sec250 = next((s for s in sections246 if tag250 in s[:180]), "")
    print(sec250[:700] if sec250 else "Not found: " + tag250)

np.savez("/work/ne_nonlinear250.npz",
         raw_coefficient=np.asarray(fits244[(100.0, .001)]),
         scale=scale244)
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 249-250: response-context recovery\n"
            "Run249 stopped on a closed npz during inspection; no edits changed. "
            "Run250 skips closed archives and saves the correctly scaled nonlinear "
            "detector. Raw-coefficient factual RMS=.02323, exact admission nulls, "
            "positive NE2/half sensitivity. Recovering forward-only response contexts. "
            "Python runs: 250; forwards: 29.\n")
```
