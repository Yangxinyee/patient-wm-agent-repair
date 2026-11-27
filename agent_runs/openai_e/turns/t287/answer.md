The earlier nonlinear attempt used a different admission constraint and initially had misaligned features, so it does not settle whether this detector will work. The new fit protects initial admissions and has positive sensitivity to both norepinephrine edits. I’ll verify its scaling and downstream direction; inconsistent signs or appreciable admission leakage would rule out installing it.

```python
print("NEW DETECTOR CHECKS")
for key247 in ((100.0, .001), (100.0, .01), (10.0, .001)):
    coef247 = np.asarray(fits244[key247])
    p247 = X244 @ coef247
    print(key247, "factual RMS", np.sqrt(np.mean(p247**2)),
          "initial max", np.max(np.abs(p247[initial_mask244])),
          "coef norm", np.linalg.norm(coef247))
    for contrast247 in D244:
        d247 = D244[contrast247] @ (B244.T @ coef247)
        y247 = Y244[contrast247]
        print(" ", contrast247, "slope", d247 @ y247 / (y247 @ y247),
              "mean", d247.mean(), "RMS", np.sqrt(np.mean(d247**2)))
print("\nFUNCTION DEFINITIONS")
for name247, limit247 in (("prelude_hidden93", 1500),
                          ("core_update_local", 2100),
                          ("coda46", 850),
                          ("means_all122", 850)):
    print("\n", name247)
    show_function238(name247, limit247)

print("\nTRAINING STATE CONTRACTS")
for k247, v247 in list(globals().items()):
    if k247.endswith("242") and not k247.startswith("_"):
        if isinstance(v247, dict):
            print(k247, "dict", list(v247.keys())[:15])
        elif hasattr(v247, "shape"):
            print(k247, tuple(v247.shape))
        elif isinstance(v247, (str, int, float, list, tuple)):
            print(k247, repr(v247)[:150])

print("\nPRIOR NONLINEAR OUTCOMES")
for tag247 in ("Run 95:", "Run 96:", "Run 97:", "Run 98:", "Run 99:", "Run 100:", "Run 141:"):
    sec247 = next((s for s in sections246 if tag247 in s[:180]), "")
    print(sec247[:650] if sec247 else "Not found: " + tag247)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 247: verify full nonlinear detector and local update mapping\n"
            "Checking full-fit scaling, admission nulls, and NE/insulin contrast signs; "
            "recovering the forward-only prelude/coda helpers before choosing an output "
            "direction. No weights changed. Python runs: 247; forwards: 29.\n")
```
