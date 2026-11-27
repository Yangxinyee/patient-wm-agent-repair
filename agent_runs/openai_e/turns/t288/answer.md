The coefficient check used the wrong scaling convention, so its large leakage values are not a valid assessment of the detector. I’ll test the alternative raw-feature convention and inspect the local state-update mapping. Recovering small factual RMS and near-zero admission responses would confirm the convention; otherwise I’ll reject the fit.

```python
import dis
print("RAW-FEATURE COEFFICIENT CHECK")
for key248 in ((100.0, .001), (100.0, .01), (10.0, .001)):
    rawcoef248 = np.asarray(fits244[key248])
    stdcoef248 = np.r_[rawcoef248[:768] * scale244, rawcoef248[-1]]
    p248 = X244 @ stdcoef248
    print(key248, "factual RMS", np.sqrt(np.mean(p248**2)),
          "initial max", np.max(np.abs(p248[initial_mask244])))
    for contrast248 in D244:
        d248 = D244[contrast248] @ (B244.T @ stdcoef248)
        y248 = Y244[contrast248]
        print(" ", contrast248, "slope", d248 @ y248 / (y248 @ y248),
              "mean", d248.mean(), "RMS", np.sqrt(np.mean(d248**2)))

def compact248(fn, limit=3000):
    groups, line, parts = [], None, []
    for ins in dis.get_instructions(fn):
        if ins.starts_line is not None and ins.starts_line != line:
            if parts:
                groups.append(str(line) + ": " + " | ".join(parts))
            line, parts = ins.starts_line, []
        if ins.opname not in ("RESUME", "CACHE", "EXTENDED_ARG", "PRECALL", "NOP"):
            parts.append(ins.opname + (" " + ins.argrepr if ins.argrepr else ""))
    if parts:
        groups.append(str(line) + ": " + " | ".join(parts))
    print("\n".join(groups)[:limit])

print("\nLOCAL UPDATE")
compact248(core_update_local, 4100)
print("\nPRELUDE HIDDEN")
compact248(prelude_hidden93, 1700)
print("\nWEIGHT AND STATE ACCESS")
for name248 in ("core_lin", "core_norm", "inputs191", "encoded_bundles243"):
    fn248 = globals()[name248]
    print(name248, fn248.__name__, "globals", fn248.__code__.co_names,
          "constants", repr(fn248.__code__.co_consts)[:450])
print("\nREFERENCED STATE OBJECTS")
refs248 = set(inputs191.__code__.co_names) | set(encoded_bundles243.__code__.co_names)
for name248 in sorted(refs248):
    obj248 = globals().get(name248)
    if isinstance(obj248, dict):
        print(name248, "dict", list(obj248.keys())[:14])
    elif hasattr(obj248, "shape"):
        print(name248, obj248.shape)
print("CURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 248: correct coefficient-convention audit\n"
            "Run247 incorrectly treated fits244 as standardized coefficients. "
            "Testing the raw-hidden-feature convention before any edit, and recovering "
            "the local prelude-to-state computation. Joint236 unchanged. "
            "Python runs: 248; forwards: 29.\n")
```
