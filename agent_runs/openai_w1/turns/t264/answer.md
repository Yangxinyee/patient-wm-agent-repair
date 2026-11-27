Reducing the output pathway tenfold still worsened factual NLL by 0.12212. That contradicts the idea that excessive output gain alone caused the damage. I’ll remove the marker candidate, restore the previously tested relay, and compare its inputs with the earlier carrier. The next route must show small factual state changes, not just small fitted residuals.

```python
print("REJECTED VS EARLIER CARRIER")
for n in rejected230:
    if n in candidate211_weights:
        print(n, "difference", float(np.linalg.norm(
            arr231(rejected230[n]) - arr231(candidate211_weights[n]))))
for n in list(edited()):
    drop_edit(n)
for n, v in candidate228_baseline.items():
    save_edit(n, torch.as_tensor(arr231(v), dtype=torch.float32))
print("Restored relay-only baseline:", edited())

print("\nMARKER OBJECTS")
for k, v in marker215.items():
    print(k, brief230(v))
print("Reserved rows", candidate228_reserved, "head rows", candidate228_rows)
print("Output norms at reserved columns:",
      np.linalg.norm(candidate228_out0[:, candidate228_reserved], axis=0))
print("\nCORE/HEAD HELPERS")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and hasattr(v, "__code__") and (
        any(w in k.lower() for w in ["core", "local", "step", "read", "head", "carrier", "post", "embed", "norm", "forward"])
        or any(k.endswith(str(r)) for r in [206,207,208,209,210,211,213,214,215,220,221,226,227,228,229])):
        c = v.__code__
        print(k, "args", c.co_varnames[:c.co_argcount], "file", c.co_filename,
              "line", c.co_firstlineno)

print("\nRECENT CACHED CORE INPUTS")
for k in sorted(list(globals())):
    if any(k.endswith(str(r)) for r in [206,207,214,215,220,221,227,228,229]) and not k.startswith("_"):
        v = globals()[k]
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            if np.ndim(arr231(v)) >= 2:
                print(k, tuple(v.shape))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 233\nForward22 t=.1 failed factual bound (+.12212). "
            "Rejected scaling hypothesis; restored tested relay-only baseline. "
            "Comparing latent-input changes and investigating a more direct recurrent injection. "
            "Counts: 233 Python runs, 22 forwards.\n")
```