The saved objects include the constrained fits and exact fallback weights. Record analyses still support negative insulin effects and positive norepinephrine effects, but their accumulated magnitudes remain uncertain. I’ll recover the coefficient-to-weight mapping before installing the sensitivity-preserving insulin refit. An exact reconstruction of the current edit is the prerequisite; failure would mean the mapping is wrong.

```python
key209 = (("preserve_half", "preserve_cross"), (0.003, 1e-6))
fit209 = constrained_fits206[key209]
print("SELECTED FIT", type(fit209).__name__)
if isinstance(fit209, dict):
    for k, v in fit209.items():
        print(k, type(v).__name__, getattr(v, "shape", None),
              repr(v)[:160] if not hasattr(v, "shape") else "")
elif isinstance(fit209, (tuple, list)):
    for j, v in enumerate(fit209):
        print(j, type(v).__name__, getattr(v, "shape", None),
              repr(v)[:160] if not hasattr(v, "shape") else "")
else:
    print(getattr(fit209, "shape", None), repr(fit209)[:250])

print("\nNORMALIZATION / MAPPING OBJECTS")
for k, v in sorted(dict(globals()).items()):
    if any(s in k.lower() for s in
           ["mu147", "sd147", "std147", "mean147", "scale147",
            "direction", "gain", "affine", "convert", "lift", "decoder_from"]):
        print(describe208(k, v))

print("\nVECTOR-SHAPED NORMALIZATION OBJECTS")
for k, v in sorted(dict(globals()).items()):
    if (hasattr(v, "shape") and tuple(v.shape) in [(192,), (193,)] and
        any(s in k.lower() for s in ["mean", "mu", "std", "sd", "scale", "norm"])):
        print(describe208(k, v))

print("\nRELEVANT NOTEBOOK SECTIONS")
for sec in notes_text.split("\n## "):
    heading = sec.splitlines()[0]
    if any(s in heading.lower() for s in
           ["run147", "run 147", "171", "172", "run150", "run 150"]):
        print(sec[:1200])

print("\nWEIGHT FACTORIZATION CHECK")
base_B209 = load_original("core.b_in.weight").numpy().astype(np.float64)
before_B209 = load("core.b_in.weight").numpy().copy()
update_directions209 = np.linalg.lstsq(
    candidate_W204.astype(np.float64),
    (before_B209.astype(np.float64) - base_B209).T, rcond=None
)[0].T
factor_error209 = np.max(np.abs(
    base_B209 + update_directions209 @ candidate_W204.T - before_B209))
print("maximum reconstruction error", factor_error209)
print("selected constraints target", targets206[("preserve_half", "preserve_cross")])
print("No weight change", np.array_equal(before_B209,
                                          load("core.b_in.weight").numpy()))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run209: constrained-fit mapping audit\n"
            "No weight changes. Inspecting selected preserve-half/preserve-cross "
            "fit and auditing rank-two factorization before installation.\n")
```