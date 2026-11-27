The insulin discontinuities at 180 and 220 support a negative effect without the observation-frequency jump seen at 150. I’ll now test a factually centred correction: a feature that changes when dose changes, but stays near zero on recorded doses. First I need the exact encoder helpers; small factual residuals and reliable dose contrasts would support this approach.

```python
import inspect
import dis

print("=== ENCODER / CONTEXT HELPER DEFINITIONS ===")
for name in ["recon_record_atom", "recon_sets", "recon_ctx", "_post33"]:
    fun = globals()[name]
    print("\nFUNCTION", name)
    try:
        text = inspect.getsource(fun)
        print(text[:3600])
    except (OSError, TypeError):
        print("locals:", fun.__code__.co_varnames)
        print("constants:", str(fun.__code__.co_consts)[:1600])
        if name in {"recon_record_atom", "_post33"}:
            dis.dis(fun)

print("=== ENCODING CONSTANTS ===")
for name in ["RECON_UNITS", "RECON_RATE_UNITS", "RECON_EKINDS",
             "expanded_recon_best", "END_CFG"]:
    print(name, globals().get(name))
print("=== BUNDLE / STATE FORMAT ===")
print(state_df.head(8).to_string(index=False))
for name in ["record_starts", "record_ends", "drug_starts", "drug_ends"]:
    dic = globals()[name]
    key = next(iter(dic))
    print(name, key, str(dic[key])[:500])
print("S0:", [(key, np.asarray(val).shape) for key, val in S0.items()])
print("=== RECENT PRIVATE ARRAYS ===")
print("; ".join(f"{key}:{tuple(val.shape)}" for key, val in list(globals().items())
                if key.startswith("_") and isinstance(val, (np.ndarray, torch.Tensor))
                and key not in {"_", "__"}))
print("=== CORE TENSOR LAYOUT ===")
print("\n".join(f"{name} {SHAPES[name]}" for name in NAMES
                if name.startswith("core.") or name.startswith("z_in.")))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 38: factual-centred contrast design\n"
            "Plan: fit a linear posterior-hidden feature with near-zero recorded-dose output and dose-proportional contrasts on synthetically re-encoded x2/half drug bundles. This is a statistical feature fit, not gradient training. Need exact helper conventions before constructing it. No edits. Python runs: 38; forward passes: 3.\n")
```