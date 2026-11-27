The errors are systematic, particularly for arterial MAP, rather than just small prediction differences. The local head also infers the next time by rounding to an hour, which may be wrong when bundles are skipped. I’ll check actual bundle times and the last-value register; correcting either should reduce the corresponding errors if it is responsible.

```python
print("=== STATE FILE ===")
print([(k, sfile101[k].shape, str(sfile101[k].dtype)) for k in sfile101.files])
print("=== LOCAL READOUT OBJECTS ===")
for n102 in sorted(n for n in globals() if n.startswith("r49_") or n.startswith("r59_")):
    o102 = globals()[n102]
    if isinstance(o102, (np.ndarray, torch.Tensor)):
        print(n102, tuple(o102.shape), repr(o102.reshape(-1)[:8])[:180])
    elif isinstance(o102, (list, tuple, dict)):
        print(n102, type(o102).__name__, len(o102), repr(o102)[:200])
print("=== READOUT FUNCTION, REGISTER AND LOGIT TAIL ===")
tail102 = [i for i in dis.get_instructions(r49_readouts) if i.offset >= 1498]
txt102 = [f"{i.offset}:{i.opname} {i.argrepr}" for i in tail102]
print("\n".join(" | ".join(txt102[k:k+7]) for k in range(0, len(txt102), 7))[:3700])
print("=== DATASET CONSTRUCTION ===")
print("signature", inspect.signature(dataset80))
print("variables", dataset80.__code__.co_varnames)
print("globals", dataset80.__code__.co_names)
print("constants", repr(dataset80.__code__.co_consts)[:800])
print("=== EXACT MATCHES WITH EXTREME ERRORS ===")
show102 = direct98.copy()
show102["abs_error"] = abs(show102.local_mean - show102.predicted_mean)
print(show102.nlargest(6, "abs_error").to_string(index=False))
print("=== RECENT ASSERTION INPUTS ===")
for n102 in ("fact_indices98", "fact_target_hours98", "fact_target_stays98",
             "all_pre98", "eligible98", "direct_hours98", "direct_stays98",
             "features97", "valid97"):
    o102 = globals()[n102]
    print(n102, type(o102).__name__, getattr(o102, "shape", None),
          [getattr(x, "shape", None) for x in o102] if isinstance(o102, tuple)
          else repr(o102[:7])[:180])
with open("/work/notes.md", "a") as f102:
    f102.write("\n## Run 102: timing and register audit\n"
               "- Local errors are systematic, especially arterial MAP; next-time rounding and register construction are under examination.\n"
               "- Python runs: 102; forward passes: 4; no submitted edits.\n")
```
