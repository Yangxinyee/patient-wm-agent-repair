The recovered notes show that history-adjusted associations differ substantially from simpler regressions, especially for norepinephrine. They are not sufficient causal evidence by themselves. I’ll recover the stronger identification checks and locate the reconstruction code. Agreement across those checks would support a repair target; disagreement would require a more cautious target.

```python
print("=== NOTEBOOK: IDENTIFICATION CHECKS ===")
print(notes_text[4500:10800])

print("\n=== RECONSTRUCTION FUNCTIONS ===")
for name, obj in list(globals().items()):
    code = getattr(obj, "__code__", None)
    if code is not None:
        text = repr(code.co_consts)
        if any(s in text for s in ["last_numeric.weight", "core.prelude", "embed.numeric"]):
            print(name, "args:", code.co_varnames[:code.co_argcount],
                  "names:", code.co_names)

print("\n=== CAUSAL RESULT TABLES ===")
for filename in ["dynamic_iv.csv", "sensor_iv.csv", "kernel_targets.csv"]:
    table = pd.read_csv("/work/" + filename)
    print(filename, table.shape)
    print(table.head(8).to_string(index=False))
```
