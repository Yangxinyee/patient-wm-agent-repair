The architecture confirms that the edited recurrent input can affect later readings, but the output was cut off before the causal evidence appeared. I’ll retrieve just those notebook sections and the relevant result names. Agreement between the threshold estimates and independent record analyses would support the current targets; disagreement would argue against further strengthening.

```python
print("EARLY RECORD-BASED EFFECT EVIDENCE")
budget181 = 6200
for i in [7, 8, 9, 10, 11, 12, 18, 25, 27, 28]:
    text = f"\n--- SECTION {i} ---\n{sections180[i]}"
    if len(text) <= budget181:
        print(text)
        budget181 -= len(text)
    else:
        print(text[:budget181])
        break

print("\nSTATISTICAL RESULT VARIABLES")
names181 = []
for name in sorted(list(globals())):
    if re.search(r"iv|effect|causal|threshold|latent|persist|rho|target|validated|cohort", name, re.I):
        obj = globals()[name]
        if callable(obj):
            desc = "function"
        elif isinstance(obj, dict):
            desc = "dict " + str(list(obj.keys())[:8])
        elif hasattr(obj, "shape"):
            desc = str(obj.shape)
            if hasattr(obj, "columns"):
                desc += " " + str(list(obj.columns)[:8])
        elif isinstance(obj, (str, int, float, list, tuple)):
            desc = str(obj)[:100]
        else:
            continue
        names181.append(name + ": " + desc)
print("\n".join(names181)[:2200])
print("\nWEIGHT CHECK",
      np.max(np.abs(load("core.b_in.weight") -
                    np.asarray(validated_B176, dtype=np.float32))))
```