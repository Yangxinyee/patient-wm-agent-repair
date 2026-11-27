The records support immediate insulin and norepinephrine effects near −1.95 and +3.1, respectively; the early conflicting estimates were rejected. The reweighted candidate has more factual margin offline, but I still need to check its regression’s scaling and targets. I’ll inspect that calculation before staging it.

```python
import inspect
import dis

print("=== REGRESSION DEFINITIONS ===")
for name in ["weighted_fit281", "solve281", "solve269"]:
    obj = globals().get(name)
    if callable(obj):
        print(name, inspect.signature(obj))
        if name == "weighted_fit281":
            print(dis.Bytecode(obj).dis()[:4300])

print("=== CONTRAST AND SCALING OBJECTS ===")
for name in ["partial281", "anchor281", "stats263", "sc263", "sc269",
             "D281", "Ds281", "G281", "R281", "check281", "error281"]:
    if name in globals():
        print(name, describe282(globals()[name]))

print("=== ACCEPTED PERSISTENCE AND ROLLOUT EVIDENCE ===")
start284 = notes282.find("## Run 18")
print(notes282[start284:start284 + 2100])
for name in ["record_targets22", "slope_target214", "truth259"]:
    if name in globals():
        print(name, describe282(globals()[name]))

current284 = np.vstack([
    load("attr_heads.value_bin.2.weight").numpy().T,
    load("attr_heads.value_bin.2.bias").numpy()[None, :]
])
print("Saved head versus balanced grid:",
      float(np.max(np.abs(current284 - grid269[(0.25, 0.25)]["T"]))))
print("Candidate versus saved head:",
      float(np.linalg.norm(weighted281[(0.3, 0.25, 8.0)]["T"] - current284)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 284 / regression audit\n"
            "No weights changed. Inspecting weighted_fit281 scaling and contrast assembly "
            "before staging; accepted immediate effects come from corrected DML and "
            "insulin threshold checks, not the rejected early SSM fits.\n")
```