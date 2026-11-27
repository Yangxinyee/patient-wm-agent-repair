The recovered results still show the same gap: insulin is close to its record-based target, while norepinephrine doubling is substantially too weak. No new candidate has been tested yet. I’ll identify the cached contrast arrays and installation logic so the next experiment targets that gap directly; a better doubling-to-halving balance at comparable factual loss would support it.

```python
print("=== HELPER DEPENDENCIES ===")
for name in ["head177_make", "head177_eval", "wm_rawloss152",
             "map172_tau", "scalar_mean192"]:
    f = globals()[name]
    print(name, "args", f.__code__.co_varnames[:f.__code__.co_argcount],
          "globals", f.__code__.co_names)

print("\n=== MC OBJECTS ===")
for k, v in list(globals().items()):
    if isinstance(v, dict) and re.search(r"(mc|nebase|cal19)", k, re.I):
        shapes = [(str(a), getattr(b, "shape", type(b).__name__))
                  for a, b in list(v.items())[:7]]
        if any(isinstance(b, np.ndarray) and b.ndim >= 2 for b in v.values()):
            print(k, shapes)
    elif isinstance(v, np.ndarray) and (
        k.startswith(("cal19", "trial158", "trial159", "wm_v"))
        or re.search(r"(mask|split|rate|dr)_15", k)):
        print(k, v.shape)

print("\n=== INSTALLATION LOGIC ===")
buf = io.StringIO()
dis.dis(head177_make, file=buf)
print(buf.getvalue()[2500:7500])

print("\n=== SELECTED CONSTRUCTION NOTES ===")
for i in [51, 52, 55]:
    print(sections[i][:850])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart audit after forward 21\n"
            "- Kernel targets, first 100 tuning: insulin x2 -10.5288, half +5.2644; "
            "NE x2 +3.8084, half -1.9042; estimates depend on stated identification assumptions.\n"
            "- Forward 20 factual value dNLL +0.00345; forward 21 rollout "
            "-10.983/+5.780 insulin, +0.854/-1.425 NE.\n"
            "- Current four-tensor edit retained; investigating dose-asymmetry-specific MAP directions.\n")
```
