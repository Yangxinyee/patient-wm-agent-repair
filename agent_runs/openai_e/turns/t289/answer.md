The raw-feature convention passes the check: factual RMS is 0.023, initial-admission responses are effectively zero, and both norepinephrine contrasts have the intended sign. Insulin leakage is small. I’ll choose a MAP-specific output direction using forward differences, so the detector changes the relevant prediction rather than unrelated state dimensions.

```python
print("LOCAL RECONSTRUCTION CONTRACTS")
for name249, limit249 in (("inputs191", 2300), ("memory133", 1600),
                          ("readout125", 1300)):
    print("\n" + name249)
    compact248(globals()[name249], limit249)

print("\nSTATE ARCHIVES")
for k249, v249 in list(globals().items()):
    if k249.startswith("_"):
        continue
    if isinstance(v249, np.lib.npyio.NpzFile):
        keys249 = list(v249.files)
        if "x" in keys249 or "states" in keys249:
            print(k249, keys249[:12],
                  [(a, v249[a].shape) for a in keys249[:5]])
    elif isinstance(v249, dict) and ("x" in v249 or "states" in v249):
        print(k249, list(v249.keys())[:12],
              getattr(v249.get("x"), "shape", None))
print("\nLOCAL ARRAY AND CONFIGURATION CONTRACTS")
for k249 in ("core_h_flow", "core_E_obs", "core_U_actions", "src45", "dt45",
             "static125", "transition125", "centres125", "h_flow191",
             "Bdir141", "gain81", "config", "best_config", "core_config"):
    v249 = globals().get(k249)
    print(k249, type(v249).__name__, getattr(v249, "shape", None),
          repr(v249)[:180] if isinstance(v249, (tuple, str, int, float)) else "")
for k249, v249 in list(globals().items()):
    if any(k249.endswith(str(i)) for i in (189, 190, 191)) and not k249.startswith("_"):
        if hasattr(v249, "shape"):
            print(k249, v249.shape)
        elif isinstance(v249, dict):
            print(k249, "dict", list(v249.keys())[:10])
        elif isinstance(v249, (tuple, str)):
            print(k249, repr(v249)[:130])
print("W0 b_in versus installed RMS",
      np.sqrt(np.mean((as_numpy194(W0["core.b_in.weight"]) -
                       np.asarray(load("core.b_in.weight")))**2)))

print("\nEXACT-RECONSTRUCTION NOTES")
for tag249 in ("Run 133:", "Run 134:", "Run 135:"):
    sec249 = next((s for s in sections246 if tag249 in s[:180]), "")
    print(sec249[:600] if sec249 else "Not found: " + tag249)

np.savez("/work/ne_nonlinear249.npz",
         raw_coefficient=np.asarray(fits244[(100.0, .001)]),
         scale=scale244)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 249: verified nonlinear detector; recover response contexts\n"
            "fits244 stores raw hidden-feature coefficients. Correct full-fit diagnostics "
            "for (100,.001): factual RMS=.02323; admission max=3.1e-14; "
            "NE2/half slopes=.27592/.17945; insulin slopes=.00220/.00284. "
            "Saved detector in ne_nonlinear249.npz. Recovering forward-only MAP-direction "
            "contexts. No edits changed. Python runs: 249; forwards: 29.\n")
```
