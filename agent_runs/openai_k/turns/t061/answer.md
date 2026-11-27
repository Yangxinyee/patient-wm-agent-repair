The corrected readout now returns finite values on plausible glucose and MAP scales. Earlier record analyses found immediate insulin and norepinephrine responses, while the original rollouts were much weaker; that supports testing a missing dose-response pathway. I’ll measure which latent directions can carry those responses and recover the dynamic estimates used to set their strength.

```python
print("=== RELEVANT WEIGHT SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_in", "z_prior", "z_head",
                             "core.flow", "core.a_in", "core.delta_in",
                             "core.b_in", "embed.numeric", "attr_heads.value")):
        print(n, SHAPES[n])

print("\n=== CORE PROBE OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith("core_probe"):
        print(k, type(v).__name__, getattr(v, "shape", ""),
              str(v)[:120] if not hasattr(v, "shape") else "")
print("\n=== CORRECTION / FIT OBJECTS ===")
for k, v in list(globals().items()):
    if k.startswith(("rr39", "rr40", "rr41", "r47", "rr47")):
        if isinstance(v, dict):
            detail = "keys=" + str(list(v)[:12])
        elif hasattr(v, "shape"):
            detail = str(v.shape)
        elif isinstance(v, (list, tuple)):
            detail = str(v)[:180]
        else:
            detail = ""
        print(k, type(v).__name__, detail)

print("\n=== IDENTIFICATION NOTES ===")
txt55 = Path("/work/notes.md").read_text()
print(txt55[:4300])
print("\n=== DYNAMIC-ESTIMATE OBJECT NAMES ===")
print([(k, type(v).__name__, str(getattr(v, "shape", "")))
       for k, v in list(globals().items())
       if any(s in k.lower() for s in
              ("impulse", "deconv", "causal", "effect", "response", "horizon"))][:60])
```
