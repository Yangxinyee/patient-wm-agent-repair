The channel uses a difference of two SiLU units, not a hard on/off mask. That can distort doses near the gate boundary, but the numerical diagnostics were cut off. I’ll retrieve those compactly and locate the cached fitting arrays; saturated gates would refute gate suppression as the main explanation.

```python
print("COEFFICIENT MATCH", sorted(matches190, key=lambda x: x[0])[:3])

print("\nCOMPACT HELD-OUT GATE DIAGNOSTICS")
n191 = len(ix180)
for j in range(len(raws180)):
    scores = gate_fits180[100.0].decision_function(
        xx180[j*n191:(j+1)*n191]
    )
    active = np.asarray(active_ne180, dtype=bool)
    held = np.asarray(test180, dtype=bool)
    sa = scores[held & active]
    si = scores[held & ~active]
    print(
        "augmentation", j,
        "active margin [min,p01,median]",
        np.round(np.quantile(sa, [0, .01, .5]), 4),
        "active mean gate", round(float(expit(40*sa).mean()), 6),
        "inactive max margin", round(float(si.max()), 4),
        "inactive mean gate", round(float(expit(40*si).mean()), 8),
        "raw shape", getattr(raws180[j], "shape", None),
    )

print("\nCACHED FITTING ARRAYS")
for name, value in list(globals().items()):
    if name.endswith(("87", "102", "185", "187")):
        if isinstance(value, np.ndarray) or torch.is_tensor(value):
            print(name, tuple(value.shape))
        elif isinstance(value, dict) and any(
            word in name.lower() for word in ("raw", "coef", "target", "cf", "amplitude")
        ):
            print(name, "keys", list(value.keys())[:8])
            for key, entry in list(value.items())[:2]:
                if isinstance(entry, dict):
                    print(" ", key, {k: summary189(v) for k, v in entry.items()})
                else:
                    print(" ", key, summary189(entry))

print("\nTRANSITION DEPENDENCIES", transition73.__code__.co_names)
print("DELTA185", np.quantile(delta185, [0, .1, .5, .9, 1]))
print("ACTIVE INDEX ALIGNMENT",
      np.array_equal(ix180[active_pos187], active_ix187))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 191\n"
        "The NE channel uses silu(g+q/2)-silu(g-q/2). Retrieving compact "
        "gate-margin results and cached fitting arrays after disassembly output "
        "truncation. No saved weights changed.\n"
    )
```