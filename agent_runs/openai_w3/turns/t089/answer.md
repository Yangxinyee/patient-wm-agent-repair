The empty-bundle discrepancy is resolved: both encoders produce nonzero learned baselines, and those baselines match the probe. I’ll check measurement units and overlapping infusion starts/ends next. Exact agreement there would let us fit the repair on faithful prelude inputs rather than approximations.

```python
empty_e79 = layer_norm(
    mlp(linear(torch.zeros(1, 1), "encode.size"), "encode.mlp2"),
    "encode.norm"
).numpy()[0]
empty_u79 = layer_norm(
    mlp(linear(torch.zeros(1, 1), "encode_action.size"),
        "encode_action.mlp2"),
    "encode_action.norm"
).numpy()[0]

e_cols79 = r67_probe_columns[
    (r67_probe_columns >= 192) & (r67_probe_columns < 384)
] - 192
u_cols79 = r67_probe_columns[r67_probe_columns >= 384] - 384
e_true79 = probe_inputs74[:, (r67_probe_columns >= 192)
                              & (r67_probe_columns < 384)]
u_true79 = probe_inputs74[:, r67_probe_columns >= 384]

hours79 = np.asarray(S["hour"])
integer79 = (hours79 >= 1) & np.isclose(hours79, np.round(hours79))
print("integer-hour empty-u RMSE:",
      np.sqrt(np.mean((u_true79[integer79] -
                       empty_u79[u_cols79]) ** 2)))
print("integer-hour e RMSE:",
      np.sqrt(np.mean((e_all.numpy()[integer79][:, e_cols79] -
                       e_true79[integer79]) ** 2)))

print("=== ATOM BUILDER TAIL ===")
print("\n".join(instructions78)[3300:5800])
print("=== CATEGORICAL TABLES ===")
print([(n, SHAPES[n]) for n in NAMES
       if n.startswith("embed.categorical")])
print("vocab keys:", list(VOCAB))
for k79 in VOCAB:
    if any(s79 in k79.lower() for s79 in ("unit", "kind", "categor")):
        print(k79, repr(VOCAB[k79])[:800])

print("=== BUNDLE METADATA EXAMPLES ===")
samples79 = [0, 5, int(np.flatnonzero(
    np.asarray(S["has_insulin"]) | np.asarray(S["has_norepinephrine"])
)[0])]
fractional79 = np.flatnonzero((hours79 > 1) & ~integer79)
samples79 += fractional79[:3].tolist()
for i79 in sorted(set(samples79)):
    print("INDEX", i79, "hour", hours79[i79],
          "flags", S["has_insulin"][i79], S["has_norepinephrine"][i79])
    print("r40_info:", repr(r40_info[i79])[:650])
    print("bundle_atom_info:", repr(bundle_atom_info[i79])[:650])
    print("probe u:", np.round(u_true79[i79], 4).tolist())

print("=== RETAINED DIRECTIONS ===")
print([
    (k, tuple(v.shape))
    for k, v in list(globals().items())
    if isinstance(v, (np.ndarray, torch.Tensor))
    and any(s in k.lower() for s in ("dir", "coef", "target", "sens"))
    and v.size if isinstance(v, np.ndarray)
][:35] if False else [
    (k, tuple(v.shape))
    for k, v in list(globals().items())
    if isinstance(v, (np.ndarray, torch.Tensor))
    and any(s in k.lower() for s in ("dir", "coef", "target", "sens"))
][:35])

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 79; forwards 7. Empty e/u encoder baselines now match "
        "the probe exactly. Isolating measurement-unit and overlapping END/WORLD "
        "routing conventions before prelude dose-innovation fitting. No edits.\n"
    )
```