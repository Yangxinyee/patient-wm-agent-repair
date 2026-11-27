The recovered record analysis supports a persistent insulin effect and a shorter positive norepinephrine effect—both much stronger than the original rollouts. I’ll now assemble the exact drug-bundle inputs for a direct-path repair. The key test is whether dose innovations remain extractable after attention mixes actions with context.

```python
print("=== STATE AND REPRESENTATION INDEX ===")
print("S:", type(S).__name__,
      {k: getattr(v, "shape", None) for k, v in S.items()})
for prefix78 in ("r45_", "r66_"):
    print(prefix78, [
        (k, tuple(v.shape))
        for k, v in list(globals().items())
        if k.startswith(prefix78) and isinstance(v, (np.ndarray, torch.Tensor))
    ])
for name78 in ("e_all", "u_all", "e39", "u38", "repair_actual",
               "repair_context", "r40_info", "bundle_atom_info"):
    v78 = globals().get(name78)
    print(name78, type(v78).__name__, getattr(v78, "shape", None),
          len(v78) if isinstance(v78, (list, dict)) else "")

print("=== CORE SHAPES ===")
print([(n, SHAPES[n]) for n in NAMES if n.startswith(
    ("core.b_in", "core.c_out", "core.a_in", "core.delta_in",
     "core.prelude.0.ffn", "core.prelude_norm", "core.coda"))])

print("=== ATOM BUILDER: COMPACT INSTRUCTIONS ===")
print("signature:", inspect.signature(make_atom))
instructions78 = [
    f"{i.opname} {i.argrepr}"
    for i in dis.get_instructions(make_atom)
    if i.opname not in ("RESUME", "CACHE", "PRECALL", "PUSH_NULL")
]
print("\n".join(instructions78)[:3300])

print("=== EMPTY ENCODER BASELINES ===")
for prefix78, lo78, hi78 in (
        ("encode", 192, 384), ("encode_action", 384, 576)):
    col78 = r67_probe_columns[
        (r67_probe_columns >= lo78) & (r67_probe_columns < hi78)
    ] - lo78
    zero78 = torch.zeros((1, 1), dtype=torch.float32)
    raw78 = mlp(linear(zero78, prefix78 + ".size"), prefix78 + ".mlp2")
    norm78 = layer_norm(raw78, prefix78 + ".norm")
    mask78 = np.array([
        not (bool(info[0]) or bool(info[1]))
        if isinstance(info, tuple) and len(info) == 2 else False
        for info in bundle_atom_info
    ])
    print(prefix78, "selected coordinates", col78.tolist(),
          "raw", raw78[0, col78].numpy().round(5).tolist(),
          "normalized", norm78[0, col78].numpy().round(5).tolist())

print("=== ORIGINAL STATE SAMPLE ===")
print("hours:", np.asarray(S["hour"])[:8])
print("edit count:", len(edited()))

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 78; forwards 7. Record-derived kernels suggest persistent "
        "insulin lowering and shorter positive NE response; original rollout "
        "responses are far smaller. Preparing exact drug-bundle prelude features "
        "to test extraction of dose innovations after attention. No edits.\n"
    )
```