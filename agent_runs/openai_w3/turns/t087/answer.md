The prelude implementation exposes a usable insertion point: its feed-forward output directly changes the four recurrent slots. The output was truncated before the statistical evidence, so I’ll retrieve that evidence and the available arrays compactly. A useful repair must preserve factual dose associations while responding differently to dose changes.

```python
print("=== EARLY NOTEBOOK: IDENTIFICATION EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:4300])

print("=== RETAINED EFFECT KERNELS ===")
for k, v in kernel_refs.items():
    if isinstance(v, dict):
        print(k, {kk: repr(vv)[:220] for kk, vv in v.items()})
    else:
        print(k, repr(v)[:600])

print("=== HELPER METADATA ===")
for name in ("atom_base", "encode_bundles", "layer_norm", "mlp",
             "repair_action_vectors", "context_with_flow"):
    fn77 = globals()[name]
    print(name, inspect.signature(fn77),
          "names:", fn77.__code__.co_names,
          "constants:", repr(fn77.__code__.co_consts)[:400])

print("=== TABLE / ARRAY INDEX ===")
for k, v in list(globals().items()):
    if isinstance(v, pd.DataFrame):
        print("TABLE", k, v.shape, list(v.columns)[:10])
    elif k.startswith(("r45_", "r66_", "r70_")) and isinstance(
            v, (np.ndarray, torch.Tensor)):
        print("ARRAY", k, tuple(v.shape))

print("=== RELEVANT CORE SHAPES ===")
for name in NAMES:
    if name.startswith(("core.prelude.0.ffn", "core.coda",
                        "core.c_out", "core.b_in", "core.a_in",
                        "core.delta_in", "core.out_norm")):
        print(name, SHAPES[name])

print("probe columns:", r67_probe_columns.tolist())
print("edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 77; forwards 7. Prelude FFN modifies slots immediately "
        "before their normalization and recurrent update: a potentially effective "
        "dose-innovation insertion point. Retrieving compact identification "
        "evidence and retained matrices after run 76 output truncation. No edits.\n"
    )
```