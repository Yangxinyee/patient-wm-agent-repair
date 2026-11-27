The encoder added essentially no factual NLL cost: the total increase remains 0.00245 nats. That leaves about 0.00255 for routing and readout changes. I’ll inspect the exact attention and recurrence layout next, aiming for a dose channel that resets between bundles rather than incorrectly adding a cumulative effect on top of sampled-reading feedback.

```python
print("=== ROUTING AND READOUT TENSOR SHAPES ===")
prefixes124 = (
    "core.prelude.0.", "core.prelude_norm.", "core.b_in.",
    "core.a_in", "core.delta_in", "core.flow.", "core.z_in.",
    "core.slots", "core.out_norm.", "core.c_out.",
    "encode.norm.", "attr_heads.value_bin.", "query_norm.",
    "last_numeric.", "z_head.", "gap.", "type_emb."
)
for n in NAMES:
    if n.startswith(prefixes124):
        print(n, SHAPES[n])

print("\n=== RECURRENT RESIDUAL OUTPUTS ===")
for n in NAMES:
    if n.startswith("core.blocks.") and (
        "out.weight" in n or "ffn.2." in n
    ):
        print(n, SHAPES[n])

print("\n=== ATTENTION IMPLEMENTATION ===")
print(outline122(wm_attention)[:3000])
print("\n=== TRANSFORMER LAYER IMPLEMENTATION ===")
print(outline122(wm_layer)[:1800])
print("\n=== PRELUDE DATA HELPER ===")
print(inspect.signature(prelude_all83))
print(outline122(prelude_all83)[:1800])

print("\n=== STATE FILE LAYOUT ===")
statefile124 = np.load("/work/fwd_3/states_original.npz")
print({k: statefile124[k].shape for k in statefile124.files})
```
