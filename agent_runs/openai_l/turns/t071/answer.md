The one-step reconstruction matches the exported summary state to about \(10^{-5}\) RMS. This supports using it for local dose perturbations. I’ll next check the decoder and dose-encoding interfaces, then test whether a small readout correction can recover the missing response while leaving factual predictions nearly unchanged.

```python
def context64(out, h, dt):
    flowed = h * torch.exp(-dt[:, None, None] * w63("core.flow.a").exp())
    ctx = (out + norm63(flowed.mean(1), "core.out_norm")
           + mlp63(torch.log1p(dt)[:, None] / 10, "gap"))
    return ctx, flowed

def transition64(hflow, e, u, z, memory, ages, mask):
    tokens = torch.cat([
        hflow + w63("core.slots")[None], e[:, None], u[:, None],
        linear63(z, "z_in")[:, None]
    ], 1)
    E = norm63(layer63(tokens, "core.prelude.0")[:, :4], "core.prelude_norm")
    age_emb = mlp63(torch.log1p(ages), "core.age")
    blocks = hflow + E
    for j in (0, 1):
        blocks = layer63(blocks, f"core.blocks.0.{j}",
                        memory, age_emb, mask, "both")
    delta = F.softplus(w63("core.delta_in"))
    abar = torch.exp(-delta * w63("core.a_in").exp())
    hnew = (abar * hflow + delta * linear63(E, "core.b_in")
            + blocks - (hflow + E))
    hnew = layer63(linear63(hnew, "core.c_out"), "core.coda.0")
    return norm63(hnew.mean(1), "core.out_norm"), hnew

ctx64, hf64 = context64(sx63[idx63 - 1, :192], hp63, dt63)
post64 = mlp63(torch.cat([ctx64, e_valid60[idx63], u_valid60[idx63]], 1), "z_post")
print("Posterior mean RMS:", (post64[:, :32] - z63).square().mean().sqrt().item())

mem64, ages64, mask64 = memory63[True]
out64, h64 = transition64(hf64, e_valid60[idx63], u_valid60[idx63],
                         post64[:, :32], mem64, ages64, mask64)
print("Summary-state RMS:", (out64 - sx63[idx63, :192]).square().mean().sqrt().item())

print("\n=== ENCODER INTERFACES ===")
for name64 in ("atom_embedding49", "unpack49", "encode_set50", "pool_actions49",
               "r32_features", "r30_mlp"):
    fn64 = globals()[name64]
    print(name64, "defaults", fn64.__defaults__,
          "references", fn64.__code__.co_names)

print("\n=== RAW ACTION ROWS ===")
print(type(start_rows49).__name__)
if hasattr(start_rows49, "head"):
    print(start_rows49.head(3).to_string(index=False))
else:
    print(repr(start_rows49[:3]))
try:
    unpacked64 = unpack49(start_rows49)
    print("unpacked:", [(type(v).__name__, getattr(v, "shape", None))
                         for v in unpacked64])
    print("first entries:", [repr(v[:3]) for v in unpacked64])
except Exception as exc64:
    print(type(exc64).__name__, str(exc64))

print("\n=== DECODER SPEC AND ARRAYS ===")
print("r32_best:", repr(r32_best))
for name64 in ("r32_base", "r32_dt", "ds22", "last_z22", "age_hours22",
               "has22", "code22", "r31_bins"):
    v64 = globals().get(name64)
    print(name64, type(v64).__name__, getattr(v64, "shape", None))
    if isinstance(v64, dict):
        print("keys:", list(v64)[:12])
if hasattr(ds22, "columns"):
    print("ds22 columns:", list(ds22.columns))
    print(ds22.head(2).to_string(index=False))
print("value-head tensors:", [(n, SHAPES[n]) for n in NAMES
                              if n.startswith(("attr_heads.value_bin", "last_numeric",
                                               "query_norm", "type_emb", "last_status"))])
try:
    feat64 = r32_features(r32_best)
    print("feature outputs:", [tuple(v.shape) for v in feat64])
except Exception as exc64:
    print("Feature call:", type(exc64).__name__, str(exc64))

with open("/work/notes.md", "a") as f64:
    f64.write("\n## Run 64\n"
              "Best recurrence uses GELU, includes the current e in memory, and adds "
              "core.age(log1p(age_seconds)) to both cross-attention keys and values. "
              "Out-state reconstruction RMS is approximately 1e-5. "
              "Inspecting interfaces for an analytic head-only correction before altering recurrence. "
              "No edits.\n")
```