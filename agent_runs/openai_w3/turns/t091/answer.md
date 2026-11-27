The encoding search succeeded: finger-glucose readings use unit ID 0, and END atoms are observations with cleared attributes and units. Both bundle vectors now match the probe to numerical precision. I’ll verify the posterior across the full stream, then extract prelude features for a dose-innovation correction.

```python
idx81 = np.asarray(r45_valid, dtype=int)
h81 = torch.as_tensor(r45_hf, dtype=torch.float32)
ctx81 = torch.as_tensor(r45_ctx, dtype=torch.float32)
e81 = torch.as_tensor(e80[idx81], dtype=torch.float32)
u81 = torch.as_tensor(u80[idx81], dtype=torch.float32)
z81_original = torch.as_tensor(S["x"][idx81, 960:992], dtype=torch.float32)
z81_local = mlp(torch.cat([ctx81, e81, u81], dim=1), "z_post")[:, :32]
print("full-stream posterior RMSE:",
      float((z81_local - z81_original).square().mean().sqrt()))
print("posterior max absolute error:",
      float((z81_local - z81_original).abs().max()))

def prelude81(h, e, u, z):
    tokens = torch.cat([
        h + W["core.slots"][None],
        e[:, None], u[:, None], linear(z, "z_in")[:, None]
    ], dim=1)
    n1 = layer_norm(tokens, "core.prelude.0.norm1")
    q = linear(n1, "core.prelude.0.attn.q")
    k, v = linear(n1, "core.prelude.0.attn.kv").chunk(2, dim=-1)
    def heads81(t):
        return t.reshape(len(t), 7, 4, 48).transpose(1, 2)
    q, k, v = heads81(q), heads81(k), heads81(v)
    attention = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(48), dim=-1)
    joined = (attention @ v).transpose(1, 2).reshape(len(tokens), 7, 192)
    tokens1 = tokens + linear(joined, "core.prelude.0.attn.out")
    features = layer_norm(tokens1, "core.prelude.0.norm3")
    hidden = torch.nn.functional.gelu(
        linear(features, "core.prelude.0.ffn.0")
    )
    tokens2 = tokens1 + linear(hidden, "core.prelude.0.ffn.2")
    E = layer_norm(tokens2[:, :4], "core.prelude_norm")
    return E, features, hidden, attention, tokens2

E81, F81, H81, attention81, tokens81 = prelude81(
    h81, e81, u81, z81_original
)
E81_reference = r66_prelude(h81, e81, u81, z81_original)
print("prelude implementation RMSE:",
      float((E81 - E81_reference).square().mean().sqrt()))

rates81 = np.zeros((len(r40_info), 2), dtype=np.float32)
presence81 = np.zeros_like(rates81)
drug_names81 = ("insulin", "norepinephrine")
for i81, info81 in enumerate(r40_info):
    for row81, end81 in info81:
        if not end81 and row81.event in drug_names81:
            j81 = drug_names81.index(row81.event)
            rates81[i81, j81] += float(row81.rate)
            presence81[i81, j81] = 1
active81 = presence81[idx81].any(axis=1)
print("WORLD drug bundles:", int(active81.sum()),
      "per drug:", presence81[idx81].sum(axis=0).tolist())
print("mean slot attention to [observation, action, latent]:",
      attention81[active81, :, :4, 4:].mean((0, 1, 2)).numpy().tolist())
print("FFN feature shapes:", tuple(F81.shape), tuple(H81.shape))

print("=== RECURRENT BLOCK TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("core.blocks")])
print("=== HEAD SUPPORT TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES if n.startswith(
    ("query_norm", "type_emb", "last_", "z_head", "gap", "core.age"))])

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 81; forwards 7. Exact preprocessing recovered: "
        "glucose_finger unit=0, glucose_lab=5, MAP=2; END atoms route to "
        "observations with cleared numeric/unit fields; empty action is its "
        "encoded baseline. Verified full-stream posterior and reconstructed "
        "prelude features. No weight edits.\n"
    )
```