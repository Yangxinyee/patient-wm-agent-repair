The inspection stopped because the numeric tables are a list, not a dictionary; it produced no new model result. I’ll handle that structure correctly and finish the activation check. The test remains whether the reconstructed posterior matches saved states, not merely whether its error improves.

```python
print("=== NUMERIC TABLE STRUCTURE ===")
numeric_tables = VOCAB["numeric"]
print("Type:", type(numeric_tables).__name__, "length:", len(numeric_tables))
for i, entry in enumerate(numeric_tables):
    print(i, repr(entry)[:420])

print("\n=== EXAMPLE OBSERVATIONS ===")
for event, cases in measurement_cases.items():
    print(event, cases[0][1])

print("\n=== MLP HELPERS ===")
for name in ("rmlp", "rmlp_act", "act_fn"):
    f = globals().get(name)
    if callable(f):
        print(name, inspect.signature(f),
              getattr(getattr(f, "__code__", None), "co_names", None))

activations = {
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
    "relu": torch.nn.functional.relu,
}
observation_activation_results = []

for event, cases in measurement_cases.items():
    pos = np.array([j for j, row in cases], dtype=int)
    atoms = torch.stack([embed_record_atom(row) for j, row in cases])
    target = sx_repair[
        torch.tensor(posterior_rows_repair[pos], dtype=torch.long), 960:992
    ]
    for name1, activate1 in activations.items():
        enc1 = activate1(
            atoms @ Wrepair["encode.mlp1.0.weight"].T +
            Wrepair["encode.mlp1.0.bias"]
        ) @ Wrepair["encode.mlp1.2.weight"].T + Wrepair["encode.mlp1.2.bias"]
        enc1 = enc1 + (
            torch.full((len(cases), 1), np.log(2), dtype=torch.float32)
            @ Wrepair["encode.size.weight"].T + Wrepair["encode.size.bias"]
        )
        for name2, activate2 in activations.items():
            enc2 = activate2(
                enc1 @ Wrepair["encode.mlp2.0.weight"].T +
                Wrepair["encode.mlp2.0.bias"]
            ) @ Wrepair["encode.mlp2.2.weight"].T + Wrepair["encode.mlp2.2.bias"]
            e_test = rnorm(enc2, "encode.norm")
            inp = torch.cat((
                posterior_ctx_repair[pos], e_test, posterior_u_repair[pos]
            ), dim=1)
            phi = torch.nn.functional.gelu(
                inp @ Wrepair["z_post.0.weight"].T +
                Wrepair["z_post.0.bias"]
            )
            pred = (
                phi @ Wrepair["z_post.2.weight"].T +
                Wrepair["z_post.2.bias"]
            )[:, :32]
            error = float(torch.sqrt(((pred - target) ** 2).mean()))
            observation_activation_results.append((event, name1, name2, error))

observation_activation_table = pd.DataFrame(
    observation_activation_results,
    columns=["event", "mlp1_activation", "mlp2_activation", "z_rmse"]
)
print("\n=== BEST OBSERVATION ACTIVATIONS ===")
print(observation_activation_table.sort_values("z_rmse")
      .groupby("event", sort=False).head(2)
      .sort_values(["event", "z_rmse"]).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 73–78: observation reconstruction checks\n"
            "Architecture reread; posterior hidden features are 192-dimensional. "
            "Measurement unit and categorical-tag searches did not resolve the "
            "observation-only posterior mismatch. No weights edited. "
            "Checking numerical tables and encoder activations before relying "
            "on observation features for a factual-preserving edit.\n")
```
