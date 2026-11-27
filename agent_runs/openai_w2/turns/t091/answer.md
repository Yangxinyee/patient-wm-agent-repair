The categorical search also failed, so the observation mismatch is not explained by those tags. This is an implementation issue, not evidence about treatment effects. I’ll check encoder activations and numeric conventions next; if neither resolves it, I’ll stop treating reconstructed observation features as reliable.

```python
print("=== NUMERIC CONVENTIONS ===")
print("Codes:", VOCAB["codes"])
print("Event kinds:", VOCAB["event_kinds"])
print("Numeric note:", VOCAB.get("numeric_note"))
print("Numeric structure:", {
    k: (list(v.keys())[:12] if isinstance(v, dict) else type(v).__name__)
    for k, v in VOCAB["numeric"].items()
})
print("Example observation records:")
for event, cases in measurement_cases.items():
    print(event, cases[0][1])
print("MLP helpers:")
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
```
