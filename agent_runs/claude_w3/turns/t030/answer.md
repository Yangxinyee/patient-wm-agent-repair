The tensor list shows the value head is a two-layer MLP on the code-specific vector with_code (192→192, an activation, 192→17), plus a transition bias table embed_transition.value_bin with one row per code and last bin (13 codes × 17). Because of the nonlinear hidden layer, a code-specific edit is possible. I could shift glucose_finger predictions in response to an insulin feature without shifting MAP.

A few details are still unclear:
- last_numeric has 8 inputs, while ARCHITECTURE.md lists 4.
- The activation function is not named.
- I need to know exactly what the action encoder sees.

Next I will read tensors.json, which may document each tensor, and print the embed.numeric weights and the bin embeddings' norms. That shows how strongly dose magnitude enters the model, and whether bins never used in training (norepinephrine rate bins 13–16) look untrained.

```python
for k, v in list(T.items())[:8]:
    print(k, v)
print("....")
for k in ['last_numeric.weight','attr_heads.value_bin.0.weight','embed.numeric.weight','encode_action.mlp1.0.weight','filter_gate.weight','embed_transition.value_bin.weight','type_emb.weight','embed.kq.weight']:
    print(k, T.get(k))
Wn = load('embed.numeric.weight')
print("embed.numeric column norms [z_value, has_value, z_dose, has_dose, z_rate, has_rate]:", np.linalg.norm(Wn, axis=0).round(3))
for nm in ['embed.bins.value.weight','embed.bins.rate.weight','embed.bins.dose.weight']:
    W = load(nm); print(nm, "row norms:", np.linalg.norm(W, axis=1).round(2))
```