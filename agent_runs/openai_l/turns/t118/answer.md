The corrected latent probe found almost no sensitivity to the model’s existing posterior coordinates. So extracting dose information there is not enough: that information needs a functioning route into the state. I’ll check whether unused latent coordinates can provide that route, and whether the residual can be gated to drug bundles. The hypothesis fails if the route cannot move readings selectively or activates strongly on non-drug bundles.

```python
print("=== LATENT PATH MAGNITUDES ===")
for name109 in (
    "z_in.weight", "z_in.bias", "z_head.weight", "z_head.bias",
    "z_post.0.weight", "z_post.2.weight",
    "z_prior.0.weight", "z_prior.2.weight"
):
    a109 = np.asarray(load_original(name109))
    print(name109, "norm", float(np.linalg.norm(a109)),
          "max_abs", float(np.max(np.abs(a109))))
print("Existing finite-difference sensitivity:",
      float(np.sqrt(np.mean(np.asarray(G93) ** 2))),
      float(np.max(np.abs(np.asarray(G93)))))

post_all109 = mlp63(torch.as_tensor(X104, dtype=torch.float32),
                   "z_post", act="gelu")
print("Posterior mean SD per coordinate:",
      np.round(post_all109[:, :32].std(0).numpy(), 5))

print("\n=== RESIDUALS BY BUNDLE TYPE ===")
for drug109 in ("insulin", "norepinephrine"):
    v109 = extractors104[("raw", drug109, 0.1)]
    score109 = (np.asarray(X104) - mean_raw104) @ v109
    dose109 = np.asarray(dose_by_bundle102[drug109])[ids104]
    active109 = dose109 > 0
    for label109, mask109 in (("all", np.ones(len(score109), dtype=bool)),
                             ("positive dose", active109),
                             ("zero dose", ~active109)):
        print(drug109, label109, "n", int(mask109.sum()),
              "mean", float(score109[mask109].mean()),
              "SD", float(score109[mask109].std()),
              "range", np.round(np.quantile(score109[mask109],
                                           [0.01, 0.5, 0.99]), 4))

print("\n=== PRELUDE TENSORS ===")
for name109 in NAMES:
    if "core.prelude." in name109:
        print(name109, SHAPES[name109])

print("\n=== ACTION AND COUNTERFACTUAL INTERFACES ===")
for fun109 in (prelude_features102, action_embedding67, make_cf81):
    print(fun109.__name__, "names", fun109.__code__.co_names,
          "constants", repr(fun109.__code__.co_consts)[:550])
print("sid/hour shapes:", np.shape(sid63), np.shape(hour63))
print("Raw feature type:", type(X104), "action feature type:", type(u_valid60))

with open("/work/notes.md", "a") as f109:
    f109.write(
        "\n## Runs 106–109 / causal and latent-path recovery\n"
        "Accepted immediate effects: I≈−1.94; NE≈+3.05. Record impulse kernels "
        "support delayed/declining insulin and faster-decaying NE responses. "
        "Existing z-injection finite differences are nearly insensitive, so a "
        "posterior-only edit is insufficient. Checking unused latent channels and "
        "drug-bundle gating before constructing a new selective state route. No edits.\n"
    )
```
