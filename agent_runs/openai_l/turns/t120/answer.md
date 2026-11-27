Direct token perturbations also produced almost no reading change. Before concluding that the route is unusable, I need to check that the probe actually changes the recurrence—not merely a cache that the recurrence does not read. I’ll compare prelude features and states under a large perturbation. A state change without a reading change would implicate the downstream readout; no state change would implicate the probe or token route.

```python
print("=== TRANSITION DEPENDENCIES ===")
print("names:", transition64.__code__.co_names)
print("constants:", transition64.__code__.co_consts)
print("\n=== TRANSITION IMPLEMENTATION ===")
show91(transition64, limit=4200)
print("\n=== fn64 IMPLEMENTATION ===")
show91(fn64, limit=900)

with torch.no_grad():
    e111 = e_valid60[ids93]
    u111 = u_valid60[ids93]
    oo0_111, hh0_111 = transition64(
        hf93, e111, u111, post93, mem93, ages93, ~padding93
    )
    oo1_111, hh1_111 = transition64(
        hf93, e111, u111,
        post93 + 10.0 * torch.ones_like(post93),
        mem93, ages93, ~padding93
    )
    print("\n=== LARGE LATENT PERTURBATION ===")
    print("out RMS/max change:",
          float(torch.sqrt(((oo1_111 - oo0_111) ** 2).mean())),
          float((oo1_111 - oo0_111).abs().max()))
    print("h RMS/max change:",
          float(torch.sqrt(((hh1_111 - hh0_111) ** 2).mean())),
          float((hh1_111 - hh0_111).abs().max()))

    E0_111 = prelude_features102(ctx93, hf93, e111, u111)
    old_bias111 = w63_cache["z_in.bias"]
    try:
        w63_cache["z_in.bias"] = original_bias110 + 10.0 * torch.as_tensor(
            basis110[:, 0], dtype=torch.float32
        )
        E1_111 = prelude_features102(ctx93, hf93, e111, u111)
        oo2_111, hh2_111 = transition64(
            hf93, e111, u111, post93, mem93, ages93, ~padding93
        )
    finally:
        w63_cache["z_in.bias"] = old_bias111
    print("\n=== CACHE PERTURBATION CHECK ===")
    print("Prelude E RMS/max change:",
          float(torch.sqrt(((E1_111 - E0_111) ** 2).mean())),
          float((E1_111 - E0_111).abs().max()))
    print("Transition out RMS/max change:",
          float(torch.sqrt(((oo2_111 - oo0_111) ** 2).mean())),
          float((oo2_111 - oo0_111).abs().max()))
    print("Transition h RMS/max change:",
          float(torch.sqrt(((hh2_111 - hh0_111) ** 2).mean())),
          float((hh2_111 - hh0_111).abs().max()))

print("Edits:", edited())
```
