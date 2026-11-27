With common random draws, the latent directions barely move predictions. The earlier large responses were mostly sampling artifacts, so I won’t use those directions. I’ll examine the direct action-to-state path instead; evidence for that path would be a reproducible reading change when action inputs—not posterior latents—are perturbed.

```python
print("=== DIRECT-PATH HELPERS ===")
for name in ("action_vectors4", "action_atom_exact", "wm_attention", "wm_layer", "wm_lin"):
    fn81 = globals()[name]
    print(name, inspect.signature(fn81), "globals:", fn81.__code__.co_names)
compact_impl79(action_vectors4)
compact_impl79(wm_attention)

print("\n=== CORE SCALES ===")
for name in ("core.delta_in", "core.a_in", "core.prelude_norm.weight",
             "core.b_in.weight", "core.c_out.weight", "z_in.weight"):
    a = as_np(W0[name])
    print(name, tuple(a.shape), "RMS", float(np.sqrt(np.mean(a*a))),
          "range", (float(a.min()), float(a.max())))
delta81 = torch.nn.functional.softplus(W0["core.delta_in"])
print("softplus(delta) quantiles:",
      np.quantile(as_np(delta81), [0, .1, .5, .9, 1]))

print("\n=== STATE / ACTION OBJECTS ===")
for name in ("rows4", "fit4", "H4", "Q4", "U4", "ctx4", "e4_used",
             "wm_e4", "innovation55", "target", "X55", "data55", "data4"):
    if name in globals():
        val81 = globals()[name]
        print(name, type(val81).__name__,
              tuple(val81.shape) if isinstance(val81, (np.ndarray, torch.Tensor, pd.DataFrame))
              else list(val81)[:12] if isinstance(val81, dict) else repr(val81)[:150])
print("s4 first bundles:",
      pd.DataFrame({k: v[:8] for k, v in s4.items() if k != "x"}).to_string(index=False))
print("train data objects:",
      [(k, tuple(v.shape), list(v.columns)[:8])
       for k, v in globals().items()
       if isinstance(v, pd.DataFrame) and "train" in k.lower()])

# Check whether finite action changes produce a signal that latent changes did not.
saved_U81 = U4.clone() if torch.is_tensor(U4) else U4.copy()
j81 = int(sample80[0])
z81 = s4["x"][j81, 960:992][None, :]
h81 = local_step67(j81, z81, **step_options68)
m81, e81 = next_means_crn74(j81, h81, nsamples=32, seed=913)
print("Baseline diagnostic bundle:", j81, "events:", e81, "means:", m81)
print("Current edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 80: corrected latent-response audit\n"
            "- With shared random draws, posterior-latent Jacobian norms were "
            "roughly 0.0001–0.0009 reading units per latent unit.\n"
            "- Earlier large latent-response directions were sampling artifacts "
            "and are rejected. No edits.\n"
            "- Next examine the direct action/recurrent pathway.\n")
```