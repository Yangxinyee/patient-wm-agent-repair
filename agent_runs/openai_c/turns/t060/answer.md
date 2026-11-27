The saved latent directions have small factual variation relative to their dose sensitivity, especially for norepinephrine. I’ll amplify their joint subspace only in `z_in`, leaving the direct prediction heads unchanged. This tests whether the recurrent update is underusing dose information; a negligible rollout change would refute that explanation.

```python
latent_sensor_matrix = np.column_stack(
    [repair_white[drug]["z"] for drug in DRUGS]
).astype(np.float64)
latent_u, latent_sv, _ = np.linalg.svd(latent_sensor_matrix, full_matrices=False)
latent_candidate_basis = latent_u[:, latent_sv > 0.15 * latent_sv[0]]

z_reference = np.load("/work/fwd_13/states_original.npz")["x"][:, -32:]
print("LATENT SENSOR COSINE", float(latent_sensor_matrix[:, 0] @
                                     latent_sensor_matrix[:, 1]))
print("SUBSPACE SINGULAR VALUES", latent_sv.tolist())
print("RETAINED DIMENSIONS", latent_candidate_basis.shape[1])
print("FACTUAL PROJECTION RMS",
      np.sqrt(np.mean((z_reference @ latent_candidate_basis) ** 2, axis=0)).tolist())
print("DOSE PROJECTIONS",
      (dose_state_coeff[:, -32:] @ latent_candidate_basis).tolist())

def install_latent_gain(gain):
    replay_ops(repair_candidates["numeric_gain4"])
    for vector in latent_candidate_basis.T:
        rank1("z_in.weight", vector.tolist(), float(gain - 1), side="in")

install_latent_gain(21.0)
print("INSTALLED OPERATIONS", ops())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 40: latent-input gain pilot\n"
            "- Installed numeric gain 4 plus gain 21 on the low-factual-variance latent dose-sensor subspace of z_in.weight.\n"
            "- Direct z_head and observation inputs unchanged. This tests posterior-latent participation in recurrent dose effects.\n"
            "- Factual safety and rollout direction remain untested. Forward passes: 19.\n")
```