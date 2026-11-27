The diagonal pilot raised factual NLL by 0.06917, far beyond the 0.005 limit, so I’ll reject it and restore numeric gain 4. This also shows that the observational readout fits badly underestimated its impact. Before trying another direction, I’ll verify that the earlier factored shears implemented the intended transformations; a mismatch could explain their weak results.

```python
replay_ops(repair_candidates["numeric_gain4"])

def effective_transform(saved_ops, tensor_name, width, side):
    T = np.eye(width)
    for operation in saved_ops:
        if operation["name"] != tensor_name:
            continue
        v = np.asarray(operation["vector"], dtype=float)
        v /= np.linalg.norm(v)
        F = np.eye(width)+float(operation["alpha"])*np.outer(v,v)
        T = F@T if side=="out" else T@F
    return T

head_actual = effective_transform(repair_candidates["head_ne_pilot"],
                                 "attr_heads.value_bin.0.weight",192,"in")
head_goal = np.eye(192)+np.outer(head_ne_design["output"],head_ne_design["sensor"])
head_basis_goal = (np.eye(192)-head_ne_basis@head_ne_basis.T+
                   head_ne_basis@head_ne_reduced@head_ne_basis.T)
print("HEAD FACTORIZATION ERRORS",
      {"direct":float(np.linalg.norm(head_actual-head_goal)),
       "transpose":float(np.linalg.norm(head_actual-head_goal.T)),
       "saved_reduced":float(np.linalg.norm(head_actual-head_basis_goal))})

mix_actual = effective_transform(repair_candidates["numeric_value_mix_m2"],
                                "embed.numeric.weight",6,"in")
gain4_transform = np.eye(6)+3*np.outer(numeric_dose_axis,numeric_dose_axis)
mix_only = np.linalg.solve(gain4_transform,mix_actual)
print("NUMERIC MIX ONLY", np.round(mix_only,6).tolist())
print("SAVED NUMERIC MIX", np.round(numeric_mix_transform,6).tolist())

print("PILOT DESIGN KEYS", summarize_saved(pilot_design))
print("LATENT FACTUAL PROJECTION SD",
      np.std(z_reference@latent_candidate_basis,axis=0).tolist())
print("CURRENT OPERATION COUNT",len(ops()))
with open("/work/notes.md","a") as f:
    f.write("\n## Forward 22 / Python run 48: diagonal pilot rejected\n"
            "- out_norm diagonal 0.10 factual NLL 1.70378 vs 1.63461, increase +0.06917; rejected without a rollout pass.\n"
            "- Restored numeric gain 4. Observational readout coefficients are not reliable intervention gradients.\n"
            "- Auditing factored input/output shears before selecting the next pathway.\n"
            "- Forward passes used: 22.\n")
```
