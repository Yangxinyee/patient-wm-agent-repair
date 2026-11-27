The dose sensors are noisier than the actual dose innovations, so a full correction looks risky. I’ll test a 15% pilot correction on `core.c_out.weight`, retaining numeric gain 4. The correction is designed not to amplify its own sensor signals directly. A small factual-NLL change would justify checking its rollouts; a large change would reject this pathway.

```python
# Save the current numeric-gain-4 candidate in a replayable form.
numeric4_saved_ops = json.loads(Path("/work/ops.json").read_text())
repair_candidates["numeric_gain4"] = numeric4_saved_ops

def replay_ops(specifications):
    clear_ops()
    for specification in specifications:
        if specification["op"] == "scale":
            scale(specification["name"], specification.get("c", specification.get("factor")))
        elif specification["op"] == "rank1":
            vector = specification["vector"]
            if isinstance(vector, str):
                vector = np.load(Path("/work") / vector)
            rank1(specification["name"], vector, specification["alpha"],
                  side=specification.get("side", "out"))
        else:
            raise ValueError(specification)

def orthogonal_reflection_sequence(O):
    B = O.copy()
    factors = []
    for j in range(len(O)):
        v = B[:,j].copy()
        v[j] -= 1
        if np.linalg.norm(v) < 1e-10:
            continue
        v /= np.linalg.norm(v)
        B -= 2*np.outer(v, v @ B)
        factors.append(v)
    assert np.linalg.norm(B-np.eye(len(O))) < 1e-8
    return factors[::-1]

def apply_lowrank_left_transform(name, basis, reduced_transform):
    U, singular_values, Vt = np.linalg.svd(reduced_transform)
    sequence = []
    for v in orthogonal_reflection_sequence(Vt):
        sequence.append((v, -2.0))
    for j, sigma in enumerate(singular_values):
        if abs(sigma-1) > 1e-10:
            v = np.eye(len(singular_values))[:,j]
            sequence.append((v, float(sigma-1)))
    for v in orthogonal_reflection_sequence(U):
        sequence.append((v, -2.0))
    reconstructed = np.eye(len(reduced_transform))
    for v, alpha in sequence:
        reconstructed += alpha*np.outer(v, v @ reconstructed)
    assert np.linalg.norm(reconstructed-reduced_transform) < 1e-8
    for v, alpha in sequence:
        rank1(name, basis @ v, alpha, side="out")
    return len(sequence)

pilot_strength = 0.15
pilot_design = state_correction_designs[0.01]
pilot_basis, _ = np.linalg.qr(
    np.column_stack([pilot_design["T"], pilot_design["R"]]), mode="reduced")
pilot_reduced = np.eye(pilot_basis.shape[1]) + pilot_strength * (
    pilot_basis.T @ pilot_design["S"] @ pilot_basis)

replay_ops(numeric4_saved_ops)
n_pilot_ops = apply_lowrank_left_transform(
    "core.c_out.weight", pilot_basis, pilot_reduced)
repair_candidates["slot_shear_015"] = json.loads(Path("/work/ops.json").read_text())
print("Additional rank-one operations:", n_pilot_ops)
print("Reduced transform singular values:", np.linalg.svd(pilot_reduced, compute_uv=False))
print("Approximate one-application factual reading-shift SD:",
      pilot_strength*pilot_design["predicted_shift"].std(axis=0))
print("Current operations:", ops())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 24: candidate D, selective slot shear\n")
    f.write("- Applied numeric gain 4 plus a 15%-strength, rank-two left transformation "
            "of core.c_out.weight, factored exactly into allowed rank-one operations.\n")
    f.write("- Based on dose-innovation state coefficients and reading surrogates; "
            "the output directions are orthogonal to the dose sensors (nilpotent correction).\n")
    f.write("- Full-strength sensor noise was too high; this is a small empirical pilot, "
            "not a validated causal repair.\n")
    f.write("- Next test factual NLL. Python runs: 24; forward passes: 13.\n")
```