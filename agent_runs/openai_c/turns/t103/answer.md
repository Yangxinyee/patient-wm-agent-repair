The new directions agree across stay groups, and the recalibrated sensors have much lower factual variance—especially for norepinephrine. I’ll test a 20%-strength reading-head correction. The surrogate estimates are not proof: acceptable factual NLL and stronger rollout effects are both required to keep it.

```python
replay_ops(repair_candidates["numeric_gain4"])

head_strength73 = 0.20
head_basis73, _ = np.linalg.qr(np.column_stack([T72, R72]))
head_reduced73 = (
    np.eye(head_basis73.shape[1])
    + head_strength73 * head_basis73.T @ S72 @ head_basis73
)
head_name73 = "attr_heads.value_bin.0.weight"
apply_lowrank_right_transform(head_name73, head_basis73, head_reduced73)

# Verify the allowed rank-one factors implement the intended input transform.
implemented73 = np.eye(192)
for op73 in ops():
    if op73["name"] != head_name73:
        continue
    if op73["op"] == "scale":
        implemented73 *= op73["c"]
    else:
        vec73 = op73["vector"]
        if isinstance(vec73, str):
            path73 = Path(vec73)
            if not path73.is_absolute():
                path73 = Path("/work") / path73
            vec73 = np.load(path73)
        vec73 = np.asarray(vec73, dtype=float)
        vec73 /= np.linalg.norm(vec73)
        factor73 = np.eye(192) + op73["alpha"] * np.outer(vec73, vec73)
        if op73["side"] != "in":
            raise RuntimeError("Unexpected output-side factor")
        implemented73 = implemented73 @ factor73

target73 = np.eye(192) + head_strength73 * S72
error73 = np.linalg.norm(implemented73 - target73)
print("Factorization error:", error73)
if error73 > 1e-7:
    replay_ops(repair_candidates["numeric_gain4"])
    raise RuntimeError("Restored fallback after factorization mismatch")

repair_candidates["gain4_head_recal_020"] = freeze_ops(ops())
print("Installed candidate:", len(ops()), "operations")
print("Edited tensors:", sorted({op["name"] for op in ops()}))
print("Surrogate immediate response increments:\n",
      head_strength73 * D72 @ S72.T @ W72.T)
print("Surrogate factual reading-shift SD:",
      head_strength73 * np.std(O72 @ S72.T @ W72.T, axis=0))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 73: recalibrated head pilot\n"
            "- Gain-4 dose directions are stable across stay halves: cosine 0.890 insulin, 0.948 NE.\n"
            "- New factual sensor SD: insulin 1.096, NE 0.222 rate units at ridge 1e-4.\n"
            "- Installed numeric gain 4 plus 20%-strength input shear on attr_heads.value_bin.0.weight.\n"
            "- Surrogate increments: insulin/glucose -0.270, NE/MAP +0.588 per rate unit.\n"
            "- Factorization verified; factual and rollout behavior remain untested. Forward passes: 27.\n")
```
