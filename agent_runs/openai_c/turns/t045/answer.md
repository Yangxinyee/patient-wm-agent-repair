The normalization designs require large changes, so I won’t apply them. I’ll instead pilot a small reading-head shear: a low-variance norepinephrine sensor coupled to a MAP output direction, with no predicted first-order insulin or glucose effect. A safe factual result followed by stronger MAP dose response would support this approach.

```python
from scipy.linalg import solve

# A minimum-variance NE sensor in the observable reading-state coordinates.
Oall = np.vstack([Og, Om])
Chead = Oall.T @ Oall / len(Oall)
constraints = np.column_stack([
    Ddiag[:, 1], Ddiag[:, 0],
    Og.mean(axis=0), Om.mean(axis=0), np.ones(192)
])
reg_level = 0.01 * np.trace(Chead) / 192
CA = solve(Chead + reg_level * np.eye(192), constraints, assume_a="pos")
sensor_ne = CA @ np.linalg.pinv(constraints.T @ CA, rcond=1e-10) @ np.array([1., 0., 0., 0., 0.])

# Output direction: +0.5 approximate MAP units per sensed NE rate unit,
# zero approximate glucose effect, orthogonal to the sensor.
output_constraints = np.stack([wg, wm, sensor_ne, np.ones(192)])
output_ne = (
    output_constraints.T
    @ np.linalg.pinv(output_constraints @ output_constraints.T, rcond=1e-12)
    @ np.array([0., 0.5, 0., 0.])
)
print("Sensor constraints:", np.round(constraints.T @ sensor_ne, 6))
print("Output constraints:", np.round(output_constraints @ output_ne, 6))
print("Sensor/output norms:", np.linalg.norm(sensor_ne), np.linalg.norm(output_ne))
print("Predicted factual MAP SD:",
      0.5 * np.std(Om @ sensor_ne))
print("Predicted sensor means glucose/MAP:",
      np.mean(Og @ sensor_ne), np.mean(Om @ sensor_ne))

def orthogonal_rank1_factors(R):
    R = np.asarray(R, dtype=float)
    work = R.copy()
    factors = []
    for i in range(len(R)):
        difference = work[i:, i].copy()
        difference[0] -= 1.0
        length = np.linalg.norm(difference)
        if length < 1e-10:
            continue
        v = np.zeros(len(R))
        v[i:] = difference / length
        work -= 2.0 * np.outer(v, v @ work)
        factors.append((v, -2.0))
    assert np.max(np.abs(work - np.eye(len(R)))) < 1e-8
    return factors

def apply_lowrank_right_transform(name, basis, reduced_transform):
    U, singular, Vt = np.linalg.svd(reduced_transform)
    factors = orthogonal_rank1_factors(U)
    factors += [(np.eye(len(singular))[:, i], float(singular[i] - 1))
                for i in range(len(singular))]
    factors += orthogonal_rank1_factors(Vt)
    check = np.eye(len(singular))
    for v, alpha in factors:
        check = check @ (np.eye(len(singular)) + alpha * np.outer(v, v))
    assert np.max(np.abs(check - reduced_transform)) < 1e-8
    for v, alpha in factors:
        if abs(alpha) > 1e-12:
            rank1(name, basis @ v, alpha, side="in")

head_ne_basis, _ = np.linalg.qr(np.column_stack([output_ne, sensor_ne]))
head_ne_reduced = (
    np.eye(2)
    + np.outer(head_ne_basis.T @ output_ne, head_ne_basis.T @ sensor_ne)
)
head_ne_design = {
    "sensor": sensor_ne.copy(), "output": output_ne.copy(),
    "basis": head_ne_basis.copy(), "reduced": head_ne_reduced.copy()
}
replay_ops(repair_candidates["numeric_gain4"])
apply_lowrank_right_transform(
    "attr_heads.value_bin.0.weight", head_ne_basis, head_ne_reduced
)
repair_candidates["head_ne_pilot"] = freeze_ops(ops())
print("Applied NE reading-head pilot:", ops())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python run 30: direct NE reading-head pilot\n"
        "- Rejected large diagonal out_norm designs without applying them.\n"
        "- Applied numeric gain 4 plus a rank-one input shear of value_bin.0.weight, "
        "factored into allowed rank-one operations.\n"
        "- Sensor minimizes factual second moment with NE sensitivity 1, insulin sensitivity 0, "
        "and zero means in both reading-state samples. Output targets MAP +0.5, glucose 0; "
        "sensor/output orthogonality makes the shear nilpotent.\n"
        "- Head input also contains unobserved code/register terms; factual and rollout passes "
        "must validate the surrogate approximation.\n"
        "- Python runs: 30; forward passes: 15.\n"
    )
```