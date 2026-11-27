The mismatch is substantial: under gain 4, the norepinephrine direction is nearly orthogonal to the old one and about 3.6 times larger. I’ll check whether that direction is stable across stays and build a factual-centered sensor from the current states. Low sensor variance and agreement across stay groups would support a new targeted pilot.

```python
# Check whether the changed directions survive a stay-level split.
def conditional_transition_slopes72(rows, Y):
    c = np.asarray(state_C[rows], dtype=np.float64)
    c = (c - c.mean(axis=0)) / np.maximum(c.std(axis=0), 1e-4)
    c = np.column_stack([np.ones(len(c)), c])
    d = np.asarray(innovation[rows], dtype=np.float64)
    a = c.T @ c + 20.0 * np.eye(c.shape[1])
    dc = d - c @ np.linalg.solve(a, c.T @ d)
    yc = Y[rows] - c @ np.linalg.solve(a, c.T @ Y[rows])
    return np.linalg.lstsq(dc, yc, rcond=1e-10)[0]

gain4_transition72 = np.asarray(xe - prev_e71, dtype=np.float64)
ids72 = np.unique(ss)
fold_coeff72 = []
for half in [ids72[:len(ids72)//2], ids72[len(ids72)//2:]]:
    rows72 = m71 & np.isin(ss, half)
    fold_coeff72.append(conditional_transition_slopes72(rows72, gain4_transition72))
dose_gain4_72 = conditional_transition_slopes72(m71, gain4_transition72)

print("=== STAY-SPLIT DIRECTION STABILITY ===")
for j72, drug72 in enumerate(["insulin", "norepinephrine"]):
    a72, b72 = fold_coeff72[0][j72, :192], fold_coeff72[1][j72, :192]
    print(drug72, "out-direction cosine:",
          float(a72 @ b72 / (np.linalg.norm(a72) * np.linalg.norm(b72))),
          "norms:", float(np.linalg.norm(a72)), float(np.linalg.norm(b72)))

# Use gain-4 states immediately preceding factual readings.
ix72 = np.asarray(state_for_reading, dtype=int)
valid72 = ix72 >= 0
O72 = np.asarray(xe[np.maximum(ix72, 0), :192], dtype=np.float64)
events72 = pred_train["event"].to_numpy()
means72 = []
for ev72 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    evmask72 = valid72 & (events72 == ev72)
    means72.append(O72[evmask72].mean(axis=0))
O72 = O72[valid72]
D72 = dose_gain4_72[:, :192]

# Remove the exact layer-normalization null direction as well as event means.
ln_null72 = 1.0 / np.asarray(norm_gamma_est)
ln_null72 /= np.linalg.norm(ln_null72)
A72 = np.column_stack([D72.T, *means72, ln_null72])
B72 = np.zeros((A72.shape[1], 2))
B72[:2] = np.eye(2)
cov72 = O72.T @ O72 / len(O72)
sensor_designs72 = {}

print("\n=== CURRENT-STATE SENSOR VARIANCE ===")
for ridge72 in [1e-5, 1e-4, 1e-3, 1e-2]:
    ca72 = np.linalg.solve(cov72 + ridge72 * np.eye(192), A72)
    r72 = ca72 @ np.linalg.pinv(A72.T @ ca72, rcond=1e-10) @ B72
    sensor_designs72[ridge72] = r72
    print("ridge", ridge72,
          "constraint error", float(np.max(np.abs(A72.T @ r72 - B72))),
          "factual SD", np.std(O72 @ r72, axis=0).round(4),
          "sensor norms", np.linalg.norm(r72, axis=0).round(4))

R72 = sensor_designs72[1e-4]
P72 = np.eye(192) - R72 @ np.linalg.pinv(R72.T @ R72) @ R72.T
W72 = np.stack([head_surrogates[e]["w"]
                for e in ["glucose_finger", "map_arterial"]])
desired72 = np.diag([-1.35, 2.94])
T72 = P72 @ W72.T @ np.linalg.pinv(W72 @ P72 @ W72.T) @ desired72
S72 = T72 @ R72.T

print("\n=== FULL-STRENGTH SURROGATE CHECK ===")
print("Drug/output response:\n", D72 @ S72.T @ W72.T)
print("Output-direction norms:", np.linalg.norm(T72, axis=0))
print("Approximate factual reading-shift SD:",
      np.std(O72 @ S72.T @ W72.T, axis=0))
print("Nilpotence error:", np.linalg.norm(R72.T @ T72))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 72: gain-4 direction recalibration\n"
            "- Gain 4 changes the NE out direction markedly (old/new cosine -0.070 in run 71).\n"
            "- Checking stay-split stability and minimum-factual-variance sensors using gain-4 states.\n"
            "- No new edits or forward passes; passes used: 27.\n")
```
