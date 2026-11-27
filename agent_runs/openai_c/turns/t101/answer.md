The saved corrections appear to use drug directions estimated from the original model, although they were applied with numeric gain 4 installed. That mismatch could explain their weak effects. I’ll compare conditional drug directions before and after the gain. A substantial, stable change would justify recalibrating the correction; similar directions would refute this explanation.

```python
from scipy.linalg import solve

print("=== ORIGINAL STATE ESTIMATOR ===")
try:
    print(inspect.getsource(state_dose_coefficients)[:2200])
except Exception:
    print("Referenced names:", state_dose_coefficients.__code__.co_names)
    print("Local names:", state_dose_coefficients.__code__.co_varnames)

print("\n=== INPUT CHECKS ===")
for name in ["xo", "xe", "state_C", "innovation", "sd", "valid_actions"]:
    a = np.asarray(globals()[name])
    print(name, a.shape, "finite:", np.isfinite(a).all())
print("Action rows:", int(np.asarray(valid_actions).sum()))
print("Innovation SD on action rows:",
      np.asarray(innovation)[valid_actions].std(axis=0))

# Identical conditional regression for original and gain-4 action transitions.
m71 = np.asarray(valid_actions, dtype=bool)
C71 = np.asarray(state_C[m71], dtype=np.float64)
D71 = np.asarray(innovation[m71], dtype=np.float64)
prev_o71 = np.vstack([np.zeros((1, 992)), np.asarray(xo[:-1])])
prev_e71 = np.vstack([np.zeros((1, 992)), np.asarray(xe[:-1])])
Yo71 = (np.asarray(xo) - prev_o71)[m71].astype(np.float64)
Ye71 = (np.asarray(xe) - prev_e71)[m71].astype(np.float64)

A71 = C71.T @ C71 + 20.0 * np.eye(C71.shape[1])
Dc71 = D71 - C71 @ solve(A71, C71.T @ D71, assume_a="pos")
dose_coeff_compare71 = {}
for label, Y71 in [("original", Yo71), ("gain4", Ye71)]:
    Yc71 = Y71 - C71 @ solve(A71, C71.T @ Y71, assume_a="pos")
    dose_coeff_compare71[label] = np.linalg.lstsq(Dc71, Yc71, rcond=1e-10)[0]

print("\n=== CONDITIONAL DIRECTION COMPARISON ===")
for drug_i, drug in enumerate(["insulin", "norepinephrine"]):
    for part, lo, hi in [("out", 0, 192), ("h", 192, 960), ("z", 960, 992)]:
        bo = dose_coeff_compare71["original"][drug_i, lo:hi]
        be = dose_coeff_compare71["gain4"][drug_i, lo:hi]
        cosine = float(bo @ be / max(np.linalg.norm(bo) * np.linalg.norm(be), 1e-15))
        print(drug, part,
              "original norm", round(float(np.linalg.norm(bo)), 5),
              "gain4 norm", round(float(np.linalg.norm(be)), 5),
              "cosine", round(cosine, 4))
    for event in ["glucose_finger", "map_arterial"]:
        w71 = head_surrogates[event]["w"]
        print(drug, event, "surrogate response original/gain4:",
              float(dose_coeff_compare71["original"][drug_i, :192] @ w71),
              float(dose_coeff_compare71["gain4"][drug_i, :192] @ w71))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 71: alignment audit\n"
            "- Numeric gain 4 retained. Comparing conditional drug-state directions under original and gain-4 weights.\n"
            "- Earlier selective corrections used original-state directions; checking whether this mismatch explains failure.\n"
            "- No new forward pass; passes used: 27.\n")
```
