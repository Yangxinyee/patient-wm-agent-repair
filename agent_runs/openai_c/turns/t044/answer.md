A direction measured in recurrent state need not work inside the nonlinear reading head. The final state normalization is reconstructible, though. I’ll assess centered coordinate-wise edits there, designed to change dose sensitivity with minimal factual disturbance. Small required edits and low predicted reading noise would justify a forward test.

```python
from scipy.linalg import solve

print("=== NORMALIZATION RECONSTRUCTION ===")
for name in ["norm_gamma_est", "norm_beta_est", "D_out", "D_h",
             "paired_o", "paired_e", "state_dose_coefficients",
             "valid_actions", "predicted_dose_states", "innovation"]:
    if name in globals():
        obj = globals()[name]
        print(name, type(obj).__name__, getattr(obj, "shape", ""),
              compact_saved(obj) if isinstance(obj, np.ndarray) and obj.ndim <= 1 else "")
print("dose_models:", compact_saved(dose_models))
for event, obj in head_surrogates.items():
    print(event, "frame columns:", list(obj["frame"].columns))
    print(obj["frame"].head(2).to_string(index=False))
print("states keys:", list(state_file.keys()))

wg = head_surrogates["glucose_finger"]["w"]
wm = head_surrogates["map_arterial"]["w"]
Og = head_surrogates["glucose_finger"]["O"]
Om = head_surrogates["map_arterial"]["O"]
Cg = np.cov(Og, rowvar=False, bias=True)
Cm = np.cov(Om, rowvar=False, bias=True)
Ddiag = np.asarray(D_out)
if Ddiag.shape == (2, 192):
    Ddiag = Ddiag.T
assert Ddiag.shape == (192, 2)

Bdiag = np.stack([
    wg * Ddiag[:, 0], wg * Ddiag[:, 1],
    wm * Ddiag[:, 0], wm * Ddiag[:, 1]
])
Qdiag = Cg * np.outer(wg, wg) / 20.0**2 + Cm * np.outer(wm, wm) / 6.0**2
diag_designs = {}
for ridge_amount in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    QB = solve(Qdiag + ridge_amount * np.eye(192), Bdiag.T,
               assume_a="pos")
    change = QB @ solve(Bdiag @ QB, np.array([-1.5, 0.0, 0.0, 2.5]))
    noise_g = float(np.sqrt((wg * change) @ Cg @ (wg * change)))
    noise_m = float(np.sqrt((wm * change) @ Cm @ (wm * change)))
    diag_designs[ridge_amount] = change
    print("ridge", ridge_amount,
          "change norm/max", round(float(np.linalg.norm(change)), 3),
          round(float(np.max(np.abs(change))), 3),
          "predicted factual SD glucose/MAP", round(noise_g, 3), round(noise_m, 3),
          "dose-response changes", np.round(Bdiag @ change, 3).tolist())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python run 29\n"
        "- Assessing centered diagonal edits to reconstructible core.out_norm, "
        "rather than assuming c_out shear propagates unchanged through coda.\n"
        "- No new edits. Forward passes: 15.\n"
    )
```