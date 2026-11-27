The slots determine `out` almost exactly, so we can estimate how a slot change would affect each reading. I’ll use that link to design a correction driven by low-variance dose signals, with separate glucose and MAP outputs. I’ll check its estimated factual disturbance before testing it; large disturbance or inseparable drug signals would argue against this approach.

```python
import inspect
from scipy.linalg import null_space

print("State coefficient function:", inspect.signature(state_dose_coefficients))
print("Stored h-direction diagnostics:")
for drug_name in DRUGS:
    print(drug_name, repair_signal[drug_name]["hmean"])

D_h = dose_state_coeff[:,192:960].reshape(2,4,192).mean(axis=1)
D_out = dose_state_coeff[:,:192]
G_h = []
for event in ["glucose_finger", "map_arterial"]:
    q = norm_gamma_est * head_surrogates[event]["w"]
    # Jacobian of the inferred normalization, averaged over factual states.
    grad = (q[None,:] - q.mean()
            - h_normalized * (h_normalized @ q)[:,None]/192) / h_rms
    G_h.append(grad.mean(axis=0))
G_h = np.asarray(G_h)
print("\nDose coefficient norms:", np.linalg.norm(D_h,axis=1))
print("Approximate immediate response, rows glucose/MAP, columns insulin/NE:")
print(G_h @ D_h.T)
print("Corresponding out-surrogate response:")
print(np.stack([head_surrogates[e]["w"] for e in
                ["glucose_finger","map_arterial"]]) @ D_out.T)

# Sensors have zero projection on the factual mean and the all-ones direction.
sensor_basis = null_space(np.stack([h_mean_o.mean(axis=0), np.ones(192)]))
HC = h_mean_o @ sensor_basis
C_reduced = HC.T @ HC / len(HC)
D_reduced = D_h @ sensor_basis
state_correction_designs = {}
for ridge_level in [0.01, 0.1, 1.0, 10.0]:
    CRD = np.linalg.solve(C_reduced + ridge_level*np.eye(C_reduced.shape[0]),
                         D_reduced.T)
    R_sensor = sensor_basis @ CRD @ np.linalg.inv(D_reduced @ CRD)
    # Keep the injected output perpendicular to both dose sensors.
    P_sensor_null = np.eye(192) - R_sensor @ np.linalg.solve(
        R_sensor.T @ R_sensor, R_sensor.T)
    G_null = G_h @ P_sensor_null
    T_output = G_null.T @ np.linalg.inv(G_null @ G_null.T)
    delta_response = np.diag([-1.5, 2.5])
    S_correction = T_output @ delta_response @ R_sensor.T
    sensed = h_mean_o @ R_sensor
    predicted_shift = sensed @ delta_response.T
    state_correction_designs[ridge_level] = {
        "R": R_sensor, "T": T_output, "S": S_correction,
        "predicted_shift": predicted_shift
    }
    print("\nRidge", ridge_level, {
        "sensor_response": np.round(R_sensor.T @ D_h.T,5).tolist(),
        "sensor_sd": np.round(sensed.std(axis=0),4).tolist(),
        "predicted_factual_shift_sd": np.round(predicted_shift.std(axis=0),4).tolist(),
        "matrix_spectral_norm": round(float(np.linalg.norm(S_correction,2)),5),
        "nilpotence_error": round(float(np.linalg.norm(S_correction@S_correction)),10),
        "sensor_means_by_bundle": {
            k: np.round(sensed[m].mean(axis=0),4).tolist()
            for k,m in drug_masks.items() if m.any()
        }
    })

print("\nExisting operation specification:", Path("/work/ops.json").read_text()[:900])
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 23: calibrated slot shear design\n")
    f.write("- Designed separate minimum-variance insulin/NE sensors with zero factual mean, "
            "and reading-output directions orthogonal to the sensors.\n")
    f.write("- Candidate matrices add approximate immediate responses -1.5 insulin/glucose and "
            "+2.5 NE/MAP at full strength; approximation must be checked by forward passes.\n")
    f.write("- No additional edits applied. Python runs: 23; forward passes: 13.\n")
```