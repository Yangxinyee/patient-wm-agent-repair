The pilot now produces substantial effects, but the pattern is uneven: insulin doubling is slightly too strong, insulin halving is far too weak, and norepinephrine doubling remains weak. I’ll examine the hour-by-hour responses and alternate-sensor coupling. If contradictory sensor responses are disrupting rollouts, directions aligned with both sensors should help; otherwise the dose signal itself needs work.

```python
import pandas as pd
print("=== ROLLOUT FILE STRUCTURE ===")
roll193 = pd.read_csv("/work/fwd_12/dose_edit_edited.csv")
print(roll193.columns.tolist())
print(roll193.head(3).to_string(index=False))
print("=== FACTUAL FILE STRUCTURE ===")
fact193 = pd.read_csv("/work/fwd_11/factual_edited.csv")
print(fact193.columns.tolist())
print(fact193.head(2).to_string(index=False))

print("\n=== SENSOR CALIBRATION CONDITIONING ===")
print("Sensor order:", codes171)
print("J singular values:", np.linalg.svd(J186, compute_uv=False))
print("Full-calibration direction norms:", np.linalg.norm(directions186,axis=0))
print("Primary-calibration direction norms:", np.linalg.norm(directions187,axis=0))

# Preserve the validated pilot before any refinement.
checkpoint193 = {k:load(k).clone() for k in edited()}
jp193 = J186[[1,2]]
jo193 = J186[[0,3]]
proj193 = np.eye(192) - jp193.T @ np.linalg.solve(jp193@jp193.T, jp193)
b193 = jo193@proj193
target_other193 = np.array([[-1.94,0.0],[0.0,2.7]])
residual193 = target_other193-jo193@directions187
scale193 = max(float(np.trace(b193@b193.T)/2),1e-14)
options193 = {}
print("\n=== PRIMARY-PRESERVING ALTERNATE-SENSOR DIRECTIONS ===")
for reg193 in [0.0,0.01,0.1,1.0,10.0]:
    correction193 = proj193@b193.T@np.linalg.solve(
        b193@b193.T+(reg193*scale193+1e-14)*np.eye(2), residual193)
    dd193 = directions187+correction193
    options193[reg193] = dd193
    print("ridge", reg193, "norms", np.linalg.norm(dd193,axis=0).round(1),
          "linear sensors", (J186@dd193).round(3).tolist())
print("\n=== NONLINEAR CHECKS ===")
for reg193 in [0.1,1.0,10.0]:
    dd193 = options193[reg193]
    pred193 = response186(np.stack([dd193[:,0],-dd193[:,0],
                                   .5*dd193[:,1],-.5*dd193[:,1]]))
    print("ridge",reg193,"shifts",np.round(pred193-response186(np.zeros((1,192))),3))

hour_col193 = next((c for c in roll193.columns if c in ["hour","stay_hour","icu_hour"]),None)
edit_col193 = next((c for c in roll193.columns if c in ["edit","intervention"]),None)
shift_col193 = next((c for c in roll193.columns if c in ["shift","mean_shift","predicted_shift"]),None)
if hour_col193 and edit_col193 and shift_col193:
    rr193 = roll193
    if "reached" in rr193:
        rr193 = rr193[rr193["reached"].astype(bool)]
    print("\n=== MEAN SHIFTS BY HOUR ===")
    print(rr193.pivot_table(index=hour_col193,columns=edit_col193,
                           values=shift_col193).round(2).to_string())
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 193; forwards 12. Pilot hash 3161239253 factual NLL 1.64438 vs 1.63982 (+.00456). Actual rollout shifts: insulin x2 -15.255, half +1.669; NE x2 +2.075, half -1.991. First meaningful transport success, but dose asymmetry and alternate-sensor errors need correction. Saved checkpoint193; investigating primary-preserving multi-sensor directions and temporal rollout shape. Current pilot unchanged.\n")
```
