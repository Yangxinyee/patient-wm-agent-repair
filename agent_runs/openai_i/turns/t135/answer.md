Both atom helpers produced the same mismatch, suggesting the difference may be confined to action-free END bundles. I’ll separate active and empty bundles and test both empty-vector conventions. Matching active bundles would let us proceed with dose calibration without confusing it with END handling.

```python
import numpy as np
import torch

action_rows125 = [[r for r in c[1]
                   if r["event"] in ("insulin","norepinephrine")]
                  for c in drug_cases110]
action_has125 = np.array([bool(r) for r in action_rows125])
action_options125 = {}
print("=== ACTIVE VERSUS EMPTY ACTION CHECK ===")
for action_mode125 in ("zero","encoded"):
    action_probe125 = torch.stack([
        pool109([latent_start_atom(r) for r in rows],
                "encode_action", action_mode125)
        for rows in action_rows125]).detach().cpu().numpy()
    action_options125[action_mode125] = action_probe125
    action_er125 = np.sqrt(np.mean((action_probe125-action_U122)**2,axis=1))
    print(action_mode125,
          "active RMS", np.sqrt(np.mean(action_er125[action_has125]**2)),
          "empty RMS", np.sqrt(np.mean(action_er125[~action_has125]**2)),
          "first 12", np.round(action_er125[:12],4))
action_mode125 = min(action_options125,
    key=lambda m: np.mean((action_options125[m]-action_U122)**2))
action_match125 = np.sqrt(np.mean(
    (action_options125[action_mode125]-action_U122)**2))
print("selected:", action_mode125, "overall RMS:", action_match125)
assert action_match125 < 1e-5

action_edit_U123, action_edit_Y123 = {}, {}
for action_drug125 in ("insulin","norepinephrine"):
    action_col125 = ("insulin","norepinephrine").index(action_drug125)
    for action_factor125 in (0.0,0.5,1.5,2.0,3.0):
        action_vectors125 = []
        for action_rows_i125 in action_rows125:
            action_atoms125 = []
            for action_row125 in action_rows_i125:
                action_r125 = dict(action_row125)
                if action_r125["event"] == action_drug125:
                    for action_field125 in ("rate","dose"):
                        action_r125[action_field125] = float(action_r125[action_field125])*action_factor125
                action_atoms125.append(latent_start_atom(action_r125))
            action_vectors125.append(pool109(action_atoms125,"encode_action",action_mode125))
        action_key125 = (action_drug125,action_factor125)
        action_edit_U123[action_key125] = torch.stack(action_vectors125).detach().cpu().numpy()
        action_edit_Y123[action_key125] = action_Y122.copy()
        action_edit_Y123[action_key125][:,action_col125] *= action_factor125

action_augU123 = np.concatenate([action_U122]+list(action_edit_U123.values()))
action_augY123 = np.concatenate([action_Y122]+list(action_edit_Y123.values()))
action_augtr123 = np.tile(action_tr122,1+len(action_edit_U123))
action_augX123 = np.column_stack(((action_augU123-action_mu122)/action_sd122,
                                 np.ones(len(action_augU123))))
action_augdec123 = {}
print("\n=== HELD-STAY EDIT SLOPE / RMSE / CROSS-DRUG RMSE ===")
for action_lam125 in (0.001,0.01,0.1,1.0):
    action_pen125 = np.eye(193)*action_lam125
    action_pen125[-1,-1] = 1e-8
    action_B125 = np.linalg.solve(
        action_augX123[action_augtr123].T@action_augX123[action_augtr123]+action_pen125,
        action_augX123[action_augtr123].T@action_augY123[action_augtr123])
    action_augdec123[action_lam125] = action_B125
    action_base125 = action_X122@action_B125
    action_metrics125 = []
    for action_col125,action_drug125 in enumerate(("insulin","norepinephrine")):
        action_p125,action_t125,action_c125 = [],[],[]
        for action_key125,action_U125 in action_edit_U123.items():
            if action_key125[0] != action_drug125:
                continue
            action_X125 = np.column_stack(((action_U125-action_mu122)/action_sd122,np.ones(len(action_U125))))
            action_dp125 = action_X125@action_B125-action_base125
            action_dt125 = action_edit_Y123[action_key125]-action_Y122
            action_mask125 = action_te122 & (action_Y122[:,action_col125]>0)
            action_p125.extend(action_dp125[action_mask125,action_col125])
            action_t125.extend(action_dt125[action_mask125,action_col125])
            action_c125.extend(action_dp125[action_mask125,1-action_col125])
        action_p125,action_t125 = np.asarray(action_p125),np.asarray(action_t125)
        action_metrics125.extend([action_p125@action_t125/(action_t125@action_t125),
            np.sqrt(np.mean((action_p125-action_t125)**2)),
            np.sqrt(np.mean(np.asarray(action_c125)**2))])
    print(action_lam125,np.round(action_metrics125,4))

print("\n=== INPUT / DIRECTION OBJECTS ===")
for action_n125,action_o125 in list(globals().items()):
    if action_n125.startswith(("audit_","action_")):
        continue
    if hasattr(action_o125,"shape") and tuple(action_o125.shape) in (
            (4035,192),(4035,32),(32,),(6,),(32,6),(6,32)):
        print(action_n125,tuple(action_o125.shape))

with open("/work/notes.md","a") as f:
    f.write("\n## Python 125 — active/empty action distinction\n"
            "- Compared cached vectors with zero and encoded empty-action conventions.\n"
            "- Calibrated held-stay dose shifts only after exact vector agreement.\n"
            "- No weights changed.\n")
```