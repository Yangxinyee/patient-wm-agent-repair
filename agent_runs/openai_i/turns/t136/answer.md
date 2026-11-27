The mismatch was entirely in empty action bundles; active drug vectors match exactly. Dose-change decoding now has slopes near one for both drugs. Next I’ll fit drug-presence gates and expected factual doses, so a correction can respond to dose departures while staying small on ordinary records.

```python
import numpy as np
import torch

dose_B126 = action_augdec123[0.001]
dose_W126 = dose_B126[:192]/action_sd122[:,None]
dose_b126 = dose_B126[192]-action_mu122@dose_W126
dose_hat126 = action_U122@dose_W126+dose_b126

print("=== DRUG-PRESENCE GATES ===")
gate_Y126 = np.array([[any(r["event"]==d for r in rows)
                       for d in ("insulin","norepinephrine")]
                      for rows in action_rows125],dtype=float)
gate_augY126 = np.tile(gate_Y126,(11,1))
gate_X126 = action_augU123/action_sd122
gate_B126 = np.linalg.solve(
    gate_X126[action_augtr123].T@gate_X126[action_augtr123]+0.001*np.eye(192),
    gate_X126[action_augtr123].T@gate_augY126[action_augtr123])
gate_W126 = gate_B126/action_sd122[:,None]
gate_pred126 = action_augU123@gate_W126
gate_te126 = np.tile(action_te122,11)
print("held gate RMSE:",np.round(np.sqrt(np.mean(
    (gate_pred126[gate_te126]-gate_augY126[gate_te126])**2,axis=0)),5))
print("held maximum error:",np.round(np.max(np.abs(
    gate_pred126[gate_te126]-gate_augY126[gate_te126]),axis=0),5))
print("zero-vector gates:",np.zeros(192)@gate_W126)

print("\n=== FACTUAL POLICY-CENTERING TEST ===")
policy_C126 = drug_ctx110.detach().cpu().numpy().astype(float)
policy_mu126 = policy_C126[action_tr122].mean(0)
policy_sd126 = np.maximum(policy_C126[action_tr122].std(0),0.1)
policy_X126 = np.column_stack(((policy_C126-policy_mu126)/policy_sd126,
                              np.ones(len(policy_C126))))
policy_innov126 = np.asarray(repair_targets119[drug_idx110],dtype=float)
policy_mean126 = action_Y122-policy_innov126
policy_decoders126 = {}
for policy_lam126 in (0.1,1.0,10.0,100.0,1000.0):
    policy_B126 = np.zeros((193,2))
    policy_scores126 = []
    for policy_j126 in range(2):
        policy_tr126 = action_tr122 & (gate_Y126[:,policy_j126]>0) & np.isfinite(policy_mean126[:,policy_j126])
        policy_te126 = action_te122 & (gate_Y126[:,policy_j126]>0) & np.isfinite(policy_mean126[:,policy_j126])
        policy_P126 = np.eye(193)*policy_lam126
        policy_P126[-1,-1] = 1e-8
        policy_B126[:,policy_j126] = np.linalg.solve(
            policy_X126[policy_tr126].T@policy_X126[policy_tr126]+policy_P126,
            policy_X126[policy_tr126].T@policy_mean126[policy_tr126,policy_j126])
        policy_pred126 = policy_X126@policy_B126[:,policy_j126]
        policy_rhat126 = dose_hat126[:,policy_j126]-policy_pred126
        policy_true126 = policy_innov126[policy_te126,policy_j126]
        policy_rmse126 = np.sqrt(np.mean(
            (policy_rhat126[policy_te126]-policy_true126)**2))
        policy_r2126 = 1-np.mean(
            (policy_rhat126[policy_te126]-policy_true126)**2)/max(np.var(policy_true126),1e-12)
        policy_scores126.extend([policy_rmse126,policy_r2126,
                                  np.std(policy_rhat126[policy_te126])])
    policy_decoders126[policy_lam126] = policy_B126
    print("ridge",policy_lam126,"ins RMSE/R2/std; NE RMSE/R2/std:",
          np.round(policy_scores126,4))

print("\n=== RETAINED LATENT-DIRECTION OBJECTS ===")
for recovery_n126,recovery_o126 in list(globals().items()):
    if recovery_n126.startswith(("audit_","action_","dose_","gate_","policy_","recovery_")):
        continue
    if isinstance(recovery_o126,(np.ndarray,torch.Tensor)):
        recovery_shape126 = tuple(recovery_o126.shape)
        if recovery_shape126 in ((4035,192),(4035,32),(32,),(6,),(32,6),(6,32)):
            print(recovery_n126,recovery_shape126)
    if recovery_n126.endswith("104"):
        if isinstance(recovery_o126,(np.ndarray,torch.Tensor)):
            print("direction audit:",recovery_n126,tuple(recovery_o126.shape),
                  repr(np.asarray(recovery_o126).reshape(-1)[:6]))
        elif isinstance(recovery_o126,(int,float,str)):
            print("direction audit:",recovery_n126,repr(recovery_o126))

with open("/work/notes.md","a") as f:
    f.write("\n## Python 125–126 — calibrated action decoder and factual centering\n"
            "- Active action vectors agree exactly; cached empty actions use encoded rather than zero vectors.\n"
            "- Augmented held-stay dose-change slopes: insulin 0.9993, NE 0.9908 (ridge .001).\n"
            "- Fitting presence gates (including x0) and pre-action-context factual policy predictors.\n"
            "- No weights changed.\n")
```