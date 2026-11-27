The presence gates are accurate, and insulin centering works reasonably well. Norepinephrine centering is weaker, so I’ll start with a smaller correction there. I’ll make a reversible four-neuron posterior edit, compensate for the displaced neurons, and then check factual NLL and paired dose responses.

```python
import numpy as np
import torch
import torch.nn.functional as repair_F127
import pandas as pd
import json

repair_W0127 = np.asarray(load_original("z_post.0.weight"),dtype=np.float64)
repair_b0127 = np.asarray(load_original("z_post.0.bias"),dtype=np.float64)
repair_W2127 = np.asarray(load_original("z_post.2.weight"),dtype=np.float64)
repair_b2127 = np.asarray(load_original("z_post.2.bias"),dtype=np.float64)

repair_Hparts127 = [np.asarray(repair_H119,dtype=np.float64)]
for repair_key127,repair_U127 in action_edit_U123.items():
    repair_input127 = np.concatenate([
        np.asarray(drug_ctx110),np.asarray(drug_e110),repair_U127],axis=1)
    repair_Hpart127 = repair_F127.gelu(torch.as_tensor(
        repair_input127@repair_W0127.T+repair_b0127,
        dtype=torch.float32)).numpy().astype(np.float64)
    repair_Hparts127.append(repair_Hpart127)
repair_Hcal127 = np.concatenate(repair_Hparts127)
repair_tr127 = np.concatenate([repair_train119]+[action_tr122]*10).astype(bool)
repair_te127 = np.concatenate([repair_test119]+[action_te122]*10).astype(bool)

repair_energy127 = np.var(repair_Hcal127[repair_tr127],axis=0)*np.sum(repair_W2127**2,axis=0)
repair_slots127 = np.argsort(repair_energy127)[:4]
repair_keep127 = np.setdiff1d(np.arange(192),repair_slots127)
repair_old127 = repair_Hcal127[:,repair_slots127]@repair_W2127[:,repair_slots127].T
repair_cm127 = repair_Hcal127[repair_tr127][:,repair_keep127].mean(0)
repair_cs127 = np.maximum(repair_Hcal127[repair_tr127][:,repair_keep127].std(0),0.03)
repair_CX127 = np.column_stack((
    (repair_Hcal127[:,repair_keep127]-repair_cm127)/repair_cs127,
    np.ones(len(repair_Hcal127))))
repair_CP127 = np.eye(len(repair_keep127)+1)*0.01
repair_CP127[-1,-1] = 1e-8
repair_CB127 = np.linalg.solve(
    repair_CX127[repair_tr127].T@repair_CX127[repair_tr127]+repair_CP127,
    repair_CX127[repair_tr127].T@repair_old127[repair_tr127])
repair_comp_error127 = repair_CX127@repair_CB127-repair_old127
print("replaced neurons:",repair_slots127.tolist())
print("held preservation RMS, mean/log-sd:",
      np.sqrt(np.mean(repair_comp_error127[repair_te127,:32]**2)),
      np.sqrt(np.mean(repair_comp_error127[repair_te127,32:]**2)))

repair_compW127 = repair_CB127[:-1]/repair_cs127[:,None]
repair_baseW2127 = repair_W2127.copy()
repair_baseW2127[:,repair_keep127] += repair_compW127.T
repair_baseW2127[:,repair_slots127] = 0
repair_baseb2127 = repair_b2127+repair_CB127[-1]-repair_cm127@repair_compW127

repair_policyB127 = np.column_stack((
    policy_decoders126[100.0][:,0],
    policy_decoders126[1000.0][:,1]))
repair_policyW127 = repair_policyB127[:192]/policy_sd126[:,None]
repair_policyb127 = repair_policyB127[192]-policy_mu126@repair_policyW127
repair_alpha127,repair_gateK127 = 0.1,24.0
repair_newW0127,repair_newb0127 = repair_W0127.copy(),repair_b0127.copy()
for repair_j127 in range(2):
    repair_i127,repair_k127 = repair_slots127[2*repair_j127:2*repair_j127+2]
    repair_gatew127 = np.zeros(576)
    repair_gatew127[384:] = repair_gateK127*gate_W126[:,repair_j127]
    repair_innovw127 = np.zeros(576)
    repair_innovw127[:192] = -repair_policyW127[:,repair_j127]
    repair_innovw127[384:] = dose_W126[:,repair_j127]
    repair_innovb127 = dose_b126[repair_j127]-repair_policyb127[repair_j127]
    repair_newW0127[repair_i127] = repair_gatew127+repair_alpha127*repair_innovw127
    repair_newW0127[repair_k127] = repair_gatew127
    repair_newb0127[repair_i127] = -repair_gateK127/2+repair_alpha127*repair_innovb127
    repair_newb0127[repair_k127] = -repair_gateK127/2

def install_repair127(ins_gain,ne_gain):
    W2 = repair_baseW2127.copy()
    for j,(gain,direction) in enumerate(((ins_gain,vg104),(ne_gain,vm104))):
        i,k = repair_slots127[2*j:2*j+2]
        W2[:32,i] = gain*np.asarray(direction)/repair_alpha127
        W2[:32,k] = -W2[:32,i]
    for name,value in (
        ("z_post.0.weight",repair_newW0127),
        ("z_post.0.bias",repair_newb0127),
        ("z_post.2.weight",W2),
        ("z_post.2.bias",repair_baseb2127)):
        assert np.isfinite(value).all()
        save_edit(name,np.asarray(value,dtype=np.float32))

install_repair127(-2.0,1.5)
print("latent direction norms:",np.linalg.norm(vg104),np.linalg.norm(vm104))
print("current edits:",edited())

repair_factual_stays127 = [
    int(x) for x in pd.read_csv("/data/tuning.csv",usecols=["stay_id"])["stay_id"].unique()[:100]]
print("NEXT FACTUAL REQUEST:")
print(json.dumps({"type":"factual","model":"both","split":"tuning",
                  "stays":repair_factual_stays127}))

with open("/work/notes.md","a") as f:
    f.write("\n## Python 127 — first posterior dose-innovation candidate\n"
            "- Four low-contribution z_post hidden neurons replaced by two gated GELU differences per drug.\n"
            "- Rate decoders calibrated from training-state dose perturbations; policy centering uses pre-action context.\n"
            "- Compensated displaced original contributions with a small ridge regression, preserving mean and log-sd outputs.\n"
            "- Trial latent-direction gains: insulin -2, NE +1.5; these require GPU calibration, not assumed physical gains.\n"
            "- Edited z_post.0/.2 weights and biases only. Factual limit must be checked next.\n")
```