The dose decoders generalize reasonably, but the pre-action policy residuals are not small enough to assume factual safety. The earlier near-perfect decoder slopes did not imply near-perfect factual dose prediction.

I’ll install a reversible, drug-gated posterior candidate using the new response directions and compensate for the four displaced neurons. This is a pathway probe, not a final repair; matched GPU factual predictions and rollouts will determine whether to keep, rescale, or reject it.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from pathlib import Path

X169 = torch.cat((context, E166, U166), dim=1)
H169 = F.gelu(F.linear(X169, W["z_post.0.weight"], W["z_post.0.bias"]))
W2old169 = W["z_post.2.weight"].numpy().copy()
importance169 = H169.var(0).numpy() * np.sum(W2old169**2, axis=0)
neurons169 = np.argsort(importance169)[:4]
keep169 = np.setdiff1d(np.arange(192), neurons169)
lost169 = H169[:, neurons169].numpy() @ W2old169[:, neurons169].T

comp169 = Ridge(alpha=.03).fit(H169[fit_mask168][:, keep169].numpy(),
                              lost169[fit_mask168])
err169 = comp169.predict(H169[hold_mask168][:, keep169].numpy()) - lost169[hold_mask168]
print("Replaced neurons:", neurons169.tolist())
print("Held compensation RMS / original displaced RMS:",
      float(np.sqrt(np.mean(err169**2))),
      float(np.sqrt(np.mean(lost169[hold_mask168]**2))))
comp169.fit(H169[:, keep169].numpy(), lost169)

base169 = {n: np.asarray(load_original(n), dtype=np.float32).copy()
           for n in ("z_post.0.weight", "z_post.0.bias",
                     "z_post.2.weight", "z_post.2.bias")}
base169["z_post.2.weight"][:, keep169] += comp169.coef_.astype(np.float32)
base169["z_post.2.bias"] += comp169.intercept_.astype(np.float32)
base169["z_post.2.weight"][:, neurons169] = 0

gate_strength169 = 32.0
for j, drug in enumerate(("insulin", "norepinephrine")):
    n1, n0 = neurons169[2*j:2*j+2]
    gate_row = np.zeros(576, dtype=np.float32)
    gate_row[384:] = gate_strength169 * presence_decoder168.coef_[j]
    gate_bias = gate_strength169 * (float(presence_decoder168.intercept_[j]) - .5)
    innovation_row = np.zeros(576, dtype=np.float32)
    feature_name, policy_reg = policy168[drug]
    innovation_row[:192] = -policy_reg.coef_[:192]
    if feature_name == "context+e":
        innovation_row[192:384] = -policy_reg.coef_[192:]
    innovation_row[384:] = dose_decoder168.coef_[j]
    innovation_bias = float(dose_decoder168.intercept_[j] - policy_reg.intercept_)
    base169["z_post.0.weight"][n1] = gate_row + innovation_row
    base169["z_post.0.bias"][n1] = gate_bias + innovation_bias
    base169["z_post.0.weight"][n0] = gate_row
    base169["z_post.0.bias"][n0] = gate_bias

def features169(u):
    xx = torch.cat((context, E166, torch.as_tensor(u, dtype=torch.float32)), dim=1)
    hh = F.gelu(F.linear(
        xx, torch.as_tensor(base169["z_post.0.weight"]),
        torch.as_tensor(base169["z_post.0.bias"])
    ))
    return np.stack([
        (hh[:, neurons169[2*j]] - hh[:, neurons169[2*j+1]]).numpy()
        for j in range(2)
    ], axis=1)

Fbase169 = features169(U166)
print("\nFactual gated-feature RMS:", np.sqrt(np.mean(Fbase169**2, axis=0)))
print("Absent-drug feature maximum:",
      [float(np.max(np.abs(Fbase169[presence168[:, j] == 0, j]))) for j in range(2)])
print("\n=== DOSE-PERTURBATION FEATURE SLOPES ===")
for j, drug in enumerate(("insulin", "norepinephrine")):
    present = presence168[:, j] > 0
    for mult in (0., .5, 1.5, 2., 3.):
        sc = (mult, 1.) if j == 0 else (1., mult)
        change = features169(Uaug168[sc])[:, j] - Fbase169[:, j]
        target_change = rates152[:, j] * (mult-1)
        slope = float(np.dot(change[present], target_change[present]) /
                      np.dot(target_change[present], target_change[present]))
        print(drug, mult, "slope", round(slope, 4))

def install169(insulin_gain, ne_gain):
    candidate = {n: v.copy() for n, v in base169.items()}
    for j, (drug, gain) in enumerate((
        ("insulin", insulin_gain), ("norepinephrine", ne_gain)
    )):
        direction = np.asarray(latent_directions161[drug], dtype=np.float32).reshape(32)
        candidate["z_post.2.weight"][:32, neurons169[2*j]] = gain * direction
        candidate["z_post.2.weight"][:32, neurons169[2*j+1]] = -gain * direction
    for n, v in candidate.items():
        save_edit(n, v.astype(np.float32))
    return candidate

gains169 = (2.0, 4.0)
candidate169 = install169(*gains169)
print("\nInstalled gains:", gains169)
print("Active edits:", edited())

screen_ids169 = pd.read_csv("/work/fwd_10/factual_original.csv")["stay_id"].unique().tolist()
print("\nSCREEN STAYS:", screen_ids169[:50])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 165–169: response-based posterior probe\n"
        "- Drug-bundle posterior reconstruction RMS .00216; non-drug reconstruction remains inaccurate.\n"
        "- Augmented held-stay rate decoder RMSE: insulin .158, NE .064; presence decoding is accurate.\n"
        "- Held-stay policy RMSE: insulin .539, NE .254; factual neutrality must be measured, not assumed.\n"
        f"- Replaced neurons {neurons169.tolist()}, compensating their original outputs by ridge.\n"
        "- Installed response-based latent directions at gains insulin 2, NE 4, gated by drug presence.\n"
        "- These gains are diagnostic. Awaiting matched GPU factual and rollout validation.\n"
    )
```
