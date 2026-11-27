The previous construction reserved four posterior neurons to decode the two rates, so there is a concrete dose channel to test. I’ll measure its response to scaling each drug and translate the empirical impulse responses into approximate rollout targets. Correct, nearly linear dose decoding would support calibrating downstream directions; mixed or reversed decoding would refute that approach.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

r132_base = candidates90[0.0]

def r132_channels(indices, u=None):
    ix = np.asarray(indices, dtype=int)
    if u is None:
        u = rr33_u_all[ix]
    x = torch.cat([
        torch.as_tensor(contexts_audit[ix], dtype=torch.float32),
        torch.as_tensor(rr33_e_all[ix], dtype=torch.float32),
        torch.as_tensor(u, dtype=torch.float32)
    ], dim=-1)
    hidden = F.gelu(F.linear(
        x, r132_base["z_post.0.weight"], r132_base["z_post.0.bias"]))
    return F.linear(
        hidden, r132_base["z_post.2.weight"][30:32],
        r132_base["z_post.2.bias"][30:32]).cpu().numpy()

print("=== DOSE-CHANNEL AUDIT ===")
print("state count:", len(rr33_stays), "rate-target shape:", rtrue66.shape)
r132_rng = np.random.default_rng(132)
r132_eligible = np.flatnonzero(
    (np.arange(len(rr33_stays)) > 0)
    & np.any(rtrue66 > 0, axis=1)
)
r132_eligible = r132_eligible[
    rr33_stays[r132_eligible - 1] == rr33_stays[r132_eligible]]
r132_ix = r132_rng.choice(
    r132_eligible, size=min(320, len(r132_eligible)), replace=False)
r132_z = r132_channels(r132_ix)
print("factual correction-channel mean / SD:",
      r132_z.mean(axis=0), r132_z.std(axis=0))

for r132_d, r132_drug in enumerate(["insulin", "norepinephrine"]):
    r132_u1 = torch.as_tensor(
        rr35_scaled_action_encoding(r132_ix, r132_drug, 1.0))
    print(r132_drug, "factor-1 action reconstruction max error:",
          float((r132_u1 - rr33_u_all[r132_ix]).abs().max()))
    r132_mask = rtrue66[r132_ix, r132_d] > 0
    r132_rate = rtrue66[r132_ix, r132_d][r132_mask]
    for r132_factor in [0.0, 0.5, 1.5, 2.0, 3.0]:
        r132_u = rr35_scaled_action_encoding(
            r132_ix, r132_drug, r132_factor)
        r132_dz = (r132_channels(r132_ix, r132_u) - r132_z)[r132_mask]
        r132_expected = (r132_factor - 1.0) * r132_rate
        r132_slopes = (
            r132_expected @ r132_dz /
            max(float(r132_expected @ r132_expected), 1e-12))
        r132_error = r132_dz.copy()
        r132_error[:, r132_d] -= r132_expected
        print(r132_drug, r132_factor,
              "channel slopes:", np.round(r132_slopes, 4),
              "RMSE versus physical dose change:",
              np.round(np.sqrt(np.mean(r132_error**2, axis=0)), 4))

print("\n=== RECORD-DERIVED ROLLOUT TARGETS ===")
r132_rates = (
    tune.loc[tune.event.isin(["insulin", "norepinephrine"]),
             ["stay_id", "hour", "event", "rate"]]
    .assign(hour=lambda x: np.floor(x.hour + 1e-5).astype(int))
    .groupby(["stay_id", "hour", "event"]).rate.sum()
)
r132_targets = {}
for r132_drug, r132_edit, r132_outcome in [
    ("insulin", "insulin_x2", "glucose_lab"),
    ("norepinephrine", "ne_x2", "map_arterial")
]:
    r132_ir = np.asarray(dynamic_ir[r132_outcome], dtype=float)
    r132_tail_rho = float(np.clip(
        np.exp(np.polyfit(
            np.arange(3), np.log(np.abs(r132_ir[-3:])), 1)[0]),
        0.2, 0.95))
    r132_long_ir = np.r_[
        r132_ir, r132_ir[-1] * r132_tail_rho**np.arange(1, 7)]
    r132_rows = r128_roll.loc[
        r128_roll.edit.eq(r132_edit) & r128_roll.reached.astype(bool)].copy()
    r132_short_values, r132_long_values = [], []
    for r132_row in r132_rows.itertuples():
        r132_h = int(round(r132_row.hour))
        r132_short = r132_long = 0.0
        for r132_t in range(12, r132_h):
            r132_rate = float(r132_rates.get(
                (r132_row.stay_id, r132_t, r132_drug), 0.0))
            r132_lag = r132_h - r132_t - 1
            if r132_lag < len(r132_ir):
                r132_short += r132_rate * r132_ir[r132_lag]
            if r132_lag < len(r132_long_ir):
                r132_long += r132_rate * r132_long_ir[r132_lag]
        r132_short_values.append(r132_short)
        r132_long_values.append(r132_long)
    r132_rows["target_short"] = r132_short_values
    r132_rows["target_long"] = r132_long_values
    r132_means = r132_rows.groupby("stay_id")[
        ["target_short", "target_long"]].mean().mean()
    r132_targets[r132_drug] = r132_means.to_dict()
    print(r132_drug, "tail decay:", round(r132_tail_rho, 3),
          "x2 targets:", r132_means.round(3).to_dict(),
          "half targets:", (-0.5 * r132_means).round(3).to_dict())

with open("/work/notes.md", "a") as r132_file:
    r132_file.write(
        "\n## Run 132: physical-dose channel and convolution-target tests\n"
        "- Directly tested latent correction channels under factors 0, 0.5, 1.5, 2, and 3.\n"
        "- Computed approximate tuning rollout targets by convolving recorded doses with empirical drug impulse responses.\n"
        "- Targets are identification-based estimates, not observed counterfactual truth; tail extrapolation is reported separately.\n"
        "- Python runs: 132; forward passes: 6; no submitted edits.\n"
    )
```