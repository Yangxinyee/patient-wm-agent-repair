Event type 1 is clearly the closest match: RMSE is 2.37 for glucose and 0.77 for MAP. That supports the input layout, but it is not an exact replay. I’ll test whether the remaining difference comes from register-age units or the tool’s random prior draws. Agreement within the expected sampling variation would support response calibration.

```python
import sys
import numpy as np
import torch
from pathlib import Path

def capture159(h, out, stay, hour, event):
    captured = {}
    code = readout153.__code__
    old_profile = sys.getprofile()
    def profile(frame, action, arg):
        if action == "return" and frame.f_code is code:
            captured.update(frame.f_locals)
    try:
        sys.setprofile(profile)
        result = readout153(h, out, int(stay), float(hour), event, 1)
    finally:
        sys.setprofile(old_profile)
    return result, captured

generator159 = torch.Generator().manual_seed(159)
noise159 = torch.randn(64, 32, generator=generator159)
noise159 = torch.cat([noise159, -noise159], dim=0)
readout_audit159 = {}

for event159, items159 in validation_samples158.items():
    audit159 = {key: [] for key in
                ("hour_mean", "minute_mean", "second_mean",
                 "hour_mc", "minute_mc", "second_mc")}
    target159, sample_sd159, prior_logs159 = [], [], []
    for key159, i159 in items159:
        raw159, cap159 = capture159(
            torch.tensor(S[i159:i159+1, 192:960]).reshape(1, 4, 192),
            torch.tensor(S[i159:i159+1, :192]),
            stays156[i159], hours156[i159], event159)
        target159.append(lookup158[key159])
        prior159 = mlp136(cap159["ctx"], "z_prior")
        logsd159 = prior159[:, 32:].clamp(-7, 2)
        prior_logs159.append(logsd159.numpy())
        sampled159 = prior159[:, :32] + noise159 * torch.exp(logsd159)
        sampled_view159 = cap159["ctx"] + linear136(sampled159, "z_head")
        sampled_q159 = norm136(
            sampled_view159 + W["type_emb.weight"][1], "query_norm")
        transition159 = W["embed_transition.value_bin.weight"][
            codes[event159] * 17 + int(cap159["last_bin"])
        ]
        for label159, multiplier159 in (("hour", 1), ("minute", 60), ("second", 3600)):
            reg159 = cap159["register"].clone()
            age_hours159 = np.expm1(float(reg159[0, 1]) * 10)
            reg159[:, 1] = float(np.log1p(age_hours159 * multiplier159) / 10)
            correction159 = (
                linear136(reg159, "last_numeric")
                - linear136(cap159["register"], "last_numeric")
            )
            wc159 = cap159["with_code"] + correction159
            logits_mean159 = mlp136(wc159, "attr_heads.value_bin") + transition159
            mean159 = torch.softmax(logits_mean159, -1) @ centres153(event159)
            wc_mc159 = wc159 + sampled_q159 - cap159["q"]
            logits_mc159 = mlp136(wc_mc159, "attr_heads.value_bin") + transition159
            means_mc159 = torch.softmax(logits_mc159, -1) @ centres153(event159)
            audit159[label159 + "_mean"].append(float(mean159[0]))
            audit159[label159 + "_mc"].append(float(means_mc159.mean()))
            if label159 == "hour":
                sample_sd159.append(float(means_mc159.std()))
    targets159 = np.array(target159)
    print("\nAGE / PRIOR AUDIT:", event159, "n =", len(targets159))
    for label159, values159 in audit159.items():
        errors159 = np.array(values159) - targets159
        print(label159, "RMSE", round(float(np.sqrt(np.mean(errors159**2))), 4),
              "bias", round(float(np.mean(errors159)), 4))
    print("Expected RMSE from averaging four prior draws:",
          round(float(np.sqrt(np.mean(np.array(sample_sd159)**2) / 4)), 4))
    all_logs159 = np.concatenate(prior_logs159)
    print("Prior log-sd range:", float(all_logs159.min()), float(all_logs159.max()))
    readout_audit159[event159] = (audit159, targets159, sample_sd159)

print("\n=== CONTINUED RECORD-EVIDENCE NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[5200:10500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 159 — register age and prior-sampling audit\n"
            "- Event type 1 strongly preferred; prior-mean readout RMSE 2.37 glucose / 0.77 MAP.\n"
            "- Testing age units and antithetic prior averaging against original factual outputs.\n"
            "- No weights changed.\n")
```