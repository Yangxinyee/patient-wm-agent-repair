Across all 400 stays, the calibrated loss increase is approximately 0.00434—within the limit, but slightly worse than the pre-calibration score of 0.00414. I’ll verify that matched comparison and undo the intercept change if it raised loss. I’ll also inspect the paired value-head units: clipping there could explain weak half-dose responses, but that hypothesis needs a reversible test.

```python
import numpy as np
import pandas as pd
import torch, re, json
from pathlib import Path
from safetensors.torch import save_file

def factual_pool298(ks):
    os, es = [], []
    for k in ks:
        os.append(pd.read_csv(f"/work/fwd_{k}/factual_original.csv"))
        es.append(pd.read_csv(f"/work/fwd_{k}/factual_edited.csv"))
    o, e = pd.concat(os, ignore_index=True), pd.concat(es, ignore_index=True)
    keys = ["stay_id", "hour", "event", "observed"]
    o = o.sort_values(keys).reset_index(drop=True)
    e = e.sort_values(keys).reset_index(drop=True)
    assert o[keys].equals(e[keys])
    assert o.stay_id.nunique() == 400
    return o, e

O298, CURRENT298 = factual_pool298([41, 42, 43, 44])
OB298, BEFORE298 = factual_pool298([33, 36, 39, 40])
keys298 = ["stay_id", "hour", "event", "observed"]
assert O298[keys298].equals(OB298[keys298])
audit298 = O298[keys298].copy()
audit298["baseline_delta"] = BEFORE298.nll.values - OB298.nll.values
audit298["calibrated_delta"] = CURRENT298.nll.values - O298.nll.values
audit298["calibration_change"] = CURRENT298.nll.values - BEFORE298.nll.values

print("=== EXACT 400-STAY FACTUAL AUDIT ===")
print("readings", len(audit298))
print(audit298[["baseline_delta", "calibrated_delta", "calibration_change"]].mean().to_dict())
print(audit298.groupby("event")[["baseline_delta", "calibrated_delta", "calibration_change"]].mean().round(7).to_string())
z298 = audit298.groupby("stay_id").agg(
    baseline_sum=("baseline_delta", "sum"),
    calibrated_sum=("calibrated_delta", "sum"),
    n=("baseline_delta", "size"))
rng298 = np.random.default_rng(298)
ix298 = rng298.integers(0, len(z298), (3000, len(z298)))
for col in ("baseline_sum", "calibrated_sum"):
    boot = z298[col].to_numpy()[ix298].sum(1) / z298.n.to_numpy()[ix298].sum(1)
    print(col, "bootstrap_5_95", np.quantile(boot, [.05, .95]).tolist())

BASE298 = {n: load(n).clone().contiguous() for n in edited()}
save_file(BASE298, "/work/checkpoint_run298.safetensors")
changed298 = []
for n, t in BASE298.items():
    reference = BASE295[n] if n in BASE295 else load_original(n)
    if not torch.equal(t, reference):
        changed298.append((n, float((t-reference).abs().max())))
print("DIFFERENCES_FROM_PRE_CALIBRATION", changed298)

reverted_cal298 = False
if (float(audit298.calibration_change.mean()) > 0 and
    [n for n, _ in changed298] == ["attr_heads.value_bin.0.bias"]):
    save_edit("attr_heads.value_bin.0.bias", BASE295["attr_heads.value_bin.0.bias"].clone())
    reverted_cal298 = True
print("INTERCEPT_CALIBRATION_REVERTED", reverted_cal298)
BASE_FINAL298 = {n: load(n).clone().contiguous() for n in edited()}
save_file(BASE_FINAL298, "/work/checkpoint_baseline298.safetensors")

print("\n=== PAIRED VALUE-HEAD UNITS ===")
W0298 = load("attr_heads.value_bin.0.weight").numpy()
W2298 = load("attr_heads.value_bin.2.weight").numpy()
b0298 = load("attr_heads.value_bin.0.bias").numpy()
PAIRS298 = {}
for signal in (74, 107):
    errs = np.linalg.norm(W2298 + W2298[:, signal, None], axis=0)
    errs[signal] = np.inf
    control = int(np.argmin(errs))
    PAIRS298[signal] = control
    print("signal", signal, "control", control,
          "relative_antisymmetry_error",
          float(errs[control] / max(np.linalg.norm(W2298[:, signal]), 1e-12)),
          "biases", [float(b0298[signal]), float(b0298[control])],
          "input_row_difference_norm", float(np.linalg.norm(W0298[signal]-W0298[control])))
    for j in (signal, control):
        inds = np.argsort(np.abs(W0298[j]))[-6:][::-1]
        print("row", j, "largest_inputs",
              [(int(i), round(float(W0298[j,i]), 5)) for i in inds])

for name in ("glucose_columns289", "r250_signals", "r249_sensor_controls",
             "r249_control", "r248_control", "calibration295"):
    if name in globals():
        print(name, repr(globals()[name])[:650])

print("\n=== ACTIVATION / GATING NOTES ===")
text298 = Path("/work/notes.md").read_text()
for term in ("ReLU", "GELU", "SiLU", "signal/control"):
    matches = list(re.finditer(re.escape(term), text298, re.I))
    if matches:
        for m in ([matches[0]] if len(matches) == 1 else [matches[0], matches[-1]]):
            print(term, text298[max(0, m.start()-100):m.end()+180].replace("\n", " "))
print("RECONSTRUCTION_FUNCTIONS")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and hasattr(obj, "__code__") and re.search(r"(head|value|predict|factual)", name):
        activations = [c for c in obj.__code__.co_names if c in ("relu", "gelu", "silu", "tanh")]
        print(name, activations)
print("FEATURE_VARIABLE_NAMES",
      ", ".join(n for n in sorted(list(globals()))
                if re.search(r"(^X|with_code|head_input|head_feat|preact)", n))[:1400])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 298: complete matched calibration audit and nonlinear-head inspection\n"
            + f"- Exact pre-calibration 400-stay value-NLL delta: {audit298.baseline_delta.mean():.8f}.\n"
            + f"- Exact calibrated delta: {audit298.calibrated_delta.mean():.8f}; "
              f"calibration loss change: {audit298.calibration_change.mean():.8f}.\n"
            + f"- Intercept calibration reverted: {reverted_cal298}. "
              "Moment correction is not retained when it worsens the actual NLL.\n"
            + "- Saved both snapshots; current accepted-baseline snapshot is "
              "/work/checkpoint_baseline298.safetensors.\n"
            + f"- Inspected candidate signal/control pairs: {PAIRS298}. "
              "A shared-offset test is only justified if output columns cancel and "
              "code gating remains intact; no shared-offset edit made yet.\n"
            + "- The empirical pooled guardrail is not a guarantee on unseen stays. "
              "Total loss over non-value outputs and unavailable intervention rollouts "
              "remain unverified.\n")
```
