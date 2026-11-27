The helpers expose the frozen features immediately before the action encoder’s final projection. I can fit a small regression there to encode each drug’s rate in separate coordinates. I’ll test observed and multiplied rates, including zero-dose records. Accurate held-out decoding—and negligible signal for an absent drug—would support installing it; saturation would limit extrapolation.

```python
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge

tracked105 = reserved103.copy()
temporary105 = rank103[4:8].astype(int)
rate_scale105 = np.array([1., 0.1])
delta105 = 0.01

print("TRACKED", tracked105.tolist(), "TEMPORARY", temporary105.tolist())
print("GAP TENSORS", [n for n in NAMES if "gap" in n])
print("ACTION NUMERIC CONSTANTS",
      [x for x in action_atom_exact.__code__.co_consts
       if isinstance(x, (str, int, float, tuple))][:45])

# Detect information saturation in the fixed input representation.
caps105 = []
for drug105, maximum105 in [("insulin", 48.), ("norepinephrine", 3.)]:
    grid105 = np.linspace(0, maximum105, 1001)
    atoms105 = array103(action_atom_exact(drug105, grid105))
    changes105 = np.linalg.norm(np.diff(atoms105, axis=0), axis=1)
    changing105 = np.flatnonzero(changes105 > 1e-5)
    cap105 = grid105[changing105[-1] + 1] if len(changing105) else 0.
    caps105.append(float(cap105))
    print(drug105, "last distinguishable rate", float(cap105))
caps105 = np.array(caps105)

def action_hidden105(rates, presence):
    rates = np.asarray(rates, dtype=np.float64)
    presence = np.asarray(presence, dtype=bool)
    aggregate = torch.zeros((len(rates), 192), dtype=torch.float32)
    for d, drug in enumerate(["insulin", "norepinephrine"]):
        use = np.flatnonzero(presence[:, d])
        if len(use):
            atoms = action_atom_exact(drug, rates[use, d])
            atoms[:, tracked105] = 0  # planned removal of these atom coordinates
            aggregate[use] += mlp0(atoms, "encode_action.mlp1", "gelu")
    counts = presence.sum(axis=1)
    aggregate /= torch.tensor(np.sqrt(np.maximum(counts, 1)),
                              dtype=torch.float32)[:, None]
    size = torch.tensor(np.log1p(counts), dtype=torch.float32)[:, None]
    mixed = aggregate + lin0(size, "encode_action.size")
    hidden = F.gelu(lin0(mixed, "encode_action.mlp2.0"))
    raw = lin0(hidden, "encode_action.mlp2.2")
    return array103(hidden), array103(raw)

presence105 = np.zeros((len(dose_actual4), 2), dtype=bool)
for d105, drug105 in enumerate(["insulin", "norepinephrine"]):
    presence105[acttable4.loc[acttable4.event == drug105, "j"].to_numpy(), d105] = True

qparts105, pparts105 = [], []
for mult105 in [0., 0.5, 1., 1.5, 2., 3.]:
    qparts105.append(dose_actual4 * mult105)
    pparts105.append(presence105)
rng105 = np.random.default_rng(105)
extra_q105 = rng105.uniform(size=(3500, 2)) * caps105
extra_p105 = rng105.random((3500, 2)) < np.array([0.65, 0.55])
extra_q105[~extra_p105] = 0
qtrain105 = np.vstack(qparts105 + [extra_q105])
ptrain105 = np.vstack(pparts105 + [extra_p105])

qtest105 = rng105.uniform(size=(2000, 2)) * caps105
ptest105 = rng105.random((2000, 2)) < np.array([0.65, 0.55])
qtest105[~ptest105] = 0
qtest105[:100] = 0  # include present-but-zero and empty-action cases

xh105, raw_train105 = action_hidden105(qtrain105, ptrain105)
xt105, raw_test105 = action_hidden105(qtest105, ptest105)

def channel_targets105(rates, raw):
    base = raw.astype(np.float64).copy()
    base[:, temporary105] = 0
    variance = np.var(base, axis=1) + 1e-5
    capped = np.minimum(rates, caps105) / rate_scale105
    denom = 1 - 2 * delta105**2 * np.sum(capped**2, axis=1) / 192
    sigma = np.sqrt(variance / denom)
    return delta105 * capped * sigma[:, None], variance

yt105, base_var_train105 = channel_targets105(qtrain105, raw_train105)
_, base_var_test105 = channel_targets105(qtest105, raw_test105)
xmean105 = xh105.mean(axis=0).astype(np.float64)
xsd105 = np.maximum(xh105.std(axis=0), 0.02).astype(np.float64)
xs105 = (xh105.astype(np.float64) - xmean105) / xsd105
xst105 = (xt105.astype(np.float64) - xmean105) / xsd105

fits105 = []
for alpha105 in [1e-7, 1e-5, 1e-3, 0.1]:
    regression105 = Ridge(alpha=alpha105)
    regression105.fit(xs105, yt105)
    predicted_raw105 = regression105.predict(xst105)
    predicted_sigma105 = np.sqrt(
        base_var_test105 + 2 * np.sum(predicted_raw105**2, axis=1) / 192
    )
    decoded105 = (predicted_raw105 / predicted_sigma105[:, None]
                  / delta105 * rate_scale105)
    truth105 = np.minimum(qtest105, caps105)
    errors105 = decoded105 - truth105
    normalized_mse105 = np.mean((errors105 / rate_scale105)**2)
    report105 = {
        "alpha": alpha105,
        "rmse_I_NE": np.sqrt(np.mean(errors105**2, axis=0)).round(5).tolist(),
        "absent_rmse_I_NE": [
            round(float(np.sqrt(np.mean(errors105[~ptest105[:, d], d]**2))), 6)
            for d in range(2)
        ],
        "max_abs_I_NE": np.max(np.abs(errors105), axis=0).round(4).tolist(),
    }
    print(report105)
    fits105.append((normalized_mse105, regression105, report105))

best105 = min(fits105, key=lambda x: x[0])
channel_coef105 = best105[1].coef_ / xsd105[None, :]
channel_intercept105 = best105[1].intercept_ - channel_coef105 @ xmean105
print("SELECTED", best105[2], "coefficient norms",
      np.linalg.norm(channel_coef105, axis=1))
print("EDITS UNCHANGED", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward 8 / run 105: private-channel feasibility\n"
        "- Clearing four state coordinates costs only +.00034 factual value NLL on tuning A.\n"
        f"- Prepared, but did not install, frozen-feature rate regressions: {best105[2]}.\n"
        f"- Fixed input representation saturation limits: {caps105.tolist()}.\n"
        "- This rate-encoding fit uses no model gradients and makes no new causal-effect claim.\n"
    )
```