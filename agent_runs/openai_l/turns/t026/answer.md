The state export contains the recurrent features needed to inspect the value predictions. I’ll reconstruct the value head on CPU before editing it, and compute provisional rollout targets from the record-based response kernels. Close agreement with the GPU factual predictions would validate the reconstruction; a large mismatch would rule out using it to design an edit.

```python
import torch
import torch.nn.functional as F
import numpy as np, pandas as pd, time
torch.set_num_threads(8)
torch.set_grad_enabled(False)

# Provisional targets: measured record-based kernels, with conservative tails.
kernels22 = {
    "insulin": np.array([-1.95, -2.30956, -2.00524, -1.60385,
                         -1.08411, -.78217, -.50421, -.37554,
                         -.24, -.15, -.10, -.06]),
    "norepinephrine": np.array([3.10, 2.47544, 1.65817, .94336,
                                .35728, .27455, .01818, .16684,
                                0., 0., 0., 0.])
}
tune_ids22 = np.sort(tuning.stay_id.unique())
tune_pos22 = {s: i for i, s in enumerate(tune_ids22)}
tune_actions22, target_paths22 = {}, {}
for drug22, kernel22 in kernels22.items():
    td22 = tuning.loc[tuning.event.eq(drug22)].copy()
    td22["h_start"] = np.floor(td22.hour + 1e-6).astype(int)
    grid22 = td22.pivot_table(index="stay_id", columns="h_start",
                              values="rate", aggfunc="sum")
    aa22 = grid22.reindex(index=tune_ids22, columns=range(25)).fillna(0).to_numpy()
    tune_actions22[drug22] = aa22
    changed22 = aa22.copy()
    changed22[:, :12] = 0
    conv22 = np.array([np.convolve(row, kernel22)[:24] for row in changed22])
    target_paths22[drug22] = np.column_stack([np.zeros(len(aa22)), conv22])

roll22 = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
record_targets22 = {}
for edit22, drug22, factor22 in [
    ("insulin_x2", "insulin", 1.), ("insulin_half", "insulin", -.5),
    ("ne_x2", "norepinephrine", 1.), ("ne_half", "norepinephrine", -.5)
]:
    rr22 = roll22.loc[roll22.edit.eq(edit22) & roll22.reached].copy()
    rr22["record_target"] = [
        factor22 * target_paths22[drug22][tune_pos22[s], int(h)]
        for s, h in zip(rr22.stay_id, rr22.hour)
    ]
    record_targets22[edit22] = rr22.groupby("stay_id").record_target.mean().mean()
    print(edit22, "provisional record target", round(record_targets22[edit22], 4),
          "original", round(rr22.groupby("stay_id").shift.mean().mean(), 4),
          "row-average target", round(rr22.record_target.mean(), 4))

# Reconstruct the value head from exported pre-reading states.
s22 = np.load("/work/fwd_3/states_original.npz")
f22 = pd.read_csv("/work/fwd_2/factual_original.csv")
reg22 = tuning.loc[tuning.event.isin(VOCAB["value_bin_centres"])].copy()
reg22 = reg22.sort_values(["stay_id", "event", "hour"])
reg22["last_value22"] = reg22.groupby(["stay_id", "event"]).value.shift()
reg22["last_hour22"] = reg22.groupby(["stay_id", "event"]).hour.shift()
d22 = f22.merge(reg22[["stay_id", "hour", "event", "last_value22", "last_hour22"]],
                on=["stay_id", "hour", "event"], validate="one_to_one")

idx_by_stay22 = {s: np.flatnonzero(s22["stay_id"] == s) for s in d22.stay_id.unique()}
pre_idx22 = np.empty(len(d22), dtype=int)
for s, positions22 in d22.groupby("stay_id", sort=False).indices.items():
    si22 = idx_by_stay22[s]
    j22 = np.searchsorted(s22["hour"][si22], d22.hour.to_numpy()[positions22],
                         side="left") - 1
    assert np.all(j22 >= 0)
    pre_idx22[positions22] = si22[j22]
d22["pre_state22"] = pre_idx22
sample_idx22 = np.random.default_rng(2201).choice(len(d22), 1000, replace=False)
ds22 = d22.iloc[sample_idx22].reset_index(drop=True)
pre22 = ds22.pre_state22.to_numpy()
dt22 = (ds22.hour.to_numpy() - s22["hour"][pre22]) * 3600.
assert np.all(dt22 > 0)

weights22 = {}
def w22(name):
    if name not in weights22:
        weights22[name] = torch.as_tensor(load_original(name), dtype=torch.float32)
    return weights22[name]
def lin22(x, prefix):
    return F.linear(x, w22(prefix + ".weight"),
                    w22(prefix + ".bias") if prefix + ".bias" in SHAPES else None)
def norm22(x, prefix):
    return F.layer_norm(x, (192,), w22(prefix + ".weight"), w22(prefix + ".bias"))
def act22(x, activation):
    if activation == "gelu":
        return F.gelu(x)
    if activation == "silu":
        return F.silu(x)
    return F.relu(x)
def mlp22(x, prefix, activation):
    return lin22(act22(lin22(x, prefix + ".0"), activation), prefix + ".2")

codes22 = {v["name"]: v["id"] for v in VOCAB["codes"]}
normspec22 = {(v["code"], v["attribute"]): v for v in VOCAB["numeric"]}
code22 = torch.tensor([codes22[e] for e in ds22.event], dtype=torch.long)
has22 = ds22.last_value22.notna().to_numpy().astype(np.float32)
last_z22, last_bin22, observed_bin22 = [], [], []
centres22 = []
for row22 in ds22.itertuples():
    spec22 = normspec22[(row22.event, "value")]
    lv22 = row22.last_value22
    last_z22.append(np.clip((lv22-spec22["median"])/spec22["iqr"], -10, 10)
                    if np.isfinite(lv22) else 0.)
    last_bin22.append(1 + np.searchsorted(spec22["fine_bin_edges"], lv22, side="left")
                      if np.isfinite(lv22) else 0)
    observed_bin22.append(1 + np.searchsorted(spec22["fine_bin_edges"],
                                            row22.observed, side="left"))
    centres22.append(VOCAB["value_bin_centres"][row22.event])
last_z22 = np.array(last_z22, dtype=np.float32)
last_bin22 = torch.tensor(last_bin22, dtype=torch.long)
observed_bin22 = torch.tensor(observed_bin22, dtype=torch.long)
centres22 = torch.tensor(np.array(centres22), dtype=torch.float32)
out22 = torch.tensor(s22["x"][pre22, :192])
h22 = torch.tensor(s22["x"][pre22, 192:960].reshape(-1, 4, 192))
flow22 = torch.exp(-torch.tensor(dt22, dtype=torch.float32)[:, None, None]
                   * torch.exp(w22("core.flow.a"))[None, None, :])
flow_summary22 = norm22((h22 * flow22).mean(1), "core.out_norm")
base22 = (w22("embed.code.weight")[code22]
          + w22("embed.bins.value.weight")[last_bin22]
          + w22("embed.bins.rate.weight")[0]
          + w22("last_status.weight")[0])
transition22 = w22("embed_transition.value_bin.weight")[code22 * 17 + last_bin22]
age_hours22 = (ds22.hour.to_numpy() - ds22.last_hour22.fillna(ds22.hour).to_numpy())
reference_mean22 = ds22.predicted_mean.to_numpy()
reference_nll22 = ds22.nll.to_numpy()
head_tests22 = []
start22 = time.time()

for prior_act22 in ["gelu", "silu", "relu"]:
    for gap_scale22 in [1., .1]:
        gap_input22 = torch.tensor(np.log(dt22) * gap_scale22, dtype=torch.float32)[:, None]
        ctx22 = out22 + flow_summary22 + mlp22(gap_input22, "gap", prior_act22)
        prior_mu22 = mlp22(ctx22, "z_prior", prior_act22)[:, :32]
        view22 = ctx22 + lin22(prior_mu22, "z_head")
        q22 = norm22(view22 + w22("type_emb.weight")[1], "query_norm")
        for age_unit22 in ["seconds", "hours"]:
            age22 = np.log1p(age_hours22 * (3600 if age_unit22 == "seconds" else 1)) / 10.
            for register_map22 in ["prefix_flags", "numeric_first", "no_value_flag"]:
                nf22 = np.zeros((len(ds22), 8), np.float32)
                if register_map22 == "prefix_flags":
                    nf22[:, 0], nf22[:, 1] = has22, age22
                    nf22[:, 2], nf22[:, 3] = last_z22, has22
                elif register_map22 == "numeric_first":
                    nf22[:, 0], nf22[:, 1] = last_z22, has22
                    nf22[:, 6], nf22[:, 7] = has22, age22
                else:
                    nf22[:, 0], nf22[:, 1], nf22[:, 2] = has22, age22, last_z22
                wc22 = q22 + base22 + lin22(torch.tensor(nf22), "last_numeric")
                for head_act22 in ["gelu", "silu", "relu"]:
                    logits22 = mlp22(wc22, "attr_heads.value_bin", head_act22) + transition22
                    probs22 = logits22.softmax(-1)
                    mean22 = (probs22[:, 1:] * centres22).sum(-1)
                    nll22 = -probs22[torch.arange(len(ds22)), observed_bin22].log()
                    rmse22 = np.sqrt(np.mean((mean22.numpy() - reference_mean22)**2))
                    nll_mae22 = np.mean(np.abs(nll22.numpy() - reference_nll22))
                    head_tests22.append((rmse22, nll_mae22, prior_act22, gap_scale22,
                                         age_unit22, register_map22, head_act22))
head_tests22.sort()
print("\nBest head reconstructions (RMSE, NLL MAE, specifications):")
for result22 in head_tests22[:12]:
    print(result22)
print("seconds", round(time.time() - start22, 2))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 22\n"
            "Computed provisional record-kernel rollout targets on the original forward stays: "
            + repr(record_targets22) + ". These remain conditional on linear response/tail assumptions. "
            "Testing CPU head reconstruction against GPU factual predictions before choosing a weight edit. "
            "Best reconstruction: " + repr(head_tests22[0]) + ". No edits.\n")
```
