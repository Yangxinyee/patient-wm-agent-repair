The matched training predictions have value NLL 1.64546. I’ll now reconstruct the value head from the saved states and compare its means and bin losses with those predictions. This checks activation, register, and timing conventions without changing any weights. A small reconstruction error would justify using it to design an edit.

```python
import torch
import torch.nn.functional as TF
from itertools import product

torch.set_num_threads(8)
torch.set_grad_enabled(False)

def wt(name):
    obj = load_original(name)
    return torch.as_tensor(np.asarray(obj), dtype=torch.float32)

def ln(x, prefix):
    return TF.layer_norm(x, (192,), wt(prefix+".weight"), wt(prefix+".bias"), 1e-5)

def activation(x, kind):
    return {"gelu": TF.gelu, "silu": TF.silu, "relu": TF.relu}[kind](x)

def mlp(x, prefix, kind):
    return TF.linear(activation(TF.linear(x, wt(prefix+".0.weight"),
                                         wt(prefix+".0.bias")), kind),
                     wt(prefix+".2.weight"), wt(prefix+".2.bias"))

archive = np.load("/work/fwd_3/states_original.npz")
SX = archive["x"].copy()
SSID = archive["stay_id"].copy()
STIME = archive["hour"].copy()
head_factual = pd.read_csv("/work/fwd_4/factual_original.csv")
code_ids = {c["name"]: c["id"] for c in VOCAB["codes"]}
numeric_specs = {(v["code"], v["attribute"]): v for v in VOCAB["numeric"]}
state_groups = {sid: np.flatnonzero(SSID == sid) for sid in np.unique(SSID)}
record_groups = {
    (sid, event): grp.sort_values("hour")
    for (sid, event), grp in train[train.stay_id.isin(np.unique(SSID))].groupby(["stay_id", "event"])
}

metadata = []
for row in head_factual.itertuples(index=False):
    ii = state_groups[row.stay_id]
    jj = np.searchsorted(STIME[ii], row.hour, side="left") - 1
    if jj < 0:
        continue
    state_i = ii[jj]
    rec = record_groups[(row.stay_id, row.event)]
    kk = np.searchsorted(rec.hour.to_numpy(), row.hour, side="left") - 1
    has_last = float(kk >= 0)
    last_value = float(rec.iloc[kk].value) if kk >= 0 else 0.0
    age_seconds = (row.hour - float(rec.iloc[kk].hour))*3600 if kk >= 0 else 0.0
    spec = numeric_specs[(row.event, "value")]
    zlast = np.clip((last_value-spec["median"])/spec["iqr"], -10, 10) if kk >= 0 else 0.0
    last_bin = 1 + np.searchsorted(spec["fine_bin_edges"], last_value, side="left") if kk >= 0 else 0
    target_bin = 1 + np.searchsorted(spec["fine_bin_edges"], row.observed, side="left")
    metadata.append((row.stay_id, row.hour, row.event, state_i,
                     (row.hour-STIME[state_i])*3600, has_last, age_seconds,
                     zlast, last_bin, target_bin, row.predicted_mean, row.nll,
                     spec["iqr"], code_ids[row.event]))
head_meta = pd.DataFrame(metadata, columns=[
    "stay_id","hour","event","state_i","dt","has_last","age_s","zlast",
    "last_bin","target_bin","original_mean","original_nll","scale","code"])
test_rows = np.random.default_rng(137).choice(len(head_meta), 320, replace=False)
hm = head_meta.iloc[test_rows].reset_index(drop=True)
sx = torch.tensor(SX[hm.state_i.to_numpy()], dtype=torch.float32)
out = sx[:, :192]
hh = sx[:, 192:960].reshape(-1,4,192)
dt = torch.tensor(hm.dt.to_numpy(), dtype=torch.float32)
hflow = hh * torch.exp(-dt[:,None,None] * torch.exp(wt("core.flow.a"))[None,None,:])
context_base = out + ln(hflow.mean(1), "core.out_norm")
codes_t = torch.tensor(hm.code.to_numpy(), dtype=torch.long)
lastbins_t = torch.tensor(hm.last_bin.to_numpy(), dtype=torch.long)
centres_t = torch.tensor(np.array([
    [0.] + VOCAB["value_bin_centres"][e] for e in hm.event
]), dtype=torch.float32)
transition_t = wt("embed_transition.value_bin.weight")[codes_t*17 + lastbins_t]
target_t = torch.tensor(hm.target_bin.to_numpy(), dtype=torch.long)
target_mu = hm.original_mean.to_numpy()
scale_mu = hm.scale.to_numpy()
last_has = torch.tensor(hm.has_last.to_numpy(), dtype=torch.float32)
last_z = torch.tensor(hm.zlast.to_numpy(), dtype=torch.float32)
last_age = torch.log1p(torch.tensor(hm.age_s.to_numpy(), dtype=torch.float32))/10
base_code = (wt("embed.code.weight")[codes_t]
             + wt("embed.bins.value.weight")[lastbins_t]
             + wt("embed.bins.rate.weight")[0])

reconstruction_results = []
for act, gap_mode, type_i, num_mode, status_mode in product(
        ["gelu","silu","relu"], ["log1p","log"], [1,6],
        ["interleaved","numbers_then_flags","four_padded"], [0,1]):
    gap_input = (torch.log1p(dt) if gap_mode == "log1p" else torch.log(dt.clamp_min(1)))/10
    gap_vec = mlp(gap_input[:,None], "gap", act)
    ctx = context_base + gap_vec
    zmean = mlp(ctx, "z_prior", act)[:, :32]
    view = ctx + TF.linear(zmean, wt("z_head.weight"), wt("z_head.bias"))
    q = ln(view + wt("type_emb.weight")[type_i], "query_norm")
    nums = torch.zeros((len(hm),8))
    nums[:,0], nums[:,1], nums[:,2] = last_has, last_age, last_z
    if num_mode == "interleaved":
        nums[:,3] = last_has
    elif num_mode == "numbers_then_flags":
        nums[:,5] = last_has
    xhead = (q + base_code + TF.linear(nums, wt("last_numeric.weight"), wt("last_numeric.bias"))
             + wt("last_status.weight")[status_mode])
    hidden = activation(TF.linear(xhead, wt("attr_heads.value_bin.0.weight"),
                                  wt("attr_heads.value_bin.0.bias")), act)
    logits = TF.linear(hidden, wt("attr_heads.value_bin.2.weight"),
                       wt("attr_heads.value_bin.2.bias")) + transition_t
    probs = logits.softmax(-1)
    mu = (probs*centres_t).sum(-1).numpy()
    nll = -probs[torch.arange(len(hm)), target_t].clamp_min(1e-12).log().numpy()
    reconstruction_results.append({
        "activation":act, "gap":gap_mode, "type":type_i, "numeric":num_mode,
        "status":status_mode, "scaled_mean_rmse":np.sqrt(np.mean(((mu-target_mu)/scale_mu)**2)),
        "nll_rmse":np.sqrt(np.mean((nll-hm.original_nll.to_numpy())**2)),
        "nll":nll.mean()
    })
reconstruction_table = pd.DataFrame(reconstruction_results).sort_values("scaled_mean_rmse")
print(reconstruction_table.head(12).to_string(index=False))
print("\nMetadata:", head_meta.shape, "matched original NLL:", head_meta.original_nll.mean())
print("\nPrior / head weight norms:")
for name in ["z_head.weight","attr_heads.value_bin.0.weight","attr_heads.value_bin.2.weight"]:
    print(name, float(wt(name).norm()))
print("\nEdits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Head reconstruction setup\n"
            "- Original factual training first 100: value NLL 1.64546 (4,510 readings).\n"
            "- Saved states matched to preceding bundles; testing numerical head conventions before any edit.\n"
            "- Insulin long-horizon kernel remains weakly identified; NE evidence is more stable.\n"
            "- Python runs: 16. Forward passes: 4. No edits.\n")
```
