Saved states include single-drug bundles, which provide a clean check of the action encoder. I’ll compare reconstructed posterior means across the possible categorical IDs. A near-exact match would validate the encoding; a substantial mismatch would mean I should not edit that path yet.

```python
import torch
import torch.nn.functional as tF
torch.set_num_threads(8)
Wrepair = {n: load_original(n).float() for n in NAMES}

def rlinear(x, prefix):
    return tF.linear(x, Wrepair[prefix + ".weight"], Wrepair.get(prefix + ".bias"))

def rnorm(x, prefix):
    return tF.layer_norm(x, (192,), Wrepair[prefix + ".weight"], Wrepair[prefix + ".bias"])

def rmlp(x, prefix):
    return rlinear(tF.gelu(rlinear(x, prefix + ".0")), prefix + ".2")

def action_encode_atoms(x):
    n = x.shape[-2]
    pooled = rmlp(x, "encode_action.mlp1").sum(-2) / np.sqrt(n)
    size = rlinear(torch.full((*pooled.shape[:-1], 1), np.log1p(n)), "encode_action.size")
    return rnorm(rmlp(pooled + size, "encode_action.mlp2"), "encode_action.norm")

categorical_names = [n for n in NAMES if n.startswith("embed.categorical.") and n.endswith(".weight")]
print("ZERO-ROW NORMS:", {n: round(float(Wrepair[n][0].norm()), 5) for n in categorical_names + ["embed.kq.weight", "embed.text.weight", "embed.bins.value.weight", "embed.bins.dose.weight", "embed.bins.rate.weight"]})
refx = np.asarray(S_reference["x"])
single_rows = state_meta.index[(state_meta.n_atoms == 1) & (state_meta.has_insulin | state_meta.has_norepinephrine)].to_numpy()
single_rows = np.array([j for j in single_rows if j > 0 and state_meta.iloc[j-1].stay_id == state_meta.iloc[j].stay_id])
print("SINGLE ACTION BUNDLES:", len(single_rows))
single_results = []
for drug in ["insulin", "norepinephrine"]:
    ids1 = np.array([j for j in single_rows if bool(state_meta.iloc[j]["has_" + drug])])
    if len(ids1) == 0:
        continue
    ids1 = ids1[:60]
    lookup1 = train[train.event == drug].set_index(["stay_id", "hour"])
    sub1 = lookup1.reindex(pd.MultiIndex.from_frame(state_meta.iloc[ids1][["stay_id", "hour"]]))
    keep1 = sub1.rate.notna().to_numpy()
    ids1, sub1 = ids1[keep1], sub1.iloc[np.flatnonzero(keep1)]
    prev1 = torch.tensor(refx[ids1-1], dtype=torch.float32)
    dt1 = torch.tensor((state_meta.iloc[ids1].hour.to_numpy() - state_meta.iloc[ids1-1].hour.to_numpy()) * 3600, dtype=torch.float32)
    h1 = prev1[:, 192:960].reshape(-1, 4, 192) * torch.exp(-dt1[:, None, None] * torch.exp(Wrepair["core.flow.a"]))
    ctx1 = prev1[:, :192] + rnorm(h1.mean(1), "core.out_norm") + rmlp(torch.log1p(dt1)[:, None] / 10, "gap")
    goal1 = torch.tensor(refx[ids1, 960:992], dtype=torch.float32)
    nums1 = torch.zeros(len(ids1), 6)
    nums1[:, 3] = nums1[:, 5] = 1
    base1 = Wrepair["embed.code.weight"][code_ids[drug]] + Wrepair["embed.kind.weight"][0] + Wrepair["embed.ekind.weight"][2]
    base1 = base1 + Wrepair["embed.bins.value.weight"][0] + Wrepair["embed.numeric.bias"]
    for attr1, col1 in [("dose", 2), ("rate", 4)]:
        spec1 = numeric_specs[(drug, attr1)]
        vv1 = sub1[attr1].to_numpy()
        nums1[:, col1] = torch.tensor(np.clip((vv1-spec1["median"])/spec1["iqr"], -10, 10), dtype=torch.float32)
        bins1 = np.searchsorted(np.array(spec1["fine_bin_edges"]), vv1, side="left") + 1
        base1 = base1 + Wrepair[f"embed.bins.{attr1}.weight"][torch.tensor(bins1)]
    base1 = base1 + tF.linear(nums1, Wrepair["embed.numeric.weight"])
    for n1 in categorical_names:
        if n1 not in ["embed.categorical.unit.weight", "embed.categorical.rate_unit.weight"]:
            base1 = base1 + Wrepair[n1][0]
    for uid1 in range(6):
        for rid1 in range(4):
            for qid1 in range(4):
                for tid1 in range(2):
                    atom1 = base1 + Wrepair["embed.categorical.unit.weight"][uid1] + Wrepair["embed.categorical.rate_unit.weight"][rid1]
                    atom1 = atom1 + Wrepair["embed.kq.weight"][qid1] + Wrepair["embed.text.weight"][tid1]
                    u1 = action_encode_atoms(atom1[:, None, :])
                    zp1 = rmlp(torch.cat([ctx1, torch.zeros_like(ctx1), u1], dim=1), "z_post")[:, :32]
                    rmse1 = float(torch.sqrt(torch.mean((zp1-goal1)**2)))
                    single_results.append((drug, len(ids1), uid1, rid1, qid1, tid1, rmse1))
single_check = pd.DataFrame(single_results, columns=["drug", "n", "unit", "rate_unit", "kq", "text", "z_rmse"])
print(single_check.groupby("drug", group_keys=False).apply(lambda d: d.nsmallest(6, "z_rmse")).to_string(index=False))
```
