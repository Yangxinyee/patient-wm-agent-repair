The bundle counts match record rows plus infusion ends exactly. The earlier reconstruction omitted real inputs. I’ll first identify the WORLD-atom encoding using bundles without END atoms, then add END atoms separately. Near-zero posterior error on these simpler bundles would validate the encoding before any weight edit.

```python
import torch
import torch.nn.functional as F
torch.set_num_threads(8)
torch.set_grad_enabled(False)

print("=== EMBEDDING ROW NORMS ===")
for name in NAMES:
    if name.startswith("embed.categorical.") or name in ("embed.text.weight", "embed.kq.weight"):
        print(name, np.round(np.linalg.norm(load_original(name), axis=1), 4).tolist())

W42 = {}
def wt42(name):
    if name not in W42:
        W42[name] = torch.as_tensor(load_original(name), dtype=torch.float32)
    return W42[name]
def affine42(x, prefix):
    return F.linear(x, wt42(prefix + ".weight"), wt42(prefix + ".bias"))
def net42(x, prefix, activation="gelu"):
    y = affine42(x, prefix + ".0")
    y = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[activation](y)
    return affine42(y, prefix + ".2")
def ln42(x, prefix):
    return F.layer_norm(x, (192,), wt42(prefix + ".weight"), wt42(prefix + ".bias"), 1e-5)

ids42 = np.unique(s40["stay_id"])
dr42 = raw40[raw40.stay_id.isin(ids42) & raw40.event.isin(["insulin", "norepinephrine"])].copy()
dr42["minute_key"] = np.rint(dr42.hour * 60).astype(int)
end42 = dr42.copy()
end42["minute_key"] = np.rint(end42.end_hour * 60).astype(int)
starts42 = {(int(s), int(t)): g for (s,t), g in dr42.groupby(["stay_id", "minute_key"])}
ends42 = {(int(s), int(t)): g for (s,t), g in end42.groupby(["stay_id", "minute_key"])}
eligible42 = []
for i in np.flatnonzero(s40["has_insulin"] | s40["has_norepinephrine"]):
    if i == 0 or s40["stay_id"][i-1] != s40["stay_id"][i]:
        continue
    key = (int(s40["stay_id"][i]), int(round(s40["hour"][i] * 60)))
    g = starts42.get(key)
    if g is not None and key not in ends42 and len(g) == s40["n_atoms"][i]:
        eligible42.append(i)
idx42 = np.array(eligible42, dtype=int)
if len(idx42) > 80:
    idx42 = idx42[np.linspace(0, len(idx42)-1, 80).astype(int)]
ctx42 = contexts37(s40, idx42)
targetz42 = torch.tensor(s40["x"][idx42, 960:])
print("\nNo-END bundles:", len(eligible42), "using:", len(idx42),
      "posterior RMS:", float(targetz42.square().mean().sqrt()))

num42 = {(v["code"], v["attribute"]): v for v in VOCAB["numeric"]}
code42 = {v["name"]: v["id"] for v in VOCAB["codes"]}
catnames42 = [name for name in NAMES if name.startswith("embed.categorical.")]
defaults42 = sum((wt42(name)[0] for name in catnames42), torch.zeros(192)) + wt42("embed.text.weight")[0]
pairs42 = [(u,r) for u in range(6) for r in range(4)]
counts42 = torch.zeros(len(idx42))
atommeta42 = []
for j, i in enumerate(idx42):
    key = (int(s40["stay_id"][i]), int(round(s40["hour"][i]*60)))
    for row in starts42[key].itertuples():
        atommeta42.append((j, row.event, float(row.rate), float(row.dose)))
        counts42[j] += 1

def pooled_start42(drug, activation="gelu", side="left", extra=None):
    aa = [a for a in atommeta42 if a[1] == drug]
    if not aa:
        return torch.zeros(len(pairs42), len(idx42), 192)
    owner = torch.tensor([a[0] for a in aa], dtype=torch.long)
    rate = np.array([a[2] for a in aa])
    dose = np.array([a[3] for a in aa])
    numeric = np.zeros((len(aa), 6), np.float32)
    bins = {}
    for attribute, x, pos in [("dose", dose, 2), ("rate", rate, 4)]:
        spec = num42[drug, attribute]
        numeric[:,pos] = np.clip((x-spec["median"])/spec["iqr"], -10, 10)
        numeric[:,pos+1] = 1
        bins[attribute] = 1 + np.searchsorted(spec["fine_bin_edges"], x, side=side)
    base = (wt42("embed.code.weight")[code42[drug]]
            + wt42("embed.kind.weight")[0] + wt42("embed.ekind.weight")[2]
            + defaults42 + wt42("embed.bins.value.weight")[0]
            + wt42("embed.bins.dose.weight")[torch.tensor(bins["dose"])]
            + wt42("embed.bins.rate.weight")[torch.tensor(bins["rate"])]
            + F.linear(torch.tensor(numeric), wt42("embed.numeric.weight"), wt42("embed.numeric.bias")))
    if extra is not None:
        base = base + extra
    results = []
    for unit, rateunit in pairs42:
        emb = (base + wt42("embed.categorical.unit.weight")[unit]
               - wt42("embed.categorical.unit.weight")[0]
               + wt42("embed.categorical.rate_unit.weight")[rateunit]
               - wt42("embed.categorical.rate_unit.weight")[0])
        hidden = net42(emb, "encode_action.mlp1", activation)
        pooled = torch.zeros(len(idx42), 192)
        pooled.index_add_(0, owner, hidden)
        results.append(pooled)
    return torch.stack(results)

results42 = []
for activation in ("gelu", "silu", "relu"):
    pi = pooled_start42("insulin", activation)
    pn = pooled_start42("norepinephrine", activation)
    pooled = (pi[:,None] + pn[None,:]).reshape(-1, len(idx42), 192)
    pooled = pooled / counts42.sqrt()[None,:,None]
    pooled += affine42(torch.log1p(counts42)[:,None], "encode_action.size")[None]
    u = ln42(net42(pooled, "encode_action.mlp2", activation), "encode_action.norm")
    context_part = F.linear(ctx42, wt42("z_post.0.weight")[:,:192], wt42("z_post.0.bias"))
    hidden = context_part[None] + F.linear(u, wt42("z_post.0.weight")[:,384:])
    for za in ("gelu", "silu", "relu"):
        post = affine42({"gelu": F.gelu, "silu": F.silu, "relu": F.relu}[za](hidden), "z_post.2")[...,:32]
        rms = (post-targetz42[None]).square().mean((1,2)).sqrt()
        for k in torch.argsort(rms)[:3].tolist():
            results42.append((float(rms[k]), activation, za,
                              pairs42[k//len(pairs42)], pairs42[k%len(pairs42)]))
results42.sort()
print("\nBest no-END reconstructions:", results42[:12])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 42\nExported counts exactly include WORLD starts plus infusion END atoms. "
            "Searching action encoder conventions on no-END bundles first. "
            f"Best configurations: {results42[:3]}. No edits.\n")
```