The decoder’s register terms match exactly, but the generic action encoder does not. I’ll replace its ambiguous unit defaults with the validated per-drug unit indices and check again. I’ll also build a reusable value decoder so we can measure dose sensitivity before editing any weights.

```python
print("=== ACTION EMBEDDING CONSTANTS ===")
print(atom_embedding49.__code__.co_consts)
print("unit tables:", [(n, SHAPES[n]) for n in NAMES
                       if "categorical" in n and "unit" in n])

def action_embedding66(drugs, rates, doses):
    drugs = np.asarray(drugs)
    rates = np.asarray(rates, dtype=np.float32)
    doses = np.asarray(doses, dtype=np.float32)
    codes = torch.tensor([code42[str(d)] for d in drugs], dtype=torch.long)
    numeric = np.zeros((len(drugs), 6), np.float32)
    bins = {}
    for attr, values, pos in (("dose", doses, 2), ("rate", rates, 4)):
        bb = np.zeros(len(drugs), np.int64)
        for drug in ("insulin", "norepinephrine"):
            mask = drugs == drug
            meta = num42[(drug, attr)]
            numeric[mask, pos] = np.clip(
                (values[mask] - meta["median"]) / meta["iqr"], -8, 8)
            numeric[mask, pos + 1] = 1
            bb[mask] = np.searchsorted(meta["fine_bin_edges"],
                                       values[mask], side="left") + 1
        bins[attr] = torch.tensor(bb)
    ru = torch.tensor(np.where(drugs == "insulin", 3, 4), dtype=torch.long)
    du = torch.tensor(np.where(drugs == "insulin", 2, 3), dtype=torch.long)
    emb = (w63("embed.code.weight")[codes]
           + w63("embed.kind.weight")[0]
           + w63("embed.ekind.weight")[2]
           + linear63(torch.tensor(numeric), "embed.numeric")
           + w63("embed.bins.value.weight")[0]
           + w63("embed.bins.rate.weight")[bins["rate"]]
           + w63("embed.bins.dose.weight")[bins["dose"]]
           + w63("embed.categorical.rate_unit.weight")[ru]
           + w63("embed.categorical.dose_unit.weight")[du])
    return emb

emb66 = action_embedding66(drugs65, rates65, doses65)
u66 = encode_set50(emb66, owners65, counts65, "encode_action", False)
print("Explicit-unit action encoder RMS:",
      (u66-u_valid60).square().mean().sqrt().item())

def register66(events, last_values, ages_hours):
    events = np.asarray(events)
    last_values = np.asarray(last_values, dtype=float)
    ages_hours = np.asarray(ages_hours, dtype=float)
    present = np.isfinite(last_values)
    numeric = torch.zeros(len(events), 8)
    numeric[:, 0] = torch.tensor(present.astype(np.float32))
    numeric[:, 3] = torch.tensor(
        np.where(present, np.log1p(np.maximum(ages_hours, 0)*3600)/10, 0),
        dtype=torch.float32)
    bins = np.zeros(len(events), np.int64)
    for ev in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
        mask = (events == ev) & present
        meta = num42[(ev, "value")]
        numeric[mask, 6] = torch.tensor(np.clip(
            (last_values[mask]-meta["median"])/meta["iqr"], -8, 8),
            dtype=torch.float32)
        bins[mask] = np.searchsorted(meta["fine_bin_edges"],
                                     last_values[mask].astype(np.float32),
                                     side="left") + 1
    codes = torch.tensor([code42[str(ev)] for ev in events], dtype=torch.long)
    bins = torch.tensor(bins)
    const = (w63("embed.code.weight")[codes]
             + linear63(numeric, "last_numeric")
             + w63("embed.bins.value.weight")[bins]
             + w63("embed.bins.rate.weight")[0]
             + w63("last_status.weight")[0])
    trans = w63("embed_transition.value_bin.weight")[codes*17 + bins]
    centres = torch.tensor([[0.] + VOCAB["value_bin_centres"][str(ev)]
                            for ev in events], dtype=torch.float32)
    return const, trans, centres

def decode66(ctx, const, trans, centres, eps, return_features=False):
    prior = mlp63(ctx, "z_prior")
    z = prior[None, :, :32] + eps * prior[None, :, 32:].exp()
    view = ctx[None] + linear63(z, "z_head")
    q = norm63(view + w63("type_emb.weight")[1], "query_norm")
    hidden = F.gelu(linear63(q + const[None], "attr_heads.value_bin.0"))
    logits = linear63(hidden, "attr_heads.value_bin.2") + trans[None]
    probs = logits.softmax(-1)
    means = (probs*centres[None]).sum(-1)
    if return_features:
        return means, probs, hidden, logits
    return means.mean(0), probs.mean(0)

test_const66, test_trans66, test_centres66 = register66(
    ds22.event.to_numpy(), ds22.last_value22.to_numpy(),
    ds22.hour.to_numpy()-ds22.last_hour22.to_numpy())
print("Generic register RMS:", (test_const66-const_feat65).square().mean().sqrt().item())
print("Prior log-SD range:", prior_feat65[:, 32:].min().item(),
      prior_feat65[:, 32:].max().item())
gen66 = torch.Generator().manual_seed(66)
eps66 = torch.randn(64, len(ds22), 32, generator=gen66)
mean66, prob66 = decode66(view_feat65, test_const66, test_trans66,
                         test_centres66, eps66)
print("CPU/GPU predicted-mean RMSE (64 prior draws):",
      np.sqrt(np.mean((mean66.numpy()-ds22.predicted_mean.to_numpy())**2)))

measure66 = world51[world51.event.isin(
    ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"])].copy()
measure66 = measure66.sort_values(["stay_id", "hour", "event"]).reset_index(drop=True)
group66 = measure66.groupby(["stay_id", "event"])
measure66["last_value"] = group66.value.shift()
measure66["last_hour"] = group66.hour.shift()
preidx66 = np.full(len(measure66), -1, np.int64)
for stay66, rows66 in measure66.groupby("stay_id").groups.items():
    si66 = np.flatnonzero(sid63 == stay66)
    preidx66[rows66] = si66[
        np.searchsorted(hour63[si66], measure66.loc[rows66, "hour"], side="left")-1]
measure66["preidx"] = preidx66
const_train66, trans_train66, centres_train66 = register66(
    measure66.event.to_numpy(), measure66.last_value.to_numpy(),
    (measure66.hour-measure66.last_hour).to_numpy())
print("Training prediction rows:", len(measure66))
print("Next-reading rows after drug bundles:",
      np.isin(preidx66, drug_indices49).sum())
print("No edits:", edited())

with open("/work/notes.md", "a") as f66:
    f66.write("\n## Run 66 / generic helpers\n"
              "Generic unit defaults failed; implemented explicit I rate/dose units (3,2), "
              "NE (4,3), reusable decoder/registers, and next-reading alignment for training states. "
              "No tensors edited.\n")
```