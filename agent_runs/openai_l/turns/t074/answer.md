The check caught a unit-table indexing error before any weights changed: dose units and rate units use different tables. I’ll correct that mapping and repeat the verification. The explicit encoder should match the validated action vectors to numerical precision before I use it to generate counterfactual features.

```python
def action_embedding67(drugs, rates, doses):
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
                (values[mask]-meta["median"])/meta["iqr"], -10, 10)
            numeric[mask, pos+1] = 1
            bb[mask] = np.searchsorted(meta["fine_bin_edges"],
                                       values[mask], side="left") + 1
        bins[attr] = torch.tensor(bb)
    unit = torch.tensor(np.where(drugs == "insulin", 3, 4), dtype=torch.long)
    rate_unit = torch.tensor(np.where(drugs == "insulin", 2, 3), dtype=torch.long)
    return (w63("embed.code.weight")[codes]
            + w63("embed.kind.weight")[0]
            + w63("embed.ekind.weight")[2]
            + w63("embed.kq.weight")[0]
            + linear63(torch.tensor(numeric), "embed.numeric")
            + w63("embed.bins.value.weight")[0]
            + w63("embed.bins.rate.weight")[bins["rate"]]
            + w63("embed.bins.dose.weight")[bins["dose"]]
            + w63("embed.categorical.unit.weight")[unit]
            + w63("embed.categorical.rate_unit.weight")[rate_unit]
            + w63("embed.categorical.action.weight")[0])

emb67 = action_embedding67(drugs65, rates65, doses65)
u67 = encode_set50(emb67, owners65, counts65, "encode_action", False)
print("Explicit action encoder RMS:",
      (u67-u_valid60).square().mean().sqrt().item())

def register67(events, last_values, ages_hours):
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
            (last_values[mask]-meta["median"])/meta["iqr"], -10, 10),
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

def decode67(ctx, const, trans, centres, eps, return_features=False):
    prior = mlp63(ctx, "z_prior")
    z = prior[None, :, :32] + eps*prior[None, :, 32:].exp()
    view = ctx[None] + linear63(z, "z_head")
    q = norm63(view + w63("type_emb.weight")[1], "query_norm")
    hidden = F.gelu(linear63(q + const[None], "attr_heads.value_bin.0"))
    logits = linear63(hidden, "attr_heads.value_bin.2") + trans[None]
    probs = logits.softmax(-1)
    means = (probs*centres[None]).sum(-1)
    if return_features:
        return means, probs, hidden, logits
    return means.mean(0), probs.mean(0)

test_const67, test_trans67, test_centres67 = register67(
    ds22.event.to_numpy(), ds22.last_value22.to_numpy(),
    ds22.hour.to_numpy()-ds22.last_hour22.to_numpy())
print("Generic register RMS:", (test_const67-const_feat65).square().mean().sqrt().item())
print("Prior log-SD range:", prior_feat65[:, 32:].min().item(),
      prior_feat65[:, 32:].max().item())
gen67 = torch.Generator().manual_seed(67)
eps67 = torch.randn(64, len(ds22), 32, generator=gen67)
mean67, prob67 = decode67(view_feat65, test_const67, test_trans67,
                         test_centres67, eps67)
print("CPU/GPU mean RMSE (64 prior draws):",
      np.sqrt(np.mean((mean67.numpy()-ds22.predicted_mean.to_numpy())**2)))

measure67 = world51[world51.event.isin(
    ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"])].copy()
measure67 = measure67.sort_values(["stay_id", "hour", "event"]).reset_index(drop=True)
group67 = measure67.groupby(["stay_id", "event"])
measure67["last_value"] = group67.value.shift()
measure67["last_hour"] = group67.hour.shift()
preidx67 = np.full(len(measure67), -1, np.int64)
for stay67, rows67 in measure67.groupby("stay_id").groups.items():
    si67 = np.flatnonzero(sid63 == stay67)
    positions67 = np.searchsorted(hour63[si67],
                                 measure67.loc[rows67, "hour"], side="left")-1
    assert np.all(positions67 >= 0)
    preidx67[rows67] = si67[positions67]
measure67["preidx"] = preidx67
const_train67, trans_train67, centres_train67 = register67(
    measure67.event.to_numpy(), measure67.last_value.to_numpy(),
    (measure67.hour-measure67.last_hour).to_numpy())
print("Training prediction rows:", len(measure67))
print("Next-reading rows after drug bundles:",
      np.isin(preidx67, drug_indices49).sum())
print("No edits:", edited())

with open("/work/notes.md", "a") as f67:
    f67.write("\n## Runs 66–67\n"
              "Run 66 stopped before editing because rate/dose unit table indices were reversed. "
              "Corrected dose-unit table I=3/NE=4, rate-unit table I=2/NE=3; "
              "verifying generic encoder, decoder and prediction-row alignment. No edits.\n")
```