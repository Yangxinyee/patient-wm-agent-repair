The original factual value NLL is 1.63065 on these training stays. I’ll now reconstruct its value predictions from the exported states and weights, without gradients, and compare them with the forward tool’s results. Close agreement would let me test algebraic head edits cheaply; substantial disagreement would mean the reconstruction is not trustworthy.

```python
python_runs += 1
forward_passes = 3
import torch.nn.functional as TF

factual_reference = pd.read_csv("/work/fwd_3/factual_original.csv")
print("=== FACTUAL FILE ===")
print(factual_reference.columns.tolist())
print(factual_reference.head(4).to_string(index=False))
print("Extra normalization tensors:",
      [(n, SHAPES[n]) for n in NAMES if n.startswith("out_norm.")])
print("Last-numeric column norms:",
      np.round(np.linalg.norm(np.asarray(load_original("last_numeric.weight")), axis=0), 4))
for n in ("embed.bins.value.weight", "embed.bins.rate.weight", "last_status.weight"):
    print(n, "first row norms:",
          np.round(np.linalg.norm(np.asarray(load_original(n)), axis=1)[:3], 4))

W = {n: torch.as_tensor(np.asarray(load_original(n))).float() for n in NAMES}
def lin(x, prefix):
    return TF.linear(x, W[prefix+".weight"], W.get(prefix+".bias"))
def norm(x, prefix):
    return TF.layer_norm(x, (x.shape[-1],), W[prefix+".weight"], W[prefix+".bias"])
def activation(x, kind):
    return TF.silu(x) if kind == "silu" else TF.gelu(x)
def mlp(x, prefix, kind):
    return lin(activation(lin(x, prefix+".0"), kind), prefix+".2")

code_id = {x["name"]: x["id"] for x in VOCAB["codes"]}
numeric_meta = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
def fine_bin(event, value, attr="value"):
    return 1 + np.searchsorted(numeric_meta[(event, attr)]["fine_bin_edges"], value, side="left")

head_records = train[
    train.stay_id.isin(np.unique(S["stay_id"])) & train.event.isin(sensors)
].sort_values(["stay_id", "hour", "event"]).reset_index(drop=True)

state_lookup = {}
for sid in np.unique(S["stay_id"]):
    indices = np.flatnonzero(S["stay_id"] == sid)
    state_lookup[int(sid)] = (S["hour"][indices], indices)

meta_rows = []
for sid, group in head_records.groupby("stay_id", sort=False, observed=True):
    sh, sidx = state_lookup[int(sid)]
    previous = {}
    for row in group.itertuples(index=False):
        loc = np.searchsorted(sh, row.hour, side="left") - 1
        assert loc >= 0
        prev_idx = sidx[loc]
        if row.event in previous:
            last_hour, last_value = previous[row.event]
            nv = numeric_meta[(row.event, "value")]
            zlast = np.clip((last_value-nv["median"])/nv["iqr"], -10, 10)
            lb = fine_bin(row.event, last_value)
            haslast, age_seconds = 1., (row.hour-last_hour)*3600.
        else:
            zlast, lb, haslast, age_seconds = 0., 0, 0., 0.
        meta_rows.append((prev_idx, (row.hour-sh[loc])*3600., code_id[row.event],
                          haslast, age_seconds, zlast, lb,
                          fine_bin(row.event, row.value)))
        previous[row.event] = (row.hour, row.value)
HM = np.asarray(meta_rows)
hx = torch.from_numpy(S["x"][HM[:, 0].astype(int)])
hdt = torch.from_numpy(HM[:, 1]).float()
hcode = torch.from_numpy(HM[:, 2].astype(int))
hlastbin = torch.from_numpy(HM[:, 6].astype(int))
htruebin = torch.from_numpy(HM[:, 7].astype(int))
hcentres = torch.from_numpy(np.stack([
    [0.] + list(VOCAB["value_bin_centres"][e]) for e in head_records.event
])).float()

# Align the external reference by stay/hour/event, rather than relying on row order.
ref_mean_col = next((c for c in ("pred_mean", "mean", "predicted_mean", "prediction")
                    if c in factual_reference), None)
ref_nll_col = next((c for c in ("value_nll", "nll", "fine_bin_nll")
                   if c in factual_reference), None)
if ref_mean_col is None:
    ref_mean_col = next(c for c in factual_reference if "mean" in c)
if ref_nll_col is None:
    ref_nll_col = next(c for c in factual_reference if "nll" in c)
aligned_ref = head_records[["stay_id", "hour", "event"]].merge(
    factual_reference[["stay_id", "hour", "event", ref_mean_col, ref_nll_col]],
    on=["stay_id", "hour", "event"], how="left", validate="one_to_one"
)
refmean = aligned_ref[ref_mean_col].to_numpy()
refnll = aligned_ref[ref_nll_col].to_numpy()
assert np.isfinite(refmean).all()

view_norm_name = "out_norm" if "out_norm.weight" in W else "core.out_norm"
hflow = hx[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -hdt[:, None, None]*torch.exp(W["core.flow.a"])[None, None, :]
)
base_ctx = hx[:, :192] + norm(hflow.mean(dim=1), view_norm_name)

grid_results = []
head_test_predictions = {}
for gap_act in ("silu", "gelu"):
    for log_kind in ("log", "log1p"):
        tg = torch.log(hdt.clamp_min(1.)) if log_kind == "log" else torch.log1p(hdt)
        ctx = base_ctx + mlp((tg/10.)[:, None], "gap", gap_act)
        for prior_act in ("silu", "gelu"):
            prior = mlp(ctx, "z_prior", prior_act)
            view = ctx + lin(prior[:, :32], "z_head")
            q = norm(view + W["type_emb.weight"][1], "query_norm")
            for last_layout in ("interleaved", "grouped"):
                lastnum = torch.zeros(len(HM), 8)
                lastnum[:, 0] = torch.from_numpy(HM[:, 3]).float()
                lastnum[:, 1] = torch.log1p(torch.from_numpy(HM[:, 4]).float())/10.
                lastnum[:, 2] = torch.from_numpy(HM[:, 5]).float()
                lastnum[:, 3 if last_layout == "interleaved" else 5] = lastnum[:, 0]
                for status in (0, 1):
                    withcode = (q + W["embed.code.weight"][hcode]
                                + lin(lastnum, "last_numeric")
                                + W["embed.bins.value.weight"][hlastbin]
                                + W["embed.bins.rate.weight"][0]
                                + W["last_status.weight"][status])
                    for value_act in ("silu", "gelu"):
                        hidden = activation(lin(withcode, "attr_heads.value_bin.0"), value_act)
                        logits = (lin(hidden, "attr_heads.value_bin.2")
                                  + W["embed_transition.value_bin.weight"][hcode*17+hlastbin])
                        probs = logits.softmax(dim=1)
                        predmean = (probs*hcentres).sum(dim=1).numpy()
                        prednll = -probs[np.arange(len(HM)), htruebin].clamp_min(1e-30).log().numpy()
                        key = (gap_act, log_kind, prior_act, last_layout, status, value_act)
                        result = dict(config=str(key),
                                      mean_rmse=float(np.sqrt(np.mean((predmean-refmean)**2))),
                                      nll_rmse=float(np.sqrt(np.mean((prednll-refnll)**2))),
                                      nll=float(prednll.mean()))
                        grid_results.append(result)
                        head_test_predictions[key] = (predmean, prednll)
best_grid = sorted(grid_results, key=lambda x: x["mean_rmse"])
print("\n=== RECONSTRUCTION CANDIDATES ===")
print(pd.DataFrame(best_grid[:8]).round(6).to_string(index=False))

with notes_path.open("a") as f:
    f.write(
        "\n## Forward 3 / Run 6\n"
        "Original factual NLL on matching 100 training stays: 1.63065 (4745 values). "
        "Reconstructing prediction heads from exported states; compared candidate "
        "activations, gap encodings, and last-register layouts. No edits yet.\n"
        + pd.DataFrame(best_grid[:8]).to_string(index=False)
        + f"\nCounts: {python_runs} Python runs, {forward_passes} forward passes.\n"
    )
```
