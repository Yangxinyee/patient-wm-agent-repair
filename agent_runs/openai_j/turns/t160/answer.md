The old correction was fitted only at the latent mean, so it missed a major source of factual variation. I’ll now reconstruct whole-stay predictions with the evaluator’s four-sample probability averaging. Matching the original factual loss and row-level predictions will validate the inputs before another edit is attempted.

```python
import torch.nn.functional as F

# Recover bin conventions from already-matched training readings.
label145 = hm[["stay_id", "hour", "event", "target_bin"]].merge(
    train[["stay_id", "hour", "event", "value"]],
    on=["stay_id", "hour", "event"], how="left", validate="one_to_one")
binrule145 = {}
for ev145, g145 in label145.groupby("event"):
    edges145 = np.asarray(numeric_specs[(ev145, "value")]["fine_bin_edges"])
    choices145 = []
    for side145 in ("left", "right"):
        for offset145 in (0, 1):
            b145 = np.clip(np.searchsorted(edges145, g145.value, side=side145)
                           + offset145, 1, 16)
            choices145.append((np.mean(b145 == g145.target_bin), side145, offset145))
    best145 = max(choices145)
    assert best145[0] == 1., (ev145, choices145)
    binrule145[ev145] = best145[1:]
print("Verified bin rules:", binrule145)

def wm_meta145(records, state_data, factual):
    sg = {}
    for sid in np.unique(state_data["stay_id"]):
        ix = np.flatnonzero(state_data["stay_id"] == sid)
        sg[int(sid)] = (ix, np.asarray(state_data["hour"])[ix])
    rg = {key: g.sort_values("hour") for key, g in
          records[records.event.isin(binrule145)].groupby(["stay_id", "event"])}
    rows = []
    for r in factual.itertuples(index=False):
        ix, hours = sg[int(r.stay_id)]
        pos = np.searchsorted(hours, r.hour, side="left") - 1
        assert pos >= 0
        previous = rg[(r.stay_id, r.event)]
        j = np.searchsorted(previous.hour.to_numpy(), r.hour, side="left") - 1
        spec = numeric_specs[(r.event, "value")]
        edges = np.asarray(spec["fine_bin_edges"])
        side, offset = binrule145[r.event]
        has = int(j >= 0)
        val = float(previous.iloc[j].value) if has else 0.
        age = (r.hour - float(previous.iloc[j].hour)) * 3600 if has else 0.
        zlast = np.clip((val - spec["median"]) / spec["iqr"], -10, 10) if has else 0.
        lastbin = int(np.clip(np.searchsorted(edges, val, side=side) + offset,
                              1, 16)) if has else 0
        target = int(np.clip(np.searchsorted(edges, r.observed, side=side) + offset,
                             1, 16))
        rows.append((r.stay_id, r.hour, r.event, int(ix[pos]),
                     (r.hour - hours[pos]) * 3600, has, age, zlast, lastbin,
                     target, r.predicted_mean, r.nll, spec["iqr"], code_ids[r.event]))
    return pd.DataFrame(rows, columns=hm.columns)

train_gpu145 = pd.read_csv("/work/fwd_4/factual_original.csv")
assert len(train_gpu145) == len(hm)
rebuilt_train145 = wm_meta145(train, repair_states, train_gpu145)
check145 = hm.merge(rebuilt_train145, on=["stay_id", "hour", "event"],
                    suffixes=("_old", "_new"), validate="one_to_one")
print("Metadata reconstruction errors:", {
    c: float(np.max(np.abs(check145[c+"_old"] - check145[c+"_new"])))
    for c in ("state_i", "dt", "has_last", "age_s", "zlast", "last_bin", "target_bin")})
assert np.max(np.abs(check145.state_i_old - check145.state_i_new)) == 0
assert np.max(np.abs(check145.zlast_old - check145.zlast_new)) < 1e-5

wm_tune_meta145 = wm_meta145(tuning, tune_states, tune_factual_original)

wm_weights145 = {n: torch.as_tensor(load_original(n)).float()
                 for n in audit_original_weights}
for n145, t145 in wm_weights145.items():
    assert torch.max(torch.abs(t145 - wt(n145))) < 1e-7

def wm_mc145(meta, state_data, draws=16, seed=145):
    raw = reconstruct_reading_head(meta, state_array=state_data)
    numbers = torch.zeros_like(raw["numbers"])
    numbers[:, 0] = torch.as_tensor(meta.has_last.to_numpy(), dtype=torch.float32)
    numbers[:, 3] = torch.log1p(torch.as_tensor(meta.age_s.to_numpy(),
                                              dtype=torch.float32)) / 10
    numbers[:, 6] = torch.as_tensor(meta.zlast.to_numpy(),
                                    dtype=torch.float32).clamp(-10, 10)
    inputs0 = raw["inputs"] + F.linear(numbers - raw["numbers"],
                                       wt("last_numeric.weight"), None)
    prior = mlp(raw["context"], "z_prior", "gelu")
    noise = torch.randn(draws, len(meta), 32,
                        generator=torch.Generator().manual_seed(seed))
    view = raw["view"][None] + F.linear(
        noise * prior[:, 32:].exp()[None], wt("z_head.weight"), None)
    q0 = ln(raw["view"] + wt("type_emb.weight")[1], "query_norm")
    inputs = inputs0[None] + ln(view + wt("type_emb.weight")[1], "query_norm") - q0[None]
    hidden = F.gelu(F.linear(inputs, wm_weights145["attr_heads.value_bin.0.weight"],
                             wm_weights145["attr_heads.value_bin.0.bias"]))
    transition = wt("embed_transition.value_bin.weight")[
        raw["code"].long() * 17 + raw["last_bin"].long()]
    logits = F.linear(hidden, wm_weights145["attr_heads.value_bin.2.weight"],
                       wm_weights145["attr_heads.value_bin.2.bias"]) + transition[None]
    probs = logits.softmax(-1)
    centres = torch.as_tensor(repair_centres17(meta), dtype=torch.float32)
    targets = torch.as_tensor(meta.target_bin.to_numpy(), dtype=torch.long)
    p4 = probs.reshape(draws//4, 4, len(meta), 17).mean(1)
    nll = -p4.gather(-1, targets[None, :, None].expand(draws//4, -1, 1)).squeeze(-1).log()
    return dict(raw=raw, inputs0=inputs0, inputs=inputs, hidden=hidden,
                logits=logits, probs=probs, centres=centres, target=targets,
                mean=(probs.mean(0)*centres).sum(-1), nll=nll, noise=noise)

wm_train145 = wm_mc145(hm, repair_states["x"])
wm_tune145 = wm_mc145(wm_tune_meta145, tune_states["x"], seed=146)

print("\n=== COMPLETE-STAY ORIGINAL AUDIT ===")
for label145, meta145, mc145 in [
    ("train", hm, wm_train145), ("tuning", wm_tune_meta145, wm_tune145)]:
    pred145 = mc145["mean"].numpy()
    print(label145, "CPU NLL", float(mc145["nll"].mean()),
          "GPU NLL", float(meta145.original_nll.mean()),
          "scaled mean RMSE",
          float(np.sqrt(np.mean(((pred145-meta145.original_mean)/meta145.scale)**2))))
    print(pd.DataFrame({
        "event": meta145.event,
        "CPU_nll": mc145["nll"].mean(0).numpy(),
        "GPU_nll": meta145.original_nll,
        "scaled_bias": (pred145-meta145.original_mean)/meta145.scale,
    }).groupby("event").mean().to_string())

print("\n=== ORIGINAL ROLLOUT MEANS, EQUAL WEIGHT PER STAY ===")
d145 = dosesummary144[dosesummary144.reached & dosesummary144.hour.between(13, 24)]
print(d145.groupby(["edit", "stay_id"]).shift.mean().groupby("edit").mean())

print("\n=== LOCAL CONTRAST STRUCTURE ===")
for key145, val145 in r117_cf.items():
    print(key145, {kk: (tuple(vv.shape) if isinstance(vv, (torch.Tensor, np.ndarray))
                         else list(vv.keys()) if isinstance(vv, dict) else type(vv).__name__)
                    for kk, vv in val145.items()})
```