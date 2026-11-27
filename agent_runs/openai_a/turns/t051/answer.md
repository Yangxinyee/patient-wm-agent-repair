Separating the gap scaling did not fix the mismatch; the CPU head remains approximate. I’ll use it only to construct a small candidate correction, with the forward tool deciding factual safety. First I’ll align each reading with its preceding dose and recover the training-state IDs. If dose innovations cannot be decoded reliably, this approach should be rejected.

```python
print("=== PANEL ALIGNMENT ===")
print("event_col:", event_col)
print("hour_grid:", hour_grid)
print("train_ids:", train_ids[:3], train_ids[-3:])
print("tuning_ids:", tuning_ids[:3], tuning_ids[-3:])
print("P:", P.shape, P.dtype, "TP:", TP.shape, TP.dtype)
print("state-sampled training IDs:")
_repair_train_stays = np.unique(state_files[4]["stay_id"]).astype(int)
print(json.dumps(_repair_train_stays.tolist()))
print("training metadata sample:")
print(head_meta_train.head(4).to_string(index=False))

_repair_bin_score = torch.cat((torch.zeros(1), torch.arange(16).float() - 7.5))

def repair_head_features(meta, states):
    parts = cpu_head_parts(meta, states)
    last_z = torch.tensor(meta.last_z.to_numpy(), dtype=torch.float32)
    additive = parts["additive"] + last_z[:, None] * (
        W["last_numeric.weight"][:, 6] - W["last_numeric.weight"][:, 2]
    )[None, :]
    q = ln(
        parts["base"] + lin(torch.zeros(len(meta), 32), "z_head")
        + W["type_emb.weight"][1], "query_norm"
    )
    hidden = F.gelu(lin(q + additive, "attr_heads.value_bin.0"))
    logits = lin(hidden, "attr_heads.value_bin.2") + parts["transition"]
    probs = logits.softmax(-1)
    centres = torch.tensor(np.stack([
        VOCAB["value_bin_centres"][e] for e in meta.event
    ]), dtype=torch.float32)
    means = (probs * centres).sum(-1)
    score_mean = (probs * _repair_bin_score).sum(-1)
    covariance = (probs * centres * _repair_bin_score).sum(-1) - means * score_mean
    return {
        "hidden": hidden.numpy(), "probs": probs.numpy(),
        "mean": means.numpy(), "covariance": covariance.numpy()
    }

_repair_train_head = repair_head_features(head_meta_train, state_files[4])
_repair_tune_head = repair_head_features(reference_meta, state_files[3])
print("\n=== APPROXIMATE HEAD SENSITIVITY ===")
for _e in reference_meta.event.unique():
    _m = reference_meta.event.to_numpy() == _e
    print(_e, {
        "covariance_median": float(np.median(_repair_tune_head["covariance"][_m])),
        "mean_rmse_vs_GPU": float(np.sqrt(np.mean(
            (_repair_tune_head["mean"][_m] -
             np.asarray(as_numpy(head_gpu_mean))[_m]) ** 2
        )))
    })
```
