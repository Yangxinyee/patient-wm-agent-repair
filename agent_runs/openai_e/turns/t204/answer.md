One numeric projection agrees much better with the GPU. Using the properly scaled log-gap also resolves much of the earlier NLL mismatch. I’ll now compare paired original and edited training states with common prior draws. Agreement in aggregate would support a small factual calibration; a large unexplained mismatch would rule it out.

```python
torch.set_num_threads(8)
print("TRAIN ARCHIVE FIELDS",
      [(k, train_original168[k].shape) for k in train_original168.files])
print("TRAIN METADATA", train_meta162.head(2).to_string(index=False))

def decode_states175(archive, meta, static, transition, centres, draws=16, seed=175):
    key = next(k for k in archive.files
               if archive[k].ndim == 2 and archive[k].shape[1] == 992)
    rows = meta.state_row.to_numpy().astype(np.int64)
    x = torch.as_tensor(archive[key][rows], dtype=torch.float32)
    dt = torch.as_tensor(meta.dt.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    h = x[:, 192:960].reshape(-1, 4, 192)
    flowed = h * torch.exp(-dt[:, :, None] *
                           torch.exp(W0["core.flow.a"])[None, None, :])
    ctx = x[:, :192] + norm0(flowed.mean(1), "core.out_norm")
    ctx = ctx + mlp0(torch.log(dt.clamp_min(1e-6))/10., "gap")
    zp = mlp0(ctx, "z_prior")
    gen = torch.Generator().manual_seed(seed)
    eps = torch.randn((draws, len(meta), 32), generator=gen)
    probability = torch.zeros((len(meta), 17))
    hidden_sum = torch.zeros((len(meta), W0["attr_heads.value_bin.2.weight"].shape[1]))
    logits_sum = torch.zeros((len(meta), 17))
    for j in range(draws):
        z = zp[:, :32] + zp[:, 32:].clamp(-7., 2.).exp()*eps[j]
        q = norm0(ctx + linear0(z, "z_head") + W0["type_emb.weight"][1],
                  "query_norm")
        hidden = F.gelu(linear0(q + static, "attr_heads.value_bin.0"))
        logits = linear0(hidden, "attr_heads.value_bin.2") + transition
        probability += logits.softmax(-1)/draws
        hidden_sum += hidden/draws
        logits_sum += logits/draws
    return probability.numpy(), hidden_sum.numpy(), logits_sum.numpy()

pt_orig175, ht_orig175, lt_orig175 = decode_states175(
    train_original168, train_meta162, train_static168,
    train_transition168, train_centres168)
pt_edit175, ht_edit175, lt_edit175 = decode_states175(
    train_edited168, train_meta162, train_static168,
    train_transition168, train_centres168)
yt175 = train_meta162.target_bin.to_numpy().astype(int)
for label, p in (("original", pt_orig175), ("insulin_baseline", pt_edit175)):
    losses175 = -np.log(p[np.arange(len(p)), yt175].clip(1e-12))
    print(label, "training-state NLL", float(losses175.mean()))
    print(pd.DataFrame({"event": train_meta162.event, "nll": losses175})
          .groupby("event").nll.mean().to_string())

pt_tune175, _, _ = decode_states175(
    tune_archive162, aligned_meta168, static168,
    transition168, head_centres)
nll_tune175 = -np.log(pt_tune175[np.arange(len(b168)), b168].clip(1e-12))
mean_tune175 = (pt_tune175*head_centres.numpy()).sum(1)
print("TUNING ORIGINAL MC",
      {"NLL": float(nll_tune175.mean()),
       "GPU_NLL": float(gpu168.nll.mean()),
       "mean_RMSE": float(np.sqrt(np.mean((mean_tune175-gpu168.predicted_mean)**2)))})
print("PAIRED TRAIN DELTA",
      float(np.mean(-np.log(pt_edit175[np.arange(len(yt175)), yt175]) +
                    np.log(pt_orig175[np.arange(len(yt175)), yt175]))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 175: corrected local decoder and paired training distributions\n"
            "One numeric projection is correct by GPU agreement. Proper log(dt)/10 query gives deterministic "
            "original NLL=1.62508 vs GPU=1.63461 and mean RMSE=1.565; duplicate projection is worse. "
            "Built common-prior-draw original/insulin-baseline training probabilities for calibration diagnostics. "
            "No weights changed. Python runs: 175; forwards: 22.\n")
```