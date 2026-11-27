Neither query type nor “no value” probability explains the mismatch. The local head also hard-codes left-sided bin boundaries. Since the readings are rounded, equality with a bin edge may matter. I’ll test both conventions for the last-reading bins and target labels; a close match would validate the scoring pipeline without changing weights.

```python
print("core input bin convention:", world_bin_side_audit)

lastbins112, labels112 = {}, {}
for side112 in ("left", "right"):
    lb112 = np.zeros(len(ix104), dtype=np.int64)
    yy112 = np.zeros(len(ix104), dtype=np.int64)
    for j112, ev112 in enumerate(r49_sensors):
        mk112 = js104 == j112
        edges112 = np.asarray(num_spec0[ev112]["value"]["fine_bin_edges"])
        lb112[mk112] = np.where(
            has104[mk112],
            1 + np.searchsorted(edges112, regs104[mk112, 0], side=side112), 0)
        yy112[mk112] = 1 + np.searchsorted(
            edges112, probe104.observed.to_numpy()[mk112], side=side112)
    lastbins112[side112] = lb112
    labels112[side112] = yy112
print("last bins changed:", int(np.sum(lastbins112["left"] != lastbins112["right"])))
print("target bins changed:", int(np.sum(labels112["left"] != labels112["right"])))
print("old target agreement:",
      {s: int(np.sum(labels112[s] == label104)) for s in labels112})

query112 = norm0(ctx104[:, None, :] + zh104 +
                  W0["type_emb.weight"][1], "query_norm")
def bin_assay112(last_side, value_column, activation):
    lb = torch.tensor(lastbins112[last_side])
    numeric = np.zeros((len(ix104), 8), dtype=np.float32)
    numeric[:, 0] = has104
    numeric[:, 1] = age105
    numeric[:, value_column] = zval104
    numeric[:, 3] = has104
    wc = (query112 + codeemb104[:, None, :] +
          W0["embed.bins.value.weight"][lb][:, None, :] +
          W0["embed.bins.rate.weight"][0] + W0["last_status.weight"][0] +
          lin0(torch.tensor(numeric), "last_numeric")[:, None, :])
    hidden = act0(F.linear(wc, W0["attr_heads.value_bin.0.weight"],
                           W0["attr_heads.value_bin.0.bias"]), activation)
    logits = F.linear(hidden, W0["attr_heads.value_bin.2.weight"],
                      W0["attr_heads.value_bin.2.bias"])
    logits += W0["embed_transition.value_bin.weight"][
        torch.tensor(codes104) * 17 + lb][:, None, :]
    return torch.softmax(logits, -1).mean(1)

results112 = []
for lside112 in ("left", "right"):
    for vc112 in (2, 6):
        for ac112 in ("gelu", "relu"):
            pp112 = bin_assay112(lside112, vc112, ac112)
            mm112 = (pp112 * centres104).sum(-1).numpy()
            for tside112 in ("left", "right"):
                nn112 = -torch.log(pp112[
                    torch.arange(len(ix104)), torch.tensor(labels112[tside112])]
                    .clamp_min(1e-12)).numpy()
                results112.append({
                    "last_side": lside112, "target_side": tside112,
                    "value_col": vc112, "activation": ac112,
                    "nll": float(nn112.mean()),
                    "mean_mae": float(np.abs(mm112 - probe104.predicted_mean).mean()),
                    "nll_mae": float(np.abs(nn112 - probe104.nll).mean())
                })
rank112 = sorted(results112, key=lambda r: r["nll_mae"])
print("\n=== BIN-CONVENTION RESULTS ===")
for rr112 in rank112[:8]:
    print(rr112)

best112 = rank112[0]
p112 = bin_assay112(best112["last_side"], best112["value_col"],
                   best112["activation"])
err112 = probe104[["event"]].copy()
err112["mean_error"] = (p112 * centres104).sum(-1).numpy() - probe104.predicted_mean
err112["nll_error"] = -torch.log(p112[torch.arange(len(ix104)),
    torch.tensor(labels112[best112["target_side"]])].clamp_min(1e-12)).numpy() - probe104.nll
print(err112.groupby("event").agg(
    mean_mae=("mean_error", lambda x: np.abs(x).mean()),
    nll_mae=("nll_error", lambda x: np.abs(x).mean())).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 112: fine-bin boundary assay\n"
            f"- Best boundary/layout result: {best112}\n"
            "- Tested last-reading and target-label boundaries separately; no edits.\n"
            "- Python runs: 112; forward passes: 4.\n")
```