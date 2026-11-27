Changing type/status indices or age units did not reproduce the forward tool’s predictions, so those explanations are not supported. I’ll now test the eight-field last-reading input layout and the value-head activation. A correct layout should remove the large stale-reading errors without changing any weights.

```python
import torch.nn.functional as F

print("=== ACTIVATION / PRIOR DIAGNOSTICS ===")
print("act0 constants", act0.__code__.co_consts, "globals", act0.__code__.co_names)
print("last_numeric column norms", torch.linalg.vector_norm(W0["last_numeric.weight"], dim=0))
print("prior mean quantiles", torch.quantile(mu104.flatten(), torch.tensor([0., .1, .5, .9, 1.])))
print("prior log-sd quantiles", torch.quantile(ls104.flatten(), torch.tensor([0., .1, .5, .9, 1.])))
print("normalisation note", VOCAB["numeric_note"])

q105 = norm0(ctx104[:, None] + zh104 + W0["type_emb.weight"][1], "query_norm")
base105 = q105 + (codeemb104 + binemb104)[:, None] \
          + W0["embed.bins.rate.weight"][0] + W0["last_status.weight"][0]
age105 = np.where(has104, np.log1p(
    np.maximum(probe104.hour.to_numpy() - regs104[:, 1], 0) * 3600) / 10, 0)
results105 = []
prob_best105 = None
besterr105 = float("inf")
for zcol105 in range(2, 8):
    for flagcol105 in [None] + [c for c in range(2, 8) if c != zcol105]:
        numeric105 = np.zeros((len(probe104), 8), dtype=np.float32)
        numeric105[:, 0] = has104
        numeric105[:, 1] = age105
        numeric105[:, zcol105] = zval104
        if flagcol105 is not None:
            numeric105[:, flagcol105] = has104
        wc105 = base105 + lin0(torch.tensor(numeric105), "last_numeric")[:, None]
        a105 = lin0(wc105, "attr_heads.value_bin.0")
        for activation105, hf105 in (("gelu", F.gelu), ("silu", F.silu), ("relu", F.relu)):
            probs105 = torch.softmax(
                lin0(hf105(a105), "attr_heads.value_bin.2") + trans104[:, None], -1).mean(1)
            means105 = (probs105 * centres104).sum(-1).numpy()
            nll105 = -torch.log(probs105[
                torch.arange(len(probe104)), torch.tensor(label104)].clamp_min(1e-30)).numpy()
            err105 = float(np.mean(abs(nll105 - probe104.nll.to_numpy())))
            results105.append((zcol105, flagcol105, activation105,
                               float(nll105.mean()),
                               float(np.mean(abs(means105 - probe104.predicted_mean.to_numpy()))),
                               err105))
            if err105 < besterr105:
                besterr105 = err105
                prob_best105 = probs105.clone()
print("z-value column, flag column, activation, NLL, mean MAE, NLL MAE:")
for row105 in sorted(results105, key=lambda r: r[5])[:12]:
    print(row105)
with open("/work/notes.md", "a") as f105:
    f105.write("\n## Run 105: last-numeric layout experiment\n"
               "- Type/status and age-unit changes did not resolve the local/external prediction mismatch.\n"
               "- Testing last-reading numeric field positions and head activations, without editing weights.\n"
               "- Python runs: 105; forward passes: 4; no submitted edits.\n")
```
