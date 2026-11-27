The scratch weights are original, and the last-reading registers match the records. The mismatch therefore appears to be in the local head implementation. I’ll test type/status indices and age units against the saved factual predictions; a clear improvement in both means and NLLs would identify the missing convention.

```python
print("=== HEAD CONVENTIONS ===")
print("atom kinds", VOCAB["atom_kinds"], "event kinds", VOCAB["event_kinds"])
for n104 in ("type_emb.weight", "last_status.weight", "last_numeric.weight",
             "last_numeric.bias", "attr_heads.value_bin.0.weight",
             "attr_heads.value_bin.2.weight"):
    print(n104, tuple(W0[n104].shape))
print("mlp constants", mlp0.__code__.co_consts, "globals", mlp0.__code__.co_names)
print("norm constants", norm0.__code__.co_consts)

probe104 = direct98.sample(n=min(384, len(direct98)), random_state=104).copy()
ix104 = idx80[probe104.position.to_numpy(dtype=int)]
x104 = torch.tensor(sfile101["x"][ix104], dtype=torch.float32)
dt104 = torch.tensor((probe104.hour.to_numpy() - r49_hours[ix104]) * 3600,
                     dtype=torch.float32)
flow104 = x104[:, 192:960].reshape(-1, 4, 192) * torch.exp(
    -dt104[:, None, None] * torch.exp(W0["core.flow.a"]))
ctx104 = x104[:, :192] + norm0(flow104.mean(1), "core.out_norm") \
         + mlp0(torch.log(dt104)[:, None] / 10, "gap")
mu104, ls104 = mlp0(ctx104, "z_prior").chunk(2, -1)
zh104 = lin0(mu104[:, None] + torch.exp(ls104)[:, None] * r49_noise[None], "z_head")
codes104 = np.array([code_id0[e] for e in probe104.event], dtype=int)
js104 = np.array([r49_sensors.index(e) for e in probe104.event], dtype=int)
regs104 = r49_register[ix104, js104]
has104 = np.isfinite(regs104[:, 0])
lastbin104 = np.zeros(len(probe104), dtype=int)
label104 = np.zeros(len(probe104), dtype=int)
zval104 = np.zeros(len(probe104), dtype=np.float32)
for j104, e104 in enumerate(r49_sensors):
    m104 = js104 == j104
    sp104 = num_spec0[(e104, "value")]
    lastbin104[m104] = np.where(
        has104[m104],
        1 + np.searchsorted(sp104["fine_bin_edges"], regs104[m104, 0], side="left"), 0)
    label104[m104] = 1 + np.searchsorted(
        sp104["fine_bin_edges"], probe104.observed.to_numpy()[m104], side="left")
    zval104[m104] = np.where(
        has104[m104], np.clip((regs104[m104, 0] - sp104["median"]) / sp104["iqr"], -8, 8), 0)

codeemb104 = W0["embed.code.weight"][torch.tensor(codes104)]
binemb104 = W0["embed.bins.value.weight"][torch.tensor(lastbin104)]
trans104 = W0["embed_transition.value_bin.weight"][
    torch.tensor(codes104 * 17 + lastbin104)]
centres104 = r49_centres[torch.tensor(js104)]
results104 = []
for ty104 in range(len(W0["type_emb.weight"])):
    q104 = norm0(ctx104[:, None] + zh104 + W0["type_emb.weight"][ty104], "query_norm")
    for age_scale104 in (1.0, 60.0, 3600.0):
        numeric104 = np.zeros((len(probe104), 8), dtype=np.float32)
        numeric104[:, 0] = has104
        numeric104[:, 1] = np.where(
            has104, np.log1p(np.maximum(probe104.hour.to_numpy() - regs104[:, 1], 0)
                            * age_scale104) / 10, 0)
        numeric104[:, 2] = zval104
        numeric104[:, 3] = has104
        base104 = q104 + (codeemb104 + binemb104
                         + lin0(torch.tensor(numeric104), "last_numeric"))[:, None] \
                         + W0["embed.bins.rate.weight"][0]
        for status104 in range(len(W0["last_status.weight"])):
            logits104 = mlp0(base104 + W0["last_status.weight"][status104],
                            "attr_heads.value_bin") + trans104[:, None]
            probs104 = torch.softmax(logits104, -1).mean(1)
            means104 = (probs104 * centres104).sum(-1).numpy()
            nll104 = -torch.log(probs104[
                torch.arange(len(probe104)), torch.tensor(label104)].clamp_min(1e-30)).numpy()
            results104.append((ty104, age_scale104, status104,
                               float(nll104.mean()),
                               float(np.mean(abs(means104 - probe104.predicted_mean.to_numpy()))),
                               float(np.mean(abs(nll104 - probe104.nll.to_numpy())))))
print("external probe NLL:", probe104.nll.mean())
print("type, age scale, status, NLL, mean MAE, NLL MAE:")
for row104 in sorted(results104, key=lambda r: r[5])[:12]:
    print(row104)
with open("/work/notes.md", "a") as f104:
    f104.write("\n## Run 104: head convention experiment\n"
               "- All scratch tensors are original and registers match raw records at the largest-error rows.\n"
               "- Testing type/status indices and register age units against external factual predictions.\n"
               "- Python runs: 104; forward passes: 4; no submitted edits.\n")
```
