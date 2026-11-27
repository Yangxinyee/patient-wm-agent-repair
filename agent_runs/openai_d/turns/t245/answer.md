The rebuild stopped because the register helper returns three components, not one tensor. No weights changed. I’ll unpack those components and complete the GPU comparison. A small prediction discrepancy and consistent candidate loss differences would validate the rebuilt training cache.

```python
print("register output:", type(c209_fixed).__name__,
      [(type(v).__name__, getattr(v, "shape", None)) for v in c209_fixed])
c209_fixed, c209_codes, c209_lb = c209_fixed
c209_X = r201_queries(c209_ctx, c209_eps) + c209_fixed[None, :, :]
c209_tname = next(n for n in r201_W
                  if n.startswith("embed_transition.value_bin") and n.endswith("weight"))
c209_trans = r201_W[c209_tname][c209_codes * 17 + c209_lb]
c209_centres = torch.tensor(
    np.array([[0.0] + list(VOCAB["value_bin_centres"][e]) for e in head_meta.event]),
    dtype=torch.float32)
print("numeric specification:", r201_numeric[("glucose_finger", "value")])
c210_edges = {}
for e in head_meta.event.unique():
    spec = r201_numeric[(e, "value")]
    edge_key = next(k for k in spec if "edge" in k.lower())
    c210_edges[e] = np.asarray(spec[edge_key])
c209_y = torch.tensor([
    int(np.searchsorted(c210_edges[e], value, side="left")) + 1
    for e, value in zip(head_meta.event, head_meta.observed)
], dtype=torch.long)
c209_original = c206_candidates["original"]
c209_H0 = torch.nn.functional.gelu(torch.nn.functional.linear(
    c209_X, c209_original["attr_heads.value_bin.0.weight"],
    c209_original["attr_heads.value_bin.0.bias"]))
c209_L0 = torch.nn.functional.linear(
    c209_H0, c209_original["attr_heads.value_bin.2.weight"],
    c209_original["attr_heads.value_bin.2.bias"]) + c209_trans[None, :, :]
c209_P0 = c209_L0.softmax(-1).mean(0)
c209_mean0 = (c209_P0 * c209_centres).sum(-1).numpy()
c209_loss0 = -torch.log(c209_P0[
    torch.arange(len(head_meta)), c209_y].clamp_min(1e-12)).numpy()

print("=== TRAINING RECONSTRUCTION VALIDATION ===")
print("shape:", tuple(c209_X.shape))
print("original mean RMSE against GPU:",
      float(np.sqrt(np.mean((c209_mean0-head_meta.tool_mean.to_numpy())**2))))
print("original NLL local/GPU:",
      float(c209_loss0.mean()), float(head_meta.tool_nll.mean()))
print(pd.DataFrame({
    "event": head_meta.event, "mean_error": c209_mean0-head_meta.tool_mean,
    "local_nll": c209_loss0, "gpu_nll": head_meta.tool_nll
}).groupby("event").mean().round(5))

c210_Ht0 = torch.nn.functional.gelu(torch.nn.functional.linear(
    r201_X, c209_original["attr_heads.value_bin.0.weight"],
    c209_original["attr_heads.value_bin.0.bias"]))
c210_Lt0 = torch.nn.functional.linear(
    c210_Ht0, c209_original["attr_heads.value_bin.2.weight"],
    c209_original["attr_heads.value_bin.2.bias"]) + r201_trans[None, :, :]
c210_train = dict(X=c209_X, H0=c209_H0, L0=c209_L0, y=c209_y,
                  centres=c209_centres, meta=head_meta, loss0=c209_loss0)
c210_tune = dict(X=r201_X, H0=c210_Ht0, L0=c210_Lt0, y=r201_y,
                 centres=r201_centres, meta=r201_meta, loss0=r201_loss0)

def c210_eval(weights, cache, return_probs=False):
    w = c207_tensor_weights(weights)
    o = c209_original
    idx = torch.where(
        ((w["attr_heads.value_bin.0.weight"]-o["attr_heads.value_bin.0.weight"]).abs().amax(1) > 1e-7)
        | ((w["attr_heads.value_bin.0.bias"]-o["attr_heads.value_bin.0.bias"]).abs() > 1e-7)
    )[0]
    logits = cache["L0"] + torch.nn.functional.linear(
        cache["H0"],
        w["attr_heads.value_bin.2.weight"]-o["attr_heads.value_bin.2.weight"],
        w["attr_heads.value_bin.2.bias"]-o["attr_heads.value_bin.2.bias"])
    if len(idx):
        hnew = torch.nn.functional.gelu(torch.nn.functional.linear(
            cache["X"], w["attr_heads.value_bin.0.weight"][idx],
            w["attr_heads.value_bin.0.bias"][idx]))
        logits += torch.nn.functional.linear(
            hnew-cache["H0"][:,:,idx], w["attr_heads.value_bin.2.weight"][:,idx])
    p = logits.softmax(-1).mean(0)
    loss = -torch.log(p[torch.arange(len(cache["y"])), cache["y"]].clamp_min(1e-12)).numpy()
    mean = (p*cache["centres"]).sum(-1).numpy()
    return (loss, mean, p.numpy()) if return_probs else (loss, mean)

print("=== TRAINING CANDIDATE DELTAS ===")
for label, weights in c206_candidates.items():
    loss, mean = c210_eval(weights, c210_train)
    print(label, float((loss-c209_loss0).mean()))
print("tuning evaluator cross-check:",
      float(np.max(np.abs(c210_eval(c183_fallback, c210_tune)[0]-r201_lossc))))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 210: generic independent cache evaluator\n"
            "Completed fresh training reconstruction and built a generic exact four-head-tensor "
            "evaluator for training and tuning 100–199. It includes register transition logits "
            "and supports sparse hidden-row changes. No weights changed.\n")
```
