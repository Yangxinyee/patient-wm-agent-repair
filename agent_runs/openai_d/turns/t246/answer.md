The rebuilt training cache gives similar losses to the earlier cache, and the two tuning evaluators agree closely. I’ll test a small, closed-form calibration of each reading distribution and inspect gate activation. If calibration improves the independent tuning loss, it may create room for the stronger causal correction; a training-only gain would refute that approach.

```python
def c211_logits(weights, cache):
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
    return logits

def c211_moment_calibration(weights, ridge=0.02):
    p = c211_logits(weights, c210_train).softmax(-1).mean(0).numpy()
    y = c209_y.numpy()
    result, coefficients = {}, {}
    events = head_meta.event.to_numpy()
    for e in np.unique(events):
        mask = events == e
        spec = r201_numeric[(e, "value")]
        z = np.clip((np.r_[spec["median"], VOCAB["value_bin_centres"][e]]
                     - spec["median"]) / spec["iqr"], -3, 3)
        basis = np.column_stack([z, z*z])
        expected = p[mask] @ basis
        score = (basis[y[mask]]-expected).mean(0)
        fisher = (np.einsum("nb,bi,bj->nij", p[mask], basis, basis)
                  - expected[:,:,None]*expected[:,None,:]).mean(0)
        beta = np.linalg.solve(fisher + ridge*np.eye(2), score)
        result[e] = (basis @ beta).astype(np.float32)
        coefficients[e] = beta
    return result, coefficients

def c211_cal_eval(weights, cache, calibration, scale):
    shifts = torch.tensor(
        np.array([calibration[e] for e in cache["meta"].event]),
        dtype=torch.float32)
    p = (c211_logits(weights, cache) + scale*shifts[None,:,:]).softmax(-1).mean(0)
    loss = -torch.log(p[torch.arange(len(cache["y"])), cache["y"]].clamp_min(1e-12)).numpy()
    mean = (p*cache["centres"]).sum(-1).numpy()
    return loss, mean

c211_calibrations, c211_screen = {}, []
for label in ["original", "combined_strong"]:
    weights = c206_candidates[label]
    calibration, coefficients = c211_moment_calibration(weights)
    c211_calibrations[label] = calibration
    print("calibration coefficients", label,
          {e: np.round(v, 5).tolist() for e, v in coefficients.items()})
    for scale in [0.0, 0.25, 0.5, 0.75, 1.0]:
        row = {"candidate": label, "calibration_scale": scale}
        for split, cache in [("train", c210_train), ("tune100", c210_tune)]:
            loss, _ = c211_cal_eval(weights, cache, calibration, scale)
            row[split] = float((loss-cache["loss0"]).mean())
        c211_screen.append(row)
print("=== MOMENT-CALIBRATION SCREEN ===")
print(pd.DataFrame(c211_screen).round(6).to_string(index=False))

print("=== COMMON GATE QUANTILES ===")
for split, cache in [("train", c210_train), ("tune100", c210_tune)]:
    for label in ["insulin_only", "combined_strong"]:
        w = c206_candidates[label]
        pair = [107,109] if label == "insulin_only" else [48,171]
        g = torch.nn.functional.linear(
            cache["X"],
            w["attr_heads.value_bin.0.weight"][pair].mean(0)[None,:],
            w["attr_heads.value_bin.0.bias"][pair].mean()[None]).squeeze(-1).numpy()
        for e in np.unique(cache["meta"].event):
            m = cache["meta"].event.to_numpy() == e
            print(split, label, e,
                  "q10/50/90", np.round(np.quantile(g[:,m],[.1,.5,.9]),2),
                  "positive fraction", round(float((g[:,m]>0).mean()),3))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 211: closed-form calibration screen\n"
            "Fitted two-parameter per-reading-code probability-tilt calibrators from the fresh "
            "training cache using empirical score moments and a ridge-regularized 2x2 solve; "
            "no model gradients or weight optimizer. Screened independent tuning losses and "
            "common-gate support. No weights changed.\n"
            + pd.DataFrame(c211_screen).to_string(index=False) + "\n")
```
