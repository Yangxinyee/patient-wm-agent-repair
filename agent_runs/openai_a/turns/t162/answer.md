The dose contrasts are only modestly larger than the reconstruction error, and doubling and halving do not point in opposite directions. That suggests nonlinear encoding, so simple extrapolation is unsafe. I’ll anchor each contrast to its exact cached state and check its effect on glucose predictions and on the existing graft’s decoded signal.

```python
import torch.nn.functional as F

probe_meta = head_meta_train[
    head_meta_train["state_row"].isin(rm_pure_idx)
].copy().reset_index(drop=True)
assert np.all(
    probe_meta["stay_id"].to_numpy()
    == np.asarray(rm_states["stay_id"])[probe_meta["state_row"].to_numpy()]
)

def probe_head_eval(meta0, states0, head_weights=None, n_samples=4, seed=731):
    parts0 = cpu_head_parts(meta0, states0)
    hw = graft_original if head_weights is None else head_weights
    hw = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in hw.items()}
    prior0 = parts0["prior"]
    mu0 = prior0[:, :32]
    sd0 = torch.exp(prior0[:, 32:].clamp(-10, 5))
    generator0 = torch.Generator().manual_seed(seed)
    features0, probabilities0 = [], []
    for _ in range(n_samples):
        z0 = mu0 + sd0 * torch.randn(mu0.shape, generator=generator0)
        view0 = parts0["base"] + lin(z0, "z_head")
        q0 = ln(view0 + W["type_emb.weight"][1], "query_norm")
        feature0 = q0 + parts0["additive"]
        logits0 = mlp(feature0, "attr_heads.value_bin", weights=hw) + parts0["transition"]
        features0.append(feature0)
        probabilities0.append(torch.softmax(logits0, dim=-1))
    p0 = torch.stack(probabilities0).mean(0)
    centres0 = torch.tensor(
        np.stack([VOCAB["value_bin_centres"][event] for event in meta0["event"]]),
        dtype=torch.float32
    )
    mean0 = (p0 * centres0).sum(-1)
    bins0 = torch.tensor(meta0["observed_bin"].to_numpy(), dtype=torch.long)
    nll0 = -torch.log(p0.gather(1, bins0[:, None]).squeeze(1).clamp_min(1e-20))
    return {
        "features": torch.stack(features0).numpy(),
        "probs": p0.numpy(), "mean": mean0.numpy(), "nll": nll0.numpy()
    }

probe_current_head = {name: load(name) for name in graft_original}
probe_original_x = np.array(rm_states["x"], copy=True)
probe_results = {}
glucose_probe_mask = probe_meta["event"].eq("glucose_finger").to_numpy()

w0e = torch.as_tensor(load("attr_heads.value_bin.0.weight"))
b0e = torch.as_tensor(load("attr_heads.value_bin.0.bias"))
w0o = torch.as_tensor(load_original("attr_heads.value_bin.0.weight"))
probe_changed_nodes = torch.where((w0e - w0o).abs().amax(1) > 1e-7)[0]
print("changed hidden nodes:", probe_changed_nodes.tolist())
print("matched glucose readings:", int(glucose_probe_mask.sum()))

for anchored in (True, False):
    for multiplier in (1.0, 0.5, 2.0):
        state0 = dict(rm_states)
        state0["x"] = probe_original_x.copy()
        synthetic0 = dose_probe_states[multiplier]
        if anchored:
            synthetic0 = probe_truth + synthetic0 - dose_probe_states[1.0]
        state0["x"][rm_pure_idx] = synthetic0
        original0 = probe_head_eval(probe_meta, state0)
        edited0 = probe_head_eval(probe_meta, state0, probe_current_head)
        pre0 = F.linear(torch.tensor(original0["features"]), w0e, b0e).numpy()
        signal0 = pre0[:, :, probe_changed_nodes[0]] - pre0[:, :, probe_changed_nodes[1]]
        probe_results[(anchored, multiplier)] = (original0, edited0, signal0)

    base0, basee, bases = probe_results[(anchored, 1.0)]
    print("\nanchored:", anchored,
          "baseline edited-minus-original NLL:", float((basee["nll"] - base0["nll"]).mean()))
    for multiplier in (0.5, 2.0):
        p0, pe, ps = probe_results[(anchored, multiplier)]
        print("dose multiplier", multiplier, {
            "original_glucose_shift": float((p0["mean"] - base0["mean"])[glucose_probe_mask].mean()),
            "edited_glucose_shift": float((pe["mean"] - basee["mean"])[glucose_probe_mask].mean()),
            "graft_signal_shift": float((ps - bases)[:, glucose_probe_mask].mean())
        })
```
