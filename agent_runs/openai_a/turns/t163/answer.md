The probe stopped because the vocabulary lists 16 value centres, while the head has 17 classes including the absent-value bin. That is a bookkeeping error, not evidence about drug effects. I’ll use the existing head-centre table where available and rerun the paired probe; no weights have changed.

```python
probe_centres_by_event = {}
existing_centres = np.asarray(as_numpy(head_centres))
for event0 in probe_meta["event"].unique():
    if existing_centres.shape == (len(head_meta_tune), 17):
        first0 = np.flatnonzero(head_meta_tune["event"].eq(event0).to_numpy())[0]
        probe_centres_by_event[event0] = existing_centres[first0].copy()
    else:
        values0 = np.asarray(VOCAB["value_bin_centres"][event0], dtype=np.float32)
        probe_centres_by_event[event0] = (
            np.r_[0.0, values0] if len(values0) == 16 else values0
        )
    print(event0, "first centres:", probe_centres_by_event[event0][:3])

def probe_head_eval(meta0, states0, head_weights=None, n_samples=4, seed=731):
    parts0 = cpu_head_parts(meta0, states0)
    hw = graft_original if head_weights is None else head_weights
    hw = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in hw.items()}
    prior0 = parts0["prior"]
    mu0, sd0 = prior0[:, :32], torch.exp(prior0[:, 32:].clamp(-10, 5))
    generator0 = torch.Generator().manual_seed(seed)
    features0, probabilities0 = [], []
    for _ in range(n_samples):
        z0 = mu0 + sd0 * torch.randn(mu0.shape, generator=generator0)
        q0 = ln(parts0["base"] + lin(z0, "z_head") +
                W["type_emb.weight"][1], "query_norm")
        feature0 = q0 + parts0["additive"]
        logits0 = mlp(feature0, "attr_heads.value_bin", weights=hw) + parts0["transition"]
        features0.append(feature0)
        probabilities0.append(torch.softmax(logits0, dim=-1))
    p0 = torch.stack(probabilities0).mean(0)
    centres0 = torch.tensor(
        np.stack([probe_centres_by_event[event0] for event0 in meta0["event"]]),
        dtype=torch.float32
    )
    bins0 = torch.tensor(meta0["observed_bin"].to_numpy(), dtype=torch.long)
    return {
        "features": torch.stack(features0).numpy(),
        "probs": p0.numpy(),
        "mean": (p0 * centres0).sum(-1).numpy(),
        "nll": (-torch.log(p0.gather(1, bins0[:, None]).squeeze(1).clamp_min(1e-20))).numpy()
    }

probe_results = {}
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
          "baseline NLL delta:", float((basee["nll"] - base0["nll"]).mean()))
    for multiplier in (0.5, 2.0):
        p0, pe, ps = probe_results[(anchored, multiplier)]
        print("multiplier", multiplier, {
            "original_glucose_shift": float((p0["mean"] - base0["mean"])[glucose_probe_mask].mean()),
            "edited_glucose_shift": float((pe["mean"] - basee["mean"])[glucose_probe_mask].mean()),
            "graft_signal_shift": float((ps - bases)[:, glucose_probe_mask].mean())
        })
```
