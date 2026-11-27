The reconstruction uses the plausible interleaved register, but it substitutes the prior mean for the GPU’s four latent draws. That could explain part of the mismatch. I’ll test query types and latent averaging now. If averaging closes the NLL gap, we can use the CPU head for statistical edits while validating them on the GPU.

```python
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

def as_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

head_reference = head_meta_tune[["stay_id", "hour", "event"]].merge(
    factual_baseline, on=["stay_id", "hour", "event"],
    how="left", sort=False, validate="one_to_one"
)
head_gpu_mean = head_reference.predicted_mean.to_numpy()
head_gpu_nll = head_reference.nll.to_numpy()
head_centres = torch.tensor(np.stack([
    VOCAB["value_bin_centres"][e] for e in head_meta_tune.event
]), dtype=torch.float32)
head_obsbins = torch.tensor(
    head_meta_tune.observed_bin.to_numpy(), dtype=torch.long
)

def head_score(mean, nll):
    mean, nll = as_numpy(mean), as_numpy(nll)
    result = {
        "rmse": float(np.sqrt(np.mean((mean-head_gpu_mean)**2))),
        "nll": float(np.mean(nll)),
        "bias": float(np.mean(mean-head_gpu_mean))
    }
    for event in ["glucose_finger", "map_arterial"]:
        m = head_meta_tune.event.to_numpy() == event
        result[event] = [
            float(np.sqrt(np.mean((mean[m]-head_gpu_mean[m])**2))),
            float(np.mean(mean[m]-head_gpu_mean[m])),
            float(np.mean(nll[m]))
        ]
    return result

head_setting_scores = []
for type_id in range(15):
    res = make_head_features(
        head_meta_tune, state_files[3],
        type_id=type_id, gap_style="seconds_div10",
        register_style="interleaved", status=0
    )
    head_setting_scores.append({
        "type_id": type_id, "gap": "seconds_div10",
        "age": "seconds", "view": "mean_prior",
        **head_score(res[2], res[3])
    })
for gap_style, age_style, view_style in [
    ("hours", "seconds", "mean_prior"),
    ("seconds_div10", "hours", "mean_prior"),
    ("seconds_div10", "seconds", "posterior"),
    ("hours", "seconds", "posterior"),
]:
    res = make_head_features(
        head_meta_tune, state_files[3],
        gap_style=gap_style, age_style=age_style,
        view_style=view_style, register_style="interleaved", status=0
    )
    head_setting_scores.append({
        "type_id": 1, "gap": gap_style,
        "age": age_style, "view": view_style,
        **head_score(res[2], res[3])
    })
print("=== BEST SETTINGS ===")
for r in sorted(head_setting_scores, key=lambda r: r["rmse"])[:8]:
    print(r)
print("GPU NLL:", float(np.mean(head_gpu_nll)))

def cpu_head_parts(meta, states):
    x = torch.from_numpy(states["x"][meta.state_row.to_numpy(int)].copy())
    dt = torch.tensor(meta.dt_seconds.to_numpy(), dtype=torch.float32)
    h = x[:, 192:960].reshape(-1, 4, 192)
    hflow = h * torch.exp(
        -dt[:, None, None] * torch.exp(W["core.flow.a"])[None, None, :]
    )
    base = (x[:, :192] + ln(hflow.mean(1), "core.out_norm")
            + mlp((torch.log1p(dt)/10)[:, None], "gap"))
    prior = mlp(base, "z_prior")
    reg = torch.zeros(len(meta), 8)
    reg[:, 0] = torch.tensor(meta.has_last.to_numpy(), dtype=torch.float32)
    reg[:, 1] = torch.log1p(torch.tensor(
        meta.last_age_seconds.to_numpy(), dtype=torch.float32
    )) / 10
    reg[:, 2] = torch.tensor(meta.last_z.to_numpy(), dtype=torch.float32)
    reg[:, 3] = reg[:, 0]
    codes = torch.tensor(meta.code.to_numpy(), dtype=torch.long)
    bins = torch.tensor(meta.last_bin.to_numpy(), dtype=torch.long)
    additive = (
        W["embed.code.weight"][codes] + lin(reg, "last_numeric")
        + W["embed.bins.value.weight"][bins]
        + W["embed.bins.rate.weight"][0]
        + W["last_status.weight"][0]
    )
    transition = W["embed_transition.value_bin.weight"][codes*17+bins]
    return {"base": base, "prior": prior, "additive": additive,
            "transition": transition}

head_parts_tune = cpu_head_parts(head_meta_tune, state_files[3])

def cpu_head_probs(parts, n_samples=0, seed=731):
    gen = torch.Generator().manual_seed(seed)
    prior = parts["prior"]
    mu, sd = prior[:, :32], torch.exp(prior[:, 32:].clamp(-10, 5))
    total = torch.zeros(len(mu), 17)
    count = max(n_samples, 1)
    for j in range(count):
        z = mu if n_samples == 0 else mu + sd * torch.randn(
            mu.shape, generator=gen
        )
        view = parts["base"] + lin(z, "z_head")
        q = ln(view + W["type_emb.weight"][1], "query_norm")
        logits = (mlp(q + parts["additive"], "attr_heads.value_bin")
                  + parts["transition"])
        total += torch.softmax(logits, dim=-1)
    probs = total / count
    mean = (probs * head_centres).sum(-1)
    nll = -torch.log(probs.gather(1, head_obsbins[:, None]).squeeze(1).clamp_min(1e-20))
    return probs, mean, nll

print("\n=== MANUAL HEAD AND LATENT AVERAGING ===")
for n_samples in [0, 4, 32]:
    probs, mean, nll = cpu_head_probs(head_parts_tune, n_samples)
    print("samples", n_samples, head_score(mean, nll))
print("prior log-sd quantiles:",
      np.quantile(as_numpy(head_parts_tune["prior"][:, 32:]),
                  [0, .1, .5, .9, 1]))
print("Current edits:", edited())
```