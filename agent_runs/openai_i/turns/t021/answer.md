The glucose error is concentrated in readings with a previous value, while the state layout is exact. That points toward register handling or head computation rather than corrupted states. I’ll test gap scaling, register ages, latent handling, and bin boundaries without editing weights. A candidate must improve both mean predictions and fine-bin NLL to be credible.

```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

print("numeric_note:", VOCAB.get("numeric_note"))
print("value_bin_note:", VOCAB.get("value_bin_note"))

def head2(rows, prior_gap=0.1, view_gap=0.1, age_unit="hours",
          latent="prior", z_samples=0, seed=917, bin_side=None):
    w = head_weights
    prev = rows.previous_state.to_numpy(dtype=int)
    sx = torch.from_numpy(state_x[prev].astype(np.float32))
    dt = torch.tensor(
        (rows.hour.to_numpy() - state_hours[prev]) * 3600,
        dtype=torch.float32
    ).clamp_min(1)
    h = sx[:, 192:960].reshape(-1, 4, 192)
    flowed = h * torch.exp(-dt[:, None, None] * w["core.flow.a"].exp())
    state_base = sx[:, :192] + norm_named(flowed.mean(1), "core.out_norm")
    ctx = state_base + mlp_named(dt.log()[:, None] * prior_gap, "gap")
    view_base = state_base + mlp_named(dt.log()[:, None] * view_gap, "gap")
    prior = mlp_named(ctx, "z_prior")
    mu, logsd = prior[:, :32], prior[:, 32:]

    cid = torch.tensor([code_ids[e] for e in rows.event], dtype=torch.long)
    bins = rows.last_bin.to_numpy(dtype=int).copy()
    if bin_side is not None:
        for event in MEAS:
            ix = np.flatnonzero((rows.event == event) & rows.last_value.notna())
            if len(ix):
                edges = numeric_specs[(event, "value")]["fine_bin_edges"]
                bins[ix] = 1 + np.searchsorted(
                    edges, rows.last_value.to_numpy()[ix], side=bin_side
                )
    lb = torch.tensor(bins, dtype=torch.long)
    has = torch.tensor(rows.last_value.notna().to_numpy(), dtype=torch.float32)
    age = rows.hour.to_numpy() - rows.last_hour.fillna(rows.hour).to_numpy()
    if age_unit == "seconds":
        age = age * 3600
    lf = torch.zeros(len(rows), 8)
    lf[:, 0] = has
    lf[:, 1] = torch.tensor(np.log1p(np.maximum(age, 0)) / 10, dtype=torch.float32)
    lf[:, 2] = torch.tensor(rows.last_z.to_numpy(), dtype=torch.float32)
    lf[:, 3] = has
    reg = (
        w["embed.code.weight"][cid] + linear_named(lf, "last_numeric")
        + w["embed.bins.value.weight"][lb]
        + w["embed.bins.rate.weight"][0] + w["last_status.weight"][0]
    )
    transition = w["embed_transition.value_bin.weight"][cid * 17 + lb]
    centres = torch.tensor(
        np.array([VOCAB["value_bin_centres"][e] for e in rows.event]),
        dtype=torch.float32
    )
    if centres.shape[1] == 16:
        centres = torch.cat([torch.zeros(len(rows), 1), centres], dim=1)
    gen = torch.Generator().manual_seed(seed)
    ps, hs = [], []
    for draw in range(max(1, z_samples)):
        z = mu
        if latent == "posterior_previous":
            z = sx[:, 960:]
        elif latent == "zero":
            z = torch.zeros_like(mu)
        elif z_samples:
            z = mu + logsd.clamp(-6, 2).exp() * torch.randn(mu.shape, generator=gen)
        view = view_base + linear_named(z, "z_head")
        q = norm_named(view + w["type_emb.weight"][1], "query_norm")
        hidden = F.gelu(linear_named(q + reg, "attr_heads.value_bin.0"))
        logits = linear_named(hidden, "attr_heads.value_bin.2") + transition
        ps.append(logits.softmax(-1))
        hs.append(hidden)
    p = torch.stack(ps).mean(0)
    mean = (p * centres).sum(1)
    return mean.numpy(), p.numpy(), torch.stack(hs).mean(0).numpy()

_probe = head_rows.sample(650, random_state=71).copy()
_scores = []
for _pg in (0.1, 1.0):
    for _vg in (0.1, 1.0):
        for _au in ("hours", "seconds"):
            for _lat in ("prior", "posterior_previous", "zero"):
                _m, _p, _ = head2(_probe, _pg, _vg, _au, _lat)
                _d = _m - _probe.predicted_mean.to_numpy()
                _scores.append(dict(
                    prior_gap=_pg, view_gap=_vg, age_unit=_au, latent=_lat,
                    rms=float(np.sqrt(np.mean(_d**2))),
                    mae=float(np.mean(np.abs(_d))), bias=float(np.mean(_d)),
                    p_absent=float(_p[:, 0].mean())
                ))
head2_grid = pd.DataFrame(_scores).sort_values("rms")
print("\n=== GAP / REGISTER / LATENT TEST ===")
print(head2_grid.head(12).round(4).to_string(index=False))

_best = head2_grid.iloc[0]
head2_settings = {k: _best[k] for k in ("prior_gap", "view_gap", "age_unit", "latent")}
print("\n=== MARGINALIZATION AND BIN BOUNDARY TEST ===")
for _side in (None, "left", "right"):
    for _ns in (0, 32):
        _m, _p, _ = head2(head_rows, **head2_settings, z_samples=_ns, bin_side=_side)
        _d = _m - head_rows.predicted_mean.to_numpy()
        _obsbin = np.array([
            int(fine_bin(e, "value", np.array([v]))[0])
            for e, v in zip(head_rows.event, head_rows.observed)
        ])
        _nll = -np.log(np.maximum(_p[np.arange(len(_p)), _obsbin], 1e-12))
        print("side", _side, "draws", _ns,
              "RMS", round(float(np.sqrt(np.mean(_d**2))), 4),
              "bias", round(float(_d.mean()), 4),
              "NLL", round(float(_nll.mean()), 5),
              "reference", round(float(head_rows.nll.mean()), 5))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Head computation tests — Python 16; forward 4\n"
        "- State layout verified: out equals normalized mean(h) to ~1e-7 RMS.\n"
        "- Reconstruction glucose bias is predominantly in rows with a last-value register.\n"
        "- Testing separate prior/view gap scaling, age units, latent modes and bin boundaries; no edits.\n"
    )
```
