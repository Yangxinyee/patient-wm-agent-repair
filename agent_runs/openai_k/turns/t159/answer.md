Every reading aligns with a preceding state, and the paired edit changes glucose predictions much more than MAP predictions. I’ll fit small, smooth readout surrogates from the official states. They must predict both held-out factual means and the measured edit-induced changes; otherwise their sensitivities are not suitable for designing an edit.

```python
import numpy as np, torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import SplineTransformer

torch.set_num_threads(8)
W144 = {n: torch.as_tensor(load_original(n), dtype=torch.float32)
        for n in NAMES if n.startswith(
            ("core.flow.", "core.out_norm.", "gap.", "z_prior.",
             "z_head.", "query_norm.", "type_emb."))}

def linear144(x, p):
    return F.linear(x, W144[p + ".weight"], W144.get(p + ".bias"))

def norm144(x, p):
    return F.layer_norm(x, (x.shape[-1],),
                        W144[p + ".weight"], W144[p + ".bias"], 1e-5)

def mlp144(x, p):
    return linear144(F.gelu(linear144(x, p + ".0")), p + ".2")

def views144(state, dt):
    t = torch.as_tensor(state[:, :960], dtype=torch.float32)
    out = t[:, :192]
    h = t[:, 192:960].reshape(-1, 4, 192)
    dt = torch.as_tensor(dt, dtype=torch.float32).reshape(-1, 1, 1)
    hflow = h * torch.exp(-dt * torch.exp(W144["core.flow.a"]).reshape(1, 1, -1))
    context = out + norm144(hflow.mean(1), "core.out_norm")
    context = context + mlp144(torch.log(dt[:, 0, :].clamp_min(1)) / 10, "gap")
    prior = mlp144(context, "z_prior")[:, :32]
    view = context + linear144(prior, "z_head")
    q = norm144(view + W144["type_emb.weight"][1], "query_norm")
    return {"out": out.numpy(), "context": context.numpy(), "query": q.numpy()}

V144o = views144(S143o["x"][ix143], F143.dt_seconds.to_numpy())
V144e = views144(S143e["x"][ix143], F143.dt_seconds.to_numpy())
models144, diagnostics144 = {}, []
train144 = (F143.stay_id.to_numpy() % 4) != 0

for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    mask = F143.event.to_numpy() == event
    ids = np.flatnonzero(mask)
    tr, va = train144[ids], ~train144[ids]
    lv = F143.last_value.to_numpy()[ids]
    missing = ~np.isfinite(lv)
    median = np.nanmedian(lv)
    lv = np.where(missing, median, lv)
    spline = SplineTransformer(n_knots=7, degree=3, include_bias=False)
    spl = spline.fit_transform(lv.reshape(-1, 1))
    age = np.log1p(np.maximum(
        F143.hour.to_numpy()[ids] - F143.last_hour.fillna(F143.hour).to_numpy()[ids], 0
    ))
    lastz = (lv - median) / max(np.std(lv), 1)
    base = np.c_[spl, missing, age, age**2]
    y = F143.predicted_mean_o.to_numpy()[ids]
    dy = F143.mean_delta.to_numpy()[ids]
    best = None
    for variant in ["out", "context", "query"]:
        for interaction in [False, True]:
            so, se = V144o[variant][ids], V144e[variant][ids]
            xo = np.c_[base, so]
            xe = np.c_[base, se]
            if interaction:
                xo = np.c_[xo, so * lastz[:, None]]
                xe = np.c_[xe, se * lastz[:, None]]
            scaler = StandardScaler().fit(xo[tr])
            zo, ze = scaler.transform(xo), scaler.transform(xe)
            for alpha in [1, 10, 100, 1000]:
                model = Ridge(alpha=alpha).fit(zo[tr], y[tr])
                pred = model.predict(zo[va])
                dp = model.predict(ze[va]) - pred
                mae = mean_absolute_error(y[va], pred)
                dmae = mean_absolute_error(dy[va], dp)
                score = mae + 2 * dmae
                row = dict(event=event, variant=variant, interaction=interaction,
                           alpha=alpha, mae=mae, delta_mae=dmae,
                           delta_r2=r2_score(dy[va], dp), score=score)
                diagnostics144.append(row)
                if best is None or score < best[0]:
                    best = (score, row, scaler, model, spline, median,
                            max(np.std(lv), 1), xo, xe, ids, base)
    models144[event] = best
    print(best[1])

print("\nBest alternatives:")
print(pd.DataFrame(diagnostics144).sort_values("score")
      .groupby("event").head(3).to_string(index=False))
print("\nVocabulary keys:", list(VOCAB))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 144: authoritative readout-surrogate validation\n"
            "- Fit stay-held-out smooth ridge readouts, with last-value splines and state features.\n"
            "- Validation includes original factual means and current-edit paired mean changes.\n"
            "- Results retained as models144/diagnostics144; no weight changes.\n"
            "- Python runs: 144; forward passes: 10.\n")
```