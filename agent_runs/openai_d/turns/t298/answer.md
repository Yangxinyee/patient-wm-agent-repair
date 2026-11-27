Previous-bin cancellation lowered cached tuning loss by up to about 0.0009 nats, but held-out training gains were small. The recovered record-based targets are roughly −14.2/+7.1 for insulin and +4.7/−2.3 for norepinephrine, with long-horizon uncertainty. I’ll test a small training-only calibration for additional factual margin; it must help held-out data, not just its fitting sample.

```python
import numpy as np, pandas as pd, torch, re
from pathlib import Path

print("=== PREVIOUS CALIBRATION NOTES ===")
secs = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
for s in secs:
    if re.search(r"Run(?:s)? (211|224)\b", s.splitlines()[0]):
        print(s[:900])

r253_base_table = {
    k: -1.25 * v for k, v in r252_tables[20.].items()
}
r253_base_rows = {
    label: r252_table_rows(r253_base_table, c)
    for label, c in [("train", c210_train), ("tune", c210_tune)]
}
r253_lt = r252_logits["train"][1] + torch.tensor(
    r253_base_rows["train"], dtype=torch.float32)[None]
r253_ps = r253_lt.softmax(-1)

r253_event_vectors = {}
for event in c210_train["meta"].event.unique():
    i = np.flatnonzero(
        c210_train["meta"].event.to_numpy() == event)[0]
    centres = np.asarray(c210_train["centres"][i], dtype=np.float64)
    r253_event_vectors[event] = (
        centres - centres[1:].mean()
    ) / centres[1:].std()

def r253_value_vectors(cache):
    return np.stack([
        r253_event_vectors[e] for e in cache["meta"].event
    ])

def r253_features(event, lastbin, dim):
    z = 0. if int(lastbin) == 0 else float(
        r253_event_vectors[event][int(lastbin)])
    return np.asarray([1., z, z*z][:dim])

r253_vt = torch.tensor(r253_value_vectors(c210_train),
                      dtype=torch.float32)
r253_mu = (r253_ps * r253_vt[None]).sum(-1)
r253_var = (
    r253_ps * r253_vt[None].square()
).sum(-1) - r253_mu.square()
r253_y = c210_train["y"].long()
r253_py = r253_ps.gather(
    2, r253_y[None, :, None].expand(r253_ps.shape[0], -1, 1)
).squeeze(-1)
r253_post_mu = (r253_py * r253_mu).sum(0) / r253_py.sum(0)
r253_score = (
    r253_vt[torch.arange(len(r253_y)), r253_y] -
    r253_post_mu
).numpy()
r253_curvature = (
    (r253_py * r253_var).sum(0) / r253_py.sum(0) -
    ((r253_py * r253_mu.square()).sum(0) / r253_py.sum(0) -
     r253_post_mu.square())
).clamp_min(1e-3).numpy()

r253_models, r253_tables, r253_results = {}, {}, []
events_train = c210_train["meta"].event.to_numpy()
bins_train = c210_train["meta"].last_bin.to_numpy()

for dim in [1, 2, 3]:
    for ridge in [1., 10., 50.]:
        coefficients = {}
        for event in np.unique(events_train):
            mask = r252_fit & (events_train == event)
            F = np.stack([
                r253_features(event, b, dim)
                for b in bins_train[mask]
            ])
            A = F.T @ (r253_curvature[mask, None] * F)
            rhs = F.T @ r253_score[mask]
            coefficients[event] = np.linalg.solve(
                A + ridge*np.eye(dim), rhs)
        r253_models[(dim, ridge)] = coefficients
        for scale in [.5, 1.]:
            table = {}
            for (event, b), base in r253_base_table.items():
                alpha = scale * (
                    r253_features(event, b, dim) @ coefficients[event])
                table[(event, b)] = (
                    base + alpha*r253_event_vectors[event])
            key = (dim, ridge, scale)
            r253_tables[key] = table
            losses = {}
            for label, c in [("train", c210_train), ("tune", c210_tune)]:
                correction = r252_table_rows(table, c)
                losses[label] = r252_nll(
                    r252_logits[label][1], c, correction
                ) - r252_base[label]
            r253_results.append({
                "dim": dim, "ridge": ridge, "scale": scale,
                "train_fit": losses["train"][r252_fit].mean(),
                "train_validation": losses["train"][r252_val].mean(),
                "tune100": losses["tune"].mean()
            })

r253_screen = pd.DataFrame(r253_results)
print("\n=== CLOSED-FORM TRAINING-ONLY CALIBRATION SCREEN ===")
print(r253_screen.sort_values("train_validation").to_string(index=False))
print("\n=== PREVIOUS-BIN CANCELLATION WITHOUT OUTCOME CALIBRATION ===")
print(r252_screen.query("shrink == 20 and scale == 1.25").to_string(index=False))
print("No tensors changed.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 253: small statistical calibration screen\n"
        "On top of previous-bin cancellation (shrink20, scale1.25), fitted "
        "one closed-form ridge calibration per event using 60 training stays. "
        "Features are constant/last-bin-linear/quadratic; targets are local "
        "value-bin score residuals. No backpropagation or weight optimiser. "
        "Held-out training and tuning results below; no tensors changed.\n" +
        r253_screen.to_string(index=False) + "\n"
    )
```
