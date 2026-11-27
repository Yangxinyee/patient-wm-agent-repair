The outcome calibration improved held-out training loss but did not reliably improve tuning loss, so I won’t install it. I’ll inspect the norepinephrine neurons and test stronger previous-bin cancellation. Narrow factual activations would support an off-policy threshold repair; broad activations or worsening held-out loss would argue against it.

```python
import numpy as np, pandas as pd, torch

r254_rows = list(np.asarray(r240_ne_rows, dtype=int).reshape(-1))
print("=== NE PAIR GEOMETRY ===")
print("NE rows:", r254_rows, "insulin rows:", r246_ins_rows)
W0 = r252_head["attr_heads.value_bin.0.weight"]
b0 = r252_head["attr_heads.value_bin.0.bias"]
W2 = r252_head["attr_heads.value_bin.2.weight"]
P = (
    c210_train["X"] @ torch.tensor(W0[r254_rows].T)
    + torch.tensor(b0[r254_rows])
).mean(0).numpy()
r254_gate = P[:, 1]
r254_raw = P[:, 0] - P[:, 1]
r254_geometry = []
for event in c210_train["meta"].event.unique():
    mask = c210_train["meta"].event.to_numpy() == event
    g, r = r254_gate[mask], r254_raw[mask]
    r254_geometry.append({
        "event": event, "n": int(mask.sum()),
        "gate_q05": np.quantile(g, .05),
        "gate_median": np.median(g),
        "gate_q95": np.quantile(g, .95),
        "raw_mean": r.mean(), "raw_sd": r.std(),
        "raw_q05": np.quantile(r, .05),
        "raw_q95": np.quantile(r, .95)
    })
print(pd.DataFrame(r254_geometry).to_string(index=False))
print("Input-row difference norm:",
      np.linalg.norm(W0[r254_rows[0]] - W0[r254_rows[1]]))
print("Output-column norms:",
      [float(np.linalg.norm(W2[:, i])) for i in r254_rows])
print("Opposite-column residual:",
      np.linalg.norm(W2[:, r254_rows[0]] + W2[:, r254_rows[1]]))
for name in ["r240_q", "r240_probe", "r240_Xd", "r240_D",
             "r240_dmeans", "r240_eps", "r246_reg"]:
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, {str(k): str(getattr(v, "shape", type(v).__name__))
                         for k, v in obj.items()})
        elif hasattr(obj, "shape"):
            print(name, obj.shape)
        elif isinstance(obj, (int, float, str)):
            print(name, obj)
        else:
            print(name, type(obj).__name__)

print("\n=== EXTENDED PREVIOUS-BIN CANCELLATION SCREEN ===")
r254_results = []
for shrink in [5., 20., 80.]:
    table = r252_tables[shrink]
    corrections = {
        label: r252_table_rows(table, c)
        for label, c in [("train", c210_train), ("tune", c210_tune)]
    }
    for scale in [1., 1.5, 2., 3., 4.]:
        losses = {
            label: r252_nll(
                r252_logits[label][1], c, -scale*corrections[label]
            ) - r252_base[label]
            for label, c in [("train", c210_train), ("tune", c210_tune)]
        }
        r254_results.append({
            "shrink": shrink, "scale": scale,
            "train_fit": losses["train"][r252_fit].mean(),
            "train_validation": losses["train"][r252_val].mean(),
            "tune100": losses["tune"].mean()
        })
r254_screen = pd.DataFrame(r254_results)
print(r254_screen.to_string(index=False))
print("No tensors changed.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 254: NE activation geometry and extended cancellation\n"
        "Outcome calibration did not provide reliable tuning gains and was "
        "not installed. Inspected NE gate/raw-dose preactivations and extended "
        "training-only last-bin cancellation scales. No weights changed.\n" +
        pd.DataFrame(r254_geometry).to_string(index=False) + "\n" +
        r254_screen.to_string(index=False) + "\n"
    )
```
