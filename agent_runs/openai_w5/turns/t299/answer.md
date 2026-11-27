The CSV contains infusion starts and their end times, but no explicit end-event rows. Those end events are constructed by the model pipeline, so the records alone do not settle their encoding. I’ll test plausible end-event embeddings and compare finite dose changes through the current action encoder. Approximately linear responses across those cases would rule out the encoder as the main source of doubling–halving asymmetry.

```python
import numpy as np
import torch
import pandas as pd

W262 = {n: torch.as_tensor(load(n), dtype=torch.float32).clone() for n in NAMES}
_pairs262 = {"insulin": (21, 61), "norepinephrine": (15, 66)}

def end_atom262(drug, ekind=2, keep_units=False):
    w = W262
    c = code_id[drug]
    x = (w["embed.code.weight"][c] + w["embed.kind.weight"][2]
         + w["embed.ekind.weight"][ekind]
         + w["embed.kq.weight"][0] + w["embed.text.weight"][0]).clone()
    for field in ("action", "discharged_to", "route", "status", "unit", "rate_unit"):
        name = f"embed.categorical.{field}.weight"
        if name in w:
            idx = 0
            if keep_units and field == "unit":
                idx = 3 if drug == "insulin" else 4
            if keep_units and field == "rate_unit":
                idx = 1 if drug == "insulin" else 6
            x += w[name][idx]
    for field in ("value", "dose", "rate"):
        x += w[f"embed.bins.{field}.weight"][0]
    x += w["embed.numeric.bias"]
    return x

print("=== CURRENT ACTION-ENCODER FINITE RESPONSE ===")
for drug, rates in (
    ("insulin", np.array([0., .5, 1., 2., 4., 8., 12.], np.float32)),
    ("norepinephrine", np.array([0., .025, .05, .1, .2, .4, .8], np.float32)),
):
    atoms = drug_atom205(drug, rates, weights=W262)
    i, j = _pairs262[drug]
    rows = {"rate": rates}
    for label, end in [
        ("WORLD", None),
        ("plus_END_ek2", end_atom262(drug, 2, False)),
        ("plus_END_ek0", end_atom262(drug, 0, False)),
        ("plus_END_units", end_atom262(drug, 2, True)),
    ]:
        bundles = atoms[:, None, :] if end is None else torch.stack(
            (atoms, end.expand(len(rates), -1)), dim=1)
        u = set205(bundles, "encode_action", weights=W262)
        rows[label] = (u[:, i] - u[:, j]).detach().numpy()
    print(drug)
    print(pd.DataFrame(rows).round(7).to_string(index=False))

print("\n=== EARLIER ROLLOUT FILE SUMMARIES ===")
for k in (14, 17, 18, 20, 21, 23, 24):
    path = Path(f"/work/fwd_{k}/dose_edit_edited.csv")
    if path.exists():
        df = pd.read_csv(path)
        print(k, "shape", df.shape, "columns", df.columns.tolist())
        if {"edit", "shift"}.issubset(df.columns):
            if "hour" in df.columns:
                df = df[(df.hour >= 13) & (df.hour <= 24)]
            print(df.groupby("edit")["shift"].mean().round(4).to_dict())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 262: finite encoder audit\n"
        "- Tested finite rate transport through current action encoder, with plausible synthetic END variants.\n"
        "- END variants are diagnostic assumptions, not validated pipeline semantics.\n"
        "- No model weights changed; a persistence trial must account for large recurrent input gains.\n"
    )
```