The caches cover two separate training batches and one tuning batch, and the insulin dynamics estimate is −1.93 with persistence about 0.832. I’ll screen small intercept adjustments without changing weights. A correction that lowers cached factual cost across all three batches while preserving dose contrasts is worth GPU testing; cached improvement alone will not establish success.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("RIDGE KEYS:", list(r3_candidates.keys()))
repair224_fit = r3_candidates[(0.3, 1.0)]
print("CURRENT CONTRAST:",
      {k: v for k, v in repair224_fit.items()
       if k in ("b", "scale", "rmse")})

repair224_fact = {
    label: selective_cache(sc_fact[label], 30.0)
    for label in ("first", "extra", "tuning")
}
repair224_syn = {
    mult: selective_cache(sc_syn[mult], 30.0)
    for mult in (0.5, 1.0, 2.0)
}

def repair224_probe(c, w, b, scale):
    score = np.asarray(c["x"] @ np.asarray(w, dtype=np.float32) + b,
                       dtype=np.float32)
    g = torch.from_numpy(np.asarray(c["gate"], dtype=np.float32))
    s = torch.from_numpy(score)
    tilt = (torch.nn.functional.gelu(g + s) -
            torch.nn.functional.gelu(g)).numpy()
    logits = np.asarray(c["logits"] +
                        scale * tilt[..., None] * sc_rank,
                        dtype=np.float32)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).mean(0).numpy()
    means = np.sum(probs * c["centres"], axis=-1)
    nll = -np.log(np.maximum(
        probs[np.arange(len(c["bins"])), c["bins"]], 1e-30)).mean()
    return float(nll), means

repair224_rows = []
for repair224_scale in (6.0, 7.0, 8.0):
    for repair224_db in (-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015):
        repair224_b = repair224_fit["b"] + repair224_db
        repair224_row = {"scale": repair224_scale, "db": repair224_db}
        for repair224_label, repair224_c in repair224_fact.items():
            repair224_nll, _ = repair224_probe(
                repair224_c, repair224_fit["w"], repair224_b, repair224_scale)
            repair224_row[repair224_label] = (
                repair224_nll - float(repair224_c["original_nll"]))
        repair224_mu = {
            mult: repair224_probe(c, repair224_fit["w"], repair224_b,
                                  repair224_scale)[1]
            for mult, c in repair224_syn.items()
        }
        repair224_mask = (
            repair224_syn[1.0]["meta"]["event"].to_numpy() == "glucose_finger")
        for repair224_mult, repair224_label in ((0.5, "half"), (2.0, "double")):
            repair224_row[repair224_label] = float(np.mean(
                (repair224_mu[repair224_mult] - repair224_mu[1.0])[repair224_mask]))
        repair224_rows.append(repair224_row)

repair224_grid = pd.DataFrame(repair224_rows)
print("\nINTERCEPT SCREEN (CPU proxy, not GPU acceptance)")
print(repair224_grid.round(6).to_string(index=False))

for repair224_k in (24, 26):
    repair224_df = pd.read_csv(
        f"/work/fwd_{repair224_k}/factual_edited.csv")
    print("\nGPU CSV", repair224_k, "columns:", repair224_df.columns.tolist())
    print(repair224_df.head(1).to_dict("records"))

repair224_sections = Path("/work/notes.md").read_text().split("\n## ")
print("\nIDENTIFICATION SECTION TITLES")
print("\n".join(
    s.split("\n", 1)[0] for s in repair224_sections
    if any(t in s.split("\n", 1)[0].lower()
           for t in ("rd", "norepinephrine", "ne ", "discontinuity"))
)[:1800])
```
