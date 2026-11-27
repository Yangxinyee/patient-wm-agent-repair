The latest output repeats run 251, so the previous-bin cancellation test has not been reported. I’ll run that test now without changing weights. Lower loss on held-out training stays would support the correction; higher loss would refute it.

```python
import numpy as np, pandas as pd, torch, re
from pathlib import Path

print("=== EARLY EFFECT ESTIMATES ===")
secs = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
for s in secs:
    if re.match(r"## Run (5|7|8|10|11|12)\b", s.splitlines()[0]):
        print(s[:850])

print("\n=== TRANSITION TENSOR / CODE MAP ===")
print([(n, SHAPES[n]) for n in NAMES
       if "embed_transition.value_bin" in n])
print(VOCAB["codes"])

r252_head = {
    k: np.asarray(load(k), dtype=np.float32).copy()
    for k in r246_keys
}
r252_orig = {
    k: np.asarray(load_original(k), dtype=np.float32).copy()
    for k in r246_keys
}
r252_logits, r252_base = {}, {}
for label, c in [("train", c210_train), ("tune", c210_tune)]:
    lo = c211_logits(r252_orig, c).detach()
    le = c211_logits(r252_head, c).detach()
    assert lo.shape == le.shape
    r252_logits[label] = (lo, le)
    p0 = lo.softmax(-1).mean(0)
    r252_base[label] = -p0[
        torch.arange(len(c["y"])), c["y"].long()
    ].log().numpy()

r252_train_ids = np.sort(c210_train["meta"].stay_id.unique())
r252_fit = c210_train["meta"].stay_id.isin(r252_train_ids[:60]).to_numpy()
r252_val = ~r252_fit
r252_dmean = (
    r252_logits["train"][1] - r252_logits["train"][0]
).mean(0).numpy()
r252_dmean -= r252_dmean.mean(1, keepdims=True)

def r252_nll(logits, cache, correction=None):
    if correction is not None:
        logits = logits + torch.as_tensor(
            correction, dtype=logits.dtype
        )[None, :, :]
    p = logits.softmax(-1).mean(0)
    return -p[torch.arange(len(cache["y"])),
              cache["y"].long()].log().numpy()

def r252_fit_table(shrink):
    m = c210_train["meta"]
    table = {}
    for event in m.event.unique():
        event_mask = (m.event.to_numpy() == event) & r252_fit
        event_mean = r252_dmean[event_mask].mean(0)
        for lastbin in range(17):
            mask = event_mask & (m.last_bin.to_numpy() == lastbin)
            n = int(mask.sum())
            table[(event, lastbin)] = (
                r252_dmean[mask].sum(0) + shrink * event_mean
            ) / (n + shrink)
    return table

def r252_table_rows(table, cache):
    return np.stack([
        table[(e, int(b))]
        for e, b in zip(cache["meta"].event,
                        cache["meta"].last_bin)
    ])

r252_tables, r252_results = {}, []
for shrink in [5., 20., 80.]:
    table = r252_fit_table(shrink)
    r252_tables[shrink] = table
    corrections = {
        label: r252_table_rows(table, c)
        for label, c in [("train", c210_train), ("tune", c210_tune)]
    }
    for scale in [0., .5, 1., 1.25]:
        losses = {
            label: r252_nll(
                r252_logits[label][1], c,
                -scale * corrections[label]
            ) - r252_base[label]
            for label, c in [("train", c210_train), ("tune", c210_tune)]
        }
        r252_results.append({
            "shrink": shrink, "scale": scale,
            "train_fit": losses["train"][r252_fit].mean(),
            "train_validation": losses["train"][r252_val].mean(),
            "tune100": losses["tune"].mean()
        })

r252_screen = pd.DataFrame(r252_results)
print("\n=== TRAINING-ONLY PREVIOUS-BIN CANCELLATION SCREEN ===")
print(r252_screen.to_string(index=False))
print("\nNo tensors changed.")
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 252: previous-reading-bin factual cancellation screen\n"
        "Fitted mean edited-minus-original logits by event and last-value bin "
        "using 60 training stays, without fitting observed outcomes. Screened "
        "on 40 held-out training stays and the existing tuning cache. Direct "
        "dose sensitivity at fixed history is preserved; endogenous rollout "
        "effects remain unverified. No weights changed.\n" +
        r252_screen.to_string(index=False) + "\n"
    )
```
