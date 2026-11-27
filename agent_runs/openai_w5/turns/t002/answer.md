The architecture shows that doses enter a shared recurrent state, with no explicit drug-effect mechanism. That leaves room for learned treatment-selection correlations. I’ll inspect dose variation and event timing next; variation among otherwise similar histories would support estimating effects, while nearly deterministic dosing would limit identification.

```python
import pandas as pd
import time

t0 = time.time()
train = pd.read_csv(
    "/data/train.csv",
    dtype={"event": "category", "rate_unit": "category", "dose_unit": "category"}
)
tuning = pd.read_csv(
    "/data/tuning.csv",
    dtype={"event": "category", "rate_unit": "category", "dose_unit": "category"}
)

print("=== DATA SIZE / SCHEMA ===")
print("train:", train.shape, "tuning:", tuning.shape, "seconds:", round(time.time()-t0, 2))
print(train.dtypes.to_string())
print("stays:", train.stay_id.nunique(), tuning.stay_id.nunique())
print("events:\n", train.event.value_counts().to_string())
print("hour range:", train.hour.min(), train.hour.max())
print("fractional hours:", np.unique(np.round(train.hour.to_numpy() % 1, 6))[:80])

print("\n=== FIRST TRAINING STAY ===")
print(train.loc[train.stay_id == train.stay_id.iloc[0]].head(95).to_string(index=False))

print("\n=== NUMERIC DISTRIBUTIONS ===")
for ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff", "insulin", "norepinephrine"]:
    sub = train.loc[train.event == ev]
    col = "rate" if ev in ["insulin", "norepinephrine"] else "value"
    print(ev, "rows", len(sub), "quantiles",
          sub[col].quantile([0, .01, .1, .25, .5, .75, .9, .99, 1]).round(5).to_dict())
    if col == "rate":
        print("common rates:", sub.rate.value_counts().head(18).to_dict())
        print("duration:", (sub.end_hour-sub.hour).value_counts().head(5).to_dict())
        print("dose/rate:", (sub.dose/sub.rate).replace([np.inf, -np.inf], np.nan).dropna().round(6).value_counts().head(5).to_dict())

print("\n=== NORMALIZATION ===")
for entry in VOCAB["numeric"]:
    print(entry)

print("\n=== REMAINING VOCAB STRUCTURE ===")
for key, value in VOCAB.items():
    if key not in ["numeric", "codes", "config", "atom_kinds", "event_kinds"]:
        text = repr(value)
        print(key, text if len(text) < 2200 else text[:2200] + " ...")

print("\n=== SELECTED TENSORS ===")
for name in NAMES:
    if not any(x in name for x in [
        "core.prelude.", "core.blocks.", "core.coda.", "encode.mlp"
    ]):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Architecture inspection (run 1)\n"
        "- Doses influence posterior latent state and a generic recurrent action pathway.\n"
        "- Value heads also use a same-code last-reading register and transition logits.\n"
        "- Evaluation hides future readings but keeps recorded actions, so dose effects must "
        "be assessed with multi-step feedback, not just teacher-forced sensitivity.\n"
        "- No drug-effect estimate yet; inspect assignment variation and measurement timing next.\n"
    )
```
