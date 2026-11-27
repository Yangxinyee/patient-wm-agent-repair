The current edit has large weight changes, yet the independent factual check still passed. That does not tell us whether its rollout dynamics are right. I’ll inspect how the lag estimates were obtained and compare their provisional sustained predictions with the rollout on exactly the same stay-hours. This will show the size of the remaining gap, while keeping estimated effects separate from truth.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== DYNAMIC-IDENTIFICATION NOTES ===")
paragraphs134 = Path("/work/notes.md").read_text().split("\n\n")
dynamic134 = [
    p for p in paragraphs134
    if any(s in p.lower() for s in [
        "deconvol", "impulse", "local projection", "distributed",
        "kernel55", "six-lag", "feedback", "run 55", "run 56"
    ])
]
print("\n\n".join(dynamic134)[:3500])

print("\n=== RETAINED IDENTIFICATION OBJECTS ===")
for name in ["kernel55", "dml_results", "two_pole_kernel88"]:
    if name in globals():
        obj = globals()[name]
        print(name, brief132(obj))
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False)[:1400])
print("Relevant names:", sorted(
    n for n in list(globals())
    if not n.startswith("_")
    and any(s in n.lower() for s in [
        "impulse", "decon", "local_proj", "lp_", "lag",
        "dml", "feedback", "response55"
    ])
))

print("\n=== AVAILABLE DOSE-ROLLOUT SUMMARIES ===")
dose_files134 = sorted(Path("/work").glob("fwd_*/dose_edit_*.csv"),
                       key=lambda p: int(p.parent.name.split("_")[-1]))
for path in dose_files134:
    frame = pd.read_csv(path)
    active = frame.loc[frame["reached"].astype(bool)]
    print(str(path), "rows", len(frame), "reached", len(active),
          active.groupby("edit")["shift"].mean().round(4).to_dict())

edited_files134 = [p for p in dose_files134 if p.name == "dose_edit_edited.csv"]
rollout134 = pd.read_csv(edited_files134[-1])
tuning134 = pd.read_csv("/data/tuning.csv")
drug_rows134 = tuning134.loc[
    tuning134.event.isin(["insulin", "norepinephrine"]),
    ["stay_id", "hour", "event", "rate"]
].copy()
drug_rows134["start_hour"] = np.floor(drug_rows134.hour + 1e-6).astype(int)
stay_ids134 = sorted(tuning134.stay_id.unique())
stay_lookup134 = {sid: i for i, sid in enumerate(stay_ids134)}

estimated_paths134 = {}
for drug in ["insulin", "norepinephrine"]:
    rates = (
        drug_rows134.loc[drug_rows134.event.eq(drug)]
        .pivot_table(index="stay_id", columns="start_hour",
                     values="rate", aggfunc="sum")
        .reindex(index=stay_ids134, columns=range(26))
        .fillna(0).to_numpy()
    )
    kernel = np.asarray(kernel_models88[drug], dtype=float)
    effect = np.zeros_like(rates)
    for hour in range(13, 25):
        for start in range(12, hour):
            effect[:, hour] += kernel[hour - start - 1] * rates[:, start]
    estimated_paths134[drug] = effect
    print(drug, "post-12 mean recorded rate", round(float(rates[:, 12:24].mean()), 4),
          "kernel sum", round(float(kernel.sum()), 4))

matched134 = rollout134.copy()
matched134["provisional_record_effect"] = 0.0
for edit, drug, multiplier in [
    ("insulin_x2", "insulin", 1.0),
    ("insulin_half", "insulin", -0.5),
    ("norepinephrine_x2", "norepinephrine", 1.0),
    ("norepinephrine_half", "norepinephrine", -0.5),
]:
    mask = matched134.edit.eq(edit)
    rows = matched134.loc[mask]
    if len(rows):
        i = np.array([stay_lookup134[sid] for sid in rows.stay_id])
        h = np.rint(rows.hour).astype(int).to_numpy()
        matched134.loc[mask, "provisional_record_effect"] = (
            multiplier * estimated_paths134[drug][i, h]
        )

print("\n=== MATCHED READOUTS: PROVISIONAL, NOT TRUE EFFECTS ===")
active134 = matched134.loc[matched134.reached.astype(bool)]
print(active134.groupby("edit")[["shift", "provisional_record_effect"]].mean().round(4))
print("Equal-stay averages:")
print(
    active134.groupby(["edit", "stay_id"])[["shift", "provisional_record_effect"]]
    .mean().groupby("edit").mean().round(4)
)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 134\n"
        "Recovered dynamic-identification notes and compared retained-kernel "
        "convolutions with the latest rollout on matching stay-hours. These "
        "convolutions are provisional record-based targets, not supplied truth. "
        "No weights changed.\n"
    )
```