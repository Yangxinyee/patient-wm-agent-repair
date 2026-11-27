The recovered notes support provisional next-reading effects near −2 for insulin and +3 for norepinephrine per rate unit. Those are not the cumulative rollout targets. The output was truncated before the checkpoint and cohort checks, so I’ll retrieve only those essentials before validation.

```python
import os, re, json
import numpy as np
import pandas as pd

print("=== CHECKPOINT ===")
cp298 = np.load("/work/protected_head295.npz", allow_pickle=False)
differences298 = {}
for n298 in edited():
    differences298[n298] = (
        float(np.max(np.abs(as_numpy297(load(n298)) - cp298[n298])))
        if n298 in cp298.files else "not in checkpoint"
    )
print(json.dumps(differences298))
protected_state298 = {n: as_numpy297(load(n)).copy() for n in edited()}

print("=== TUNING IDS ===")
for i298, ids298 in enumerate(tuning_cohorts297):
    contiguous298 = all(b == a+1 for a, b in zip(ids298[:-1], ids298[1:]))
    print(i298, "count", len(ids298), "first", ids298[0],
          "last", ids298[-1], "contiguous", contiguous298)
print("FIRST FACTUAL REQUEST:")
print(json.dumps({
    "type": "factual", "model": "both", "split": "tuning",
    "stays": tuning_cohorts297[0]
}))

print("=== NUMERIC TARGET / PERSISTENCE NOTES ===")
targetlines298 = [
    s for s in notes297.splitlines()
    if re.search(r"target|persist|impulse|cumul|decay", s, re.I)
    and re.search(r"\d", s)
]
for s298 in (targetlines298[:8] + targetlines298[-12:]):
    print(s298[:230])

print("=== RECENT FORWARD SUMMARY ===")
for idx298 in range(34, 40):
    d298 = f"/work/fwd_{idx298}"
    if not os.path.isdir(d298):
        continue
    print(idx298, sorted(os.listdir(d298)))
    for m298 in ("original", "edited"):
        p298 = f"{d298}/factual_{m298}.csv"
        if os.path.exists(p298):
            f298 = pd.read_csv(p298)
            cols298 = [
                c for c in f298
                if "nll" in c.lower()
                and pd.api.types.is_numeric_dtype(f298[c])
            ]
            print(m298, len(f298), f298[cols298].mean().to_dict())

print("=== CALIBRATION OBJECTS ===")
for k298 in sorted(list(globals())):
    if re.search(r"295|296", k298) and not k298.startswith("_"):
        v298 = globals()[k298]
        if callable(v298):
            print(k298, "callable")
        elif hasattr(v298, "shape"):
            print(k298, tuple(v298.shape))
        elif isinstance(v298, (int, float, str)):
            print(k298, str(v298)[:100])

with open("/work/notes.md", "a") as log298:
    log298.write(
        "\n## Python 298 — compact final-validation inventory\n"
        "- Run297 output was truncated; retrieved checkpoint agreement, tuning cohort IDs, and numeric target notes compactly.\n"
        "- Preserved the installed tensors in protected_state298. No weights changed.\n"
        "- Next decision will use matched GPU factual and dose-rollout results, not the failed local diagnostic.\n"
    )
```