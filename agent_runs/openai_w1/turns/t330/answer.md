The notes provide six-hour response estimates consistent with insulin lowering glucose and norepinephrine raising MAP, subject to the identification assumptions. Earlier recurrent edits used incorrectly reconstructed features; the new features passed that check. I’ll test a modest injection using the corrected, feedback-constrained projections. Paired factual loss is the first gate: exceeding the limit will reject this candidate before rollout testing.

```python
import json, numpy as np, pandas as pd, torch
from pathlib import Path

print("Run 289/300; forwards 30/50.")
baseline289 = {n: np.asarray(load(n)).copy() for n in edited()}
P289 = np.asarray(fits286[("feedback4", 10.0)], dtype=np.float64)
B289 = np.asarray(B286, dtype=np.float64)
gain289 = np.array([-2.0, 4.0])
bname289 = next(n for n in NAMES if n in ("core.b_in", "core.b_in.weight"))
delta289 = (B289 * gain289[None, :]) @ P289.T
assert delta289.shape == tuple(SHAPES[bname289])
assert np.isfinite(delta289).all()
candidate289 = {**baseline289,
                bname289: np.asarray(load_original(bname289)) + delta289.astype(np.float32)}
save_edit(bname289, candidate289[bname289])

print("Installed", bname289, "gains", gain289.tolist(),
      "edit RMS", float(np.sqrt(np.mean(delta289**2))))
print("Projection immediate feedback max",
      float(np.max(np.abs(P289.T @ np.asarray(Jimm286)))))
print("Projection RMS", np.sqrt(np.mean(P289**2, axis=0)).tolist())
print("Injection direction norms", np.linalg.norm(B289, axis=0).tolist())

print("=== BASELINE VALIDATION ===")
for k in [23, 26]:
    for p in sorted(Path(f"/work/fwd_{k}").glob("factual_*.csv")):
        d = pd.read_csv(p)
        cols = [c for c in d.columns if "nll" in c.lower()]
        print(k, p.name, len(d), d[cols].mean(numeric_only=True).to_dict())
for k in [17, 27]:
    for p in sorted(Path(f"/work/fwd_{k}").glob("dose_edit_*.csv")):
        d = pd.read_csv(p)
        print(k, p.name, list(d.columns))
        ec = next((c for c in d.columns if "edit" in c.lower()
                   or "intervention" in c.lower()), None)
        sc = next((c for c in d.columns if "shift" in c.lower()), None)
        ic = next((c for c in ("stay_id", "stay") if c in d.columns), None)
        if ec and sc:
            if ic:
                print(d.groupby([ec, ic])[sc].mean().groupby(level=0).mean().to_dict())
            else:
                print(d.groupby(ec)[sc].mean().to_dict())
        else:
            print(d.head(2).to_string(index=False))

dids289 = pd.read_csv("/work/fwd_29/factual_original.csv")
idcol289 = next(c for c in ("stay_id", "stay") if c in dids289.columns)
ids289 = [int(v) for v in dids289[idcol289].drop_duplicates().tolist()[:100]]
tcols289 = pd.read_csv("/data/tuning.csv", nrows=1).columns.tolist()
tidcol289 = next(c for c in ("stay_id", "stay") if c in tcols289)
tids289 = set(pd.read_csv("/data/tuning.csv", usecols=[tidcol289])[tidcol289].unique())
split289 = "tuning" if set(ids289).issubset(tids289) else "train"
print("NEXT FACTUAL REQUEST")
print(json.dumps({"type": "factual", "model": "both", "split": split289, "stays": ids289}))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 289\nInstalled reversible corrected-feature recurrent candidate: original b_in plus B286 @ diag([-2,4]) @ fits286[feedback4,10].T, retaining baseline latent/relay edits. Immediate injection feedback is constrained; longer-loop and posterior-dose feedback remain unvalidated. Requires paired factual gate before rollout testing. Counts: 289 Python runs, 30 forwards.\n")
```
