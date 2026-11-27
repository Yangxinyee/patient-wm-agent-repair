"""How far the record's doses reach. For each glucose tier of the doctors' sliding scale (the tier of the
hour's glucose reading, finger-stick else laboratory: the reading a dose decision follows; hours without a reading hold
the previous dose and are left out), among the insulin-dosed stay-hours of the training records: the share whose dose is at least 1.5, 2 and 3 times the tier's scale dose, and the
share of stay-hours in a dosing tier with no insulin. The scale doses are the simulator's protocol (0, 1, 2, 3, 4, 6 U/h at
<150, 150-180, 180-220, 220-260, 260-300, >=300 mg/dL); doses carry a log-normal deviation of SD 0.3 and are rounded to
0.5 U/h. Training records only, CPU, synthetic data.

    python scripts/dose_support.py   -> results/dose_support.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scripted_estimate as se  # noqa: E402

EDGES = [150, 180, 220, 260, 300]
SCALE = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]
NAMES = ["<150", "150-179", "180-219", "220-259", "260-299", ">=300"]

p = se.panel(ROOT / "agent_data" / "e03" / "data" / "train.csv")
g = np.where(np.isfinite(p["glucose_finger"]), p["glucose_finger"], p["glucose_lab"])
tier = np.where(np.isfinite(g), np.searchsorted(EDGES, np.nan_to_num(g), side="right"), -1)   # -1: no reading this hour
I = p["I"]
out = {"tiers": {}}
for k, name in enumerate(NAMES):
    in_tier = tier == k
    dosed = in_tier & (I > 0)
    row = {"stay_hours": int(in_tier.sum()), "dosed": int(dosed.sum()), "scale_dose": SCALE[k]}
    if SCALE[k] > 0:
        r = I[dosed] / SCALE[k]
        for f in (1.5, 2.0, 3.0):
            row[f"ge_{f:g}x"] = float((r >= f - 1e-9).mean()) if len(r) else None
        row["zero_share"] = float((~(I > 0))[in_tier].mean())
    out["tiers"][name] = row
allr = np.concatenate([I[(tier == k) & (I > 0)] / SCALE[k] for k in range(1, 6)])
out["all_dosing_tiers"] = {f"ge_{f:g}x": float((allr >= f - 1e-9).mean()) for f in (1.5, 2.0, 3.0)}
out["all_dosing_tiers"]["n_dosed"] = int(len(allr))
z = np.concatenate([(~(I > 0))[tier == k] for k in range(1, 6)])
out["all_dosing_tiers"]["zero_share"] = float(z.mean())
(ROOT / "results" / "dose_support.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
