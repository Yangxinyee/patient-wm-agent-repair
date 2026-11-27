"""The unit of the expected-dose pathway's dose-units input, per drug: the drug's typical training rate, exp of the mean
log rate over the training infusions, as stored in the full script's model (config "effect_raw_centre").

    python scripts/dose_units.py   -> results/dose_units.json (CPU; reads work/runs/<run>/model.pt)
"""
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = "synth_e03-m-T1-full-scriptS-pnp-s0"
UNITS = {"insulin": "U/h", "norepinephrine": "µg/kg/min"}

cfg = torch.load(ROOT / "work" / "runs" / RUN / "model.pt", map_location="cpu", weights_only=False)["config"]
out = {"run": RUN}
for d, drug in enumerate(UNITS):
    c = float(cfg["effect_raw_centre"][d])
    out[drug] = {"mean_log_rate": c, "typical_rate": math.exp(c), "unit": UNITS[drug]}
(ROOT / "results" / "dose_units.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(out)
