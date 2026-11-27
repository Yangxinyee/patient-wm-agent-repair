"""Does the x3 column test the edits or the model's dose encoding?

The model reads an insulin rate through a robust score z = (rate - 1.5) / 1.0, clipped to [-10, 10] (so at 11.5 U/h), and
a fine bin (1 + the number of edges below the rate; edges up to 5.0 U/h, so every rate above 5 U/h shares bin 16)
(agent_data/e03/model/vocab.json). Over the held-out stays' insulin hours from hour 12 on (the edit window), this counts
the share of edited rates above the largest training rate, in the top bin and clipped in z, at x1, x2 and x3.

    python scripts/x3_encoding.py   -> results/x3_encoding.json     (CPU, seconds; synthetic data only)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from backend import sim  # noqa: E402  (the simulator, called and never changed)

voc = json.loads((ROOT / "agent_data" / "e03" / "model" / "vocab.json").read_text())
ins = next(d for d in voc["numeric"] if d["code"] == "insulin" and d["attribute"] == "rate")
EDGES, MED, IQR = np.array(ins["fine_bin_edges"]), ins["median"], ins["iqr"]
CLIP = MED + 10 * IQR
MAIN = sim.World(1, 1.0, 0.3)


def rates(n: int, seed: int, from_hour: int | None = None) -> np.ndarray:
    s = sim.simulate(MAIN, n, seed)
    r = np.round(s["INS"][:, sim.PRE + (from_hour or 0):sim.PRE + sim.HOURS], 4)
    return r[r > 0]


def main() -> None:
    train_max = float(rates(30_000, 101).max())
    ho = rates(600, 303, from_hour=12)
    out = {"train_max_rate": train_max, "top_bin_from": float(EDGES[-1]), "z_clip_at": float(CLIP), "held_out_hours": int(ho.size)}
    for k in (1, 2, 3):
        x = k * ho
        out[f"x{k}"] = {"above_train_max": float(np.mean(x > train_max)), "top_bin": float(np.mean(x > EDGES[-1])),
                        "z_clipped": float(np.mean(x > CLIP))}
    (ROOT / "results" / "x3_encoding.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
