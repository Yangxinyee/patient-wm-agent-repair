"""A control for the probe. The expected dose here is the sliding scale's rate at the latest reading, so
any model that predicts the next reading may carry it. This fits the same linear probe (ridge on standardised features,
the same training and held-out rows as scripts/probe.py record_only) on the dose and the latest reading alone,
[log rate, log rate^2, latest reading, latest reading^2], with no model state, and reports the held-out R^2 of the log
surprise dose s beside the dose-only and the record-ceiling probes already in results/probe.json.

    python scripts/probe_control.py   -> results/probe_control.json        (CPU, about a minute; synthetic data only)
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe as pr  # noqa: E402


def with_last(world, split: str) -> pl.DataFrame:
    """probe.truth plus the latest glucose reading the doctor saw (the same forward fill as probe.truth's scale)"""
    n, seed = pr.SPLITS[split]
    s = pr.sim.simulate(world, n, seed)
    g = np.where(np.isnan(s["FSG"]), s["GLU"], s["FSG"])
    last = g.copy()
    for t in range(1, last.shape[1]):
        last[:, t] = np.where(np.isnan(last[:, t]), last[:, t - 1], last[:, t])
    rows = []
    for h in range(1, pr.sim.HOURS):
        c = pr.sim.PRE + h
        rows.append(pl.DataFrame({"subject": s["subject"].astype(np.int64), "hour": h, "last": last[:, c]}))
    return pr.truth(world, split).join(pl.concat(rows), on=["subject", "hour"], how="inner")


def main() -> None:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {}
    for key in ("e03", "n03"):
        _, cohort, world = pr.WORLDS[key]
        tr = with_last(world, "train").join(pr.mhat(cohort, "train"), on=["subject", "hour"], how="inner")
        tr = tr.filter(pl.col("subject").is_in(tr["subject"].unique().sort().head(pr.N_TRAIN).implode()))
        ho = with_last(world, "held_out").join(pr.mhat(cohort, "held_out"), on=["subject", "hour"], how="inner")

        def feats(f):
            lr, g = np.log(f["ins"].to_numpy()), f["last"].to_numpy() / 100
            return np.c_[lr, lr ** 2, g, g ** 2]
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(feats(tr), tr["s"].to_numpy())
        r2 = pr.r2_boot(ho["s"].to_numpy(), m.predict(feats(ho)), ho["subject"].to_numpy())
        out[key] = {"dose_and_latest_reading": r2, "rows_train": tr.height, "rows_held_out": ho.height}
        print(key, out[key], flush=True)
    (ROOT / "results" / "probe_control.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
