"""An interval for Claude's first edit's total-NLL change, and the seed spread of the unedited model.

Teacher-forced on the 600 held-out stays as scripts/model_parts.py does (every loss part, random numbers fixed), but in
batches of 8 stays, so each batch's summed loss over every part is kept. The change is edited minus unedited, batch by
batch on the same stays; its 95% interval resamples the 75 batches (2,000 draws, seed 20261110). Yardstick: the unedited
models of training seeds 1 and 2 against seed 0, the same way.

    python scripts/total_nll_interval.py   (GPU, a few minutes) -> results/total_nll_interval.json
Synthetic data only; the simulator code is called, never changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "scripts")]
import backend  # noqa: E402
backend.configure(work_root=str(ROOT / "work"))
backend.configure(cohort="synth_e03")

import polars as pl  # noqa: E402
import torch  # noqa: E402

from backend import make_loader  # noqa: E402
from backend import teacher_forced as op  # noqa: E402

orig_load = op.load_model
SKIP = {"gate_target"}


def load(r: str, agent: str | None):
    model, cfg = orig_load(r)
    if agent:
        import fwd_job
        d = ROOT / "agent_runs" / agent
        arm = json.loads((d / "state.json").read_text())["arm"]
        sd, _ = fwd_job.edited_state({k: v.detach().cpu() for k, v in model.state_dict().items()}, d / "submission", arm)
        model.load_state_dict(sd)
    return model, cfg


def batch_losses(run: str, agent: str | None, stays: list[int]) -> tuple[np.ndarray, set]:
    op.load_model = lambda r: load(r, agent)
    model, _, root, _, _, _, spec, _ = op.setup(run)
    torch.manual_seed(0)
    loader = make_loader(root, "held_out", spec=spec, batch_size=8, shuffle=False, num_workers=0,
                         episode_filter=lambda ep: ep.filter(pl.col("subject_id").is_in(stays)))
    out, state, names = [], None, set()
    with torch.no_grad():
        for b in loader:
            B = b["bundle_mask"].shape[0]
            if state is None or state["h"].shape[0] != B:
                state = model.init_state(B, torch.device("cuda"))
            b = {k: v.cuda() for k, v in b.items()}
            o, state = model(b, state)
            d = {k: float(s) for k, (s, c) in o["parts"].items() if k not in SKIP and int(c) > 0}
            names |= set(d)
            out.append(d)
    del model
    torch.cuda.empty_cache()
    return out, names


def main() -> None:
    torch.cuda.set_per_process_memory_fraction(0.12)
    cuts = pl.read_parquet(op.oe.out_dir() / "cuts_held_out.parquet")
    stays = sorted(cuts["subject_id"].to_list())
    base, nb = batch_losses("synth_e03-m-T1-full-long-s0", None, stays)
    rng = np.random.default_rng(20261110)
    res = {"n_stays": len(stays), "n_batches": len(base), "pairs": {}}
    for label, run, agent in (("claude_a", "synth_e03-m-T1-full-long-s0", "claude_a"),
                              ("seed1", "synth_e03-m-T1-full-long-s1", None), ("seed2", "synth_e03-m-T1-full-long-s2", None)):
        oth, no = batch_losses(run, agent, stays)
        common = sorted(nb & no)
        diff = np.array([sum(o.get(k, 0.0) for k in common) - sum(b.get(k, 0.0) for k in common) for o, b in zip(oth, base, strict=True)])
        idx = rng.integers(0, len(diff), size=(2000, len(diff)))
        boot = diff[idx].sum(1) / len(stays)
        res["pairs"][label] = {"change_per_stay": float(diff.sum() / len(stays)),
                               "interval": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))], "parts": common}
        print(label, res["pairs"][label]["change_per_stay"], res["pairs"][label]["interval"], flush=True)
    (ROOT / "results" / "total_nll_interval.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
