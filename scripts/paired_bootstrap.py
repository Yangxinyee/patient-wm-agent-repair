"""Paired stay bootstrap of the three estimators: on each copy of a world's training records resampled by stay, Claude's
and Codex's estimation code (their Python runs replayed unchanged in the sandbox, as scripts/replay_bootstrap.py) and the
scripted estimator (scripts/scripted_estimate.py) all run on the same copy, so their differences are paired.

    python3 scripts/paired_bootstrap.py <e03|n03> <first draw> <last draw> [--workers 8]
        -> results/paired_bootstrap/<world>.jsonl, one line per copy (draw -1: the original records)

Per copy, the implied insulin x2 shift (mg/dL) of
  claude          Claude's own estimate of the tuning-stay readout scaled by its re-estimated cumulative effect (as
                  scripts/replay_bootstrap.py)
  script_tuning   the scripted estimator's kernel applied to the 400 tuning stays: the stays Claude's estimate refers to
  codex           Codex's sustained shift on the training stays (its run 24)
  script_train    the scripted estimator's kernel applied to the same copy's training stays, pooled as Codex pools them
so claude - script_tuning and codex - script_train compare estimates of the same quantity. CPU, synthetic data only.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import replay_bootstrap as rb  # noqa: E402
import scripted_estimate as se  # noqa: E402


def replay(run: str, world: str, data: Path, draw: int) -> float | None:
    from agent.kernel import Kernel
    work = rb.BOOT / f"work_paired_{run}_{world}_{draw}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    cells = [c for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= rb.LAST[run]]
    k = Kernel(work, ROOT / "agent_data" / world / "model", data, "white")
    outputs = {}
    try:
        for c in cells:
            outputs[c["run"]] = k.run(c["code"])[0]
    finally:
        k.kill()
        shutil.rmtree(work, ignore_errors=True)
    try:
        return rb.implied(run, outputs)
    except Exception:  # noqa: BLE001
        return None


def panel_typed(csv: Path) -> dict:
    """scripted_estimate.panel with the columns' types given rather than inferred: in a resampled copy the first rows may
    hold no rate, and inference would then read the rate column as text"""
    import polars as pl
    real = pl.read_csv

    def typed(f, **kw):
        return real(f, schema_overrides={"stay_id": pl.Int64, "hour": pl.Float64, "value": pl.Float64, "rate": pl.Float64}, **kw)
    se.pl.read_csv = typed
    try:
        return se.panel(csv)
    finally:
        se.pl.read_csv = real


def scripted(world: str, data: Path) -> tuple[float, float]:
    tr = panel_typed(data / "train.csv")
    tu = se.panel(ROOT / "agent_data" / world / "data" / "tuning.csv")
    with contextlib.redirect_stdout(io.StringIO()):
        k = se.kernel(se.tsls(tr, se.deviations(tr), "I", "glucose_finger"))
    return se.implied(tu, "I", "glucose_finger", k, 2.0)[0], se.implied(tr, "I", "glucose_finger", k, 2.0)[0]


def one(args) -> dict:
    world, draw = args
    data = rb.resampled("paired", world, draw)
    try:
        try:
            st, sr = scripted(world, data)
        except Exception as e:  # noqa: BLE001  (recorded, not fatal to the other copies)
            st = sr = None
            print(f"draw {draw}: scripted estimator failed: {type(e).__name__}: {e}", flush=True)
        out = {"world": world, "draw": draw, "claude": replay("claude_a", world, data, draw),
               "codex": replay("openai_a", world, data, draw), "script_tuning": st, "script_train": sr}
    finally:
        if draw >= 0:
            shutil.rmtree(data, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world", choices=("e03", "n03"))
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dest = ROOT / "results" / "paired_bootstrap" / f"{a.world}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["draw"] for x in open(dest)} if dest.is_file() else set()
    todo = [(a.world, d) for d in range(a.first, a.last + 1) if d not in done]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, todo):
            with open(dest, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
