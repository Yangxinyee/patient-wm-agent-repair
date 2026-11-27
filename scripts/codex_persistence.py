"""The one-hour effect and persistence that Codex's third main-world run (openai_e) fits in each world, read from
its replayed run 5 ("RD dynamic fit [beta, persistence]"); the persistence is bounded above by its optimiser.

    python scripts/codex_persistence.py   -> results/codex_persistence.json     (CPU, about a minute; synthetic data only)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "agent")); sys.path.insert(0, str(ROOT / "scripts"))
import estimator_audit as ea  # noqa: E402
import replay_reps_codex as rx  # noqa: E402

WORLDS = ["n03", "n4101", "n4102", "n4103", "n4104", "n4105", "e03", "e5101", "e5102", "e5103", "e5104", "e5105"]


def main() -> None:
    out = {}
    for w in WORLDS:
        o = rx.replay("openai_e", ea.data_dir(w), ROOT / "agent_data" / ea.worlds()[w][3] / "model",
                      ROOT / "work" / "codex_persistence" / w)
        b, r = (float(x) for x in re.search(r"RD dynamic fit \[beta, persistence\]: \[\s*(\S+)\s+(\S+)\s*\]", o[5]).groups())
        out[w] = {"beta": b, "persistence": r}
        print(w, out[w], flush=True)
    (ROOT / "results" / "codex_persistence.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
