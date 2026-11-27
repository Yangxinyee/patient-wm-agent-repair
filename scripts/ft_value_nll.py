"""Fine-tuning's held-out value NLL per reading, before and after, for the three main-world seeds, from the table of the
fine-tuning runs in results/controls.md, written to results/ft_value_nll.json so the release reads plain JSON.

    python3 scripts/ft_value_nll.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md = (ROOT / "results" / "controls.md").read_text()
rows = re.findall(r"\| \S+ \| s(\d) \|.*\| ([0-9.]+) → ([0-9.]+) \|", md)        # main-world rows carry the NLL column
out = {f"s{s}": {"plain": float(a), "fine_tuned": float(b), "change": round(float(b) - float(a), 4)} for s, a, b in rows}
assert sorted(out) == ["s0", "s1", "s2"], out
(ROOT / "results" / "ft_value_nll.json").write_text(json.dumps({"world": "e03", "value_nll_per_reading": out}, indent=1) + "\n")
print(out)
