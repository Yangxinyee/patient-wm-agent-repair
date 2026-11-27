The extra numeric changes are intentional training-average offsets, not changes to measurement inputs. The pre-dose checks still favor negligible effects, though delayed norepinephrine remains uncertain. I’ll record the compact evidence and final audit before submission. A coherent late-dose effect would challenge the zero-response edit; factual loss and rollout behavior already pass the available checks.

```python
from pathlib import Path
import ast
import re
import numpy as np

p = Path("/work/notes.md")
text = p.read_text()
late_checks = []

for heading in [
    "Strictly pre-dose insulin checks",
    "Strictly pre-dose norepinephrine checks",
    "Delayed insulin and dose-context checks",
]:
    start = text.find("## " + heading)
    end = text.find("\n## ", start + 3)
    section = text[start:end if end >= 0 else len(text)]
    for match in re.finditer(r"\[\{.*?\}\]", section, flags=re.S):
        try:
            rows = ast.literal_eval(match.group(0))
        except (SyntaxError, ValueError):
            continue
        for row in rows:
            subset = row.get("subset", "")
            if subset == "hour_ge12" or (
                row.get("drug") == "norepinephrine" and subset == "hour_ge4"
            ):
                late_checks.append({
                    key: row.get(key)
                    for key in ["drug", "horizon", "subset", "coef", "se", "ci95"]
                })

print("=== COMPACT LATE-DOSE EVIDENCE ===")
for row in late_checks:
    print(row)

expected = {
    "embed.numeric.weight",
    "embed.bins.dose.weight",
    "embed.bins.rate.weight",
}
assert set(edited()) == expected
w = np.asarray(load("embed.numeric.weight"))
w0 = np.asarray(load_original("embed.numeric.weight"))
assert np.isfinite(w).all()
assert np.all(w[:, [2, 4]] == 0)
assert np.array_equal(w[:, [0, 1]], w0[:, [0, 1]])
for name in expected - {"embed.numeric.weight"}:
    a = np.asarray(load(name))
    a0 = np.asarray(load_original(name))
    assert np.isfinite(a).all()
    assert np.array_equal(a[0], a0[0])
    assert np.array_equal(a[1:], np.broadcast_to(a[1], a[1:].shape))

summary = """
## Final evidence summary and candidate sign-off, Python run 39
No weights changed during restart recovery.
The three-tensor edit removes numerical dose/rate magnitude and collapses present
dose/rate bins to their empirical training-weighted means. Absent bins and measurement
numeric columns remain original. Presence columns intentionally absorb the removed
training-average numeric contributions; their changes are not unrelated edits.

Training-record evidence favors approximately null magnitude effects: prefix-only,
stay-cross-fitted insulin estimates are near zero at short and delayed horizons;
insulin threshold dose jumps do not produce convincing outcome jumps. NE short-horizon
estimates are near zero. Its isolated delayed endpoint signal remains uncertain:
early-stay associations and nuisance choices matter, and the 12-hour window-average
diagnostic is compatible with zero. These observational analyses do not prove
causal nulls or identify unrestricted extrapolation. Stability splits are not
independent replication.

All 400 tuning stays, 19,224 readings:
original factual NLL 1.5010981065; edited 1.5011797129;
delta +0.0000816063 nats/reading, versus allowed +0.005.
Stay-cluster 95% interval for the delta: [-0.0001035902, +0.0002668028].
This is validation evidence, not a guarantee for unseen evaluation stays.

Actual paired rollout on 100 exposure-enriched tuning stays returned exactly zero
for insulin x2, insulin half, NE x2 and NE half. It establishes the edited model's
response, not ground truth. Expected submitted readouts are all zero, with the
approximately-null causal interpretation more secure for insulin than delayed NE.
No additional nonzero response is imposed from the unstable NE endpoint estimate.
"""
p.write_text(text + summary + "\nCompact recovered estimates:\n" + repr(late_checks) + "\n")
print("\nFinal candidate audit passed; all current weights were left unchanged.")
```