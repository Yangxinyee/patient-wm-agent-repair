The restored edit increased factual NLL by 0.00671 on this cohort, so it is not safe to submit yet. I’ll inspect the existing cancellation and calibration methods alongside the causal estimates. The next change must lower factual loss while preserving the repaired dose response.

```python
import inspect, dis, io, contextlib, re
from pathlib import Path
import pandas as pd

remaining_print_budget = 9000
def bounded_print(label, text, cap=2000):
    global remaining_print_budget
    out = "\n=== " + label + " ===\n" + str(text)[:cap]
    out = out[:remaining_print_budget]
    print(out)
    remaining_print_budget -= len(out) + 1

def function_text(fn):
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dis.dis(fn)
        return buf.getvalue()

for name in ["r246_candidate", "r246_loss", "c211_moment_calibration"]:
    if name in globals():
        bounded_print(name, function_text(globals()[name]), 2300)

bounded_print("CACHE / STATE FUNCTIONS", [
    (k, inspect.getfullargspec(v).args)
    for k, v in sorted(list(globals().items()))
    if inspect.isfunction(v) and re.search(
        r"cache|state|precompute|prepare|probe|calibr", k, re.I)
], 1700)

txt = Path("/work/notes.md").read_text()
secs = re.split(r"(?=^## )", txt, flags=re.M)
bounded_print("EARLY SECTION HEADINGS",
              "\n".join(s.splitlines()[0] for s in secs[1:45]), 1800)
early_evidence = [
    s for s in secs[1:70]
    if re.search(r"horizon|impulse|adstock|target|g.computation|"
                 r"causal estimate|drug.effects|identif", s, re.I)
]
for s in early_evidence[-5:]:
    if remaining_print_budget > 300:
        bounded_print("EARLY CAUSAL CONCLUSION", s[-950:], 950)

fo35 = pd.read_csv("/work/fwd_35/factual_original.csv")
fe35 = pd.read_csv("/work/fwd_35/factual_edited.csv")
delta35 = float(fe35.nll.mean() - fo35.nll.mean())
print("\nForward 35 factual delta:", delta35)
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward 35 / Run 251: current checkpoint needs more factual margin\n"
        f"NE .50 cancellation checkpoint e042bb7889 has tuning0-99 factual "
        f"NLL delta {delta35:.7f}, exceeding .005 on this cohort. Earlier "
        "tuning100-199 delta was .0047104. Retained weights only for diagnosis; "
        "not accepted for submission. Recovering candidate and training-only "
        "calibration interfaces before selecting a safer final edit.\n"
    )
```
