The current edit passes the factual gate on 99 tuning stays: NLL improves by 0.00095 nats rather than worsening. I’ll recover the longer-term effect estimates and select untouched tuning stays for validation. If the record-derived effects are near the current rollout shifts, we should keep this edit; larger effects would justify testing stronger gains.

```python
from pathlib import Path
import re, numpy as np, pandas as pd

text293 = Path("/work/notes.md").read_text()
parts293 = re.split(r"(?=^## )", text293, flags=re.M)

print("RECORD IDENTIFICATION, COMPLETE EARLY SECTIONS")
for p293 in parts293:
    if re.match(r"## (?:Run |Runs )?(?:1[5-9]|20|21)\b", p293):
        print(p293[:3200])

print("\nNUMERIC DYNAMIC-EFFECT PASSAGES")
patterns293 = r"rho|autoregress|steady.state|persistence|cumulative effect|sustained effect|impulse response|dynamic response"
windows293 = []
for m293 in re.finditer(patterns293, text293, re.I):
    lo293, hi293 = max(0, m293.start()-180), min(len(text293), m293.end()+650)
    if windows293 and lo293 <= windows293[-1][1]:
        windows293[-1] = (windows293[-1][0], hi293)
    else:
        windows293.append((lo293, hi293))
budget293 = 2400
for lo293, hi293 in windows293:
    ss293 = text293[lo293:hi293]
    if re.search(r"\d+\.\d+", ss293):
        ss293 = ss293[:budget293]
        print(ss293)
        budget293 -= len(ss293)
        if budget293 <= 0:
            break

print("\nDYNAMIC OBJECT INVENTORY")
budget293 = 2300
for n293, v293 in list(globals().items()):
    if n293.startswith("_") or not re.search(
        r"dyn|impulse|rho|effect|panel|innovation|causal|lag|estimate|beta24|persist", n293, re.I
    ):
        continue
    if isinstance(v293, pd.DataFrame):
        ss293 = f"{n293}: DataFrame {v293.shape}, columns {v293.columns.tolist()[:20]}"
    elif isinstance(v293, dict):
        ss293 = f"{n293}: dict {list(v293)[:12]}"
    elif isinstance(v293, np.ndarray):
        ss293 = f"{n293}: array {v293.shape}"
        if v293.size <= 24:
            ss293 += " " + np.array2string(v293, precision=5)
    elif isinstance(v293, (list, tuple)):
        ss293 = f"{n293}: {str(v293)[:600]}"
    elif isinstance(v293, (float, int)):
        ss293 = f"{n293}: {v293}"
    else:
        continue
    print(ss293[:budget293])
    budget293 -= len(ss293)
    if budget293 <= 0:
        break

used293 = set(pd.read_csv("/work/fwd_32/factual_original.csv")["stay_id"])
tune293 = pd.read_csv("/data/tuning.csv")
allids293 = sorted(tune293["stay_id"].unique().tolist())
newids293 = [int(x) for x in allids293 if x not in used293]
rng293 = np.random.default_rng(293)
validation_ids293 = sorted(rng293.choice(newids293, min(100, len(newids293)), replace=False).tolist())
print("\nINDEPENDENT FACTUAL REQUEST")
print({"type": "factual", "model": "both", "split": "tuning", "stays": validation_ids293})
print("CURRENT GAIN RECONSTRUCTION ERROR",
      float(np.max(np.abs(a292(load("core.b_in.weight")) -
          (a292(load_original("core.b_in.weight")) +
           B286 @ np.diag([-3.0, 0.4]) @ fits286[("feedback4", 10.0)].T)))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 293\n"
            "Forward32 current candidate passes factual gate on 5059 readings: "
            "1.53096444 edited versus 1.53191401 original, delta -.00094958. "
            "Recovering complete dynamic record-effect evidence and selected 100 independent tuning "
            "stays for validation. No weights changed. Counts: 293 Python runs, 33 forwards.\n")
```
