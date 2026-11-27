The stronger-NE cancellation trials still exceed the factual-loss limit, so I have not installed them. The approximate probe may also protect the wrong direction. I’ll check whether modest cancellation buys any useful margin and recover any machinery for perturbing actual dose bins. Only an actual dose perturbation or rollout can confirm preserved treatment sensitivity.

```python
import re, numpy as np, pandas as pd
from pathlib import Path

r241_results = []
for (npc, ridge), reg in r240_regressions.items():
    if npc < 0:
        continue
    for amp in (0.4, 0.45, 0.5, 0.55):
        factor = amp/0.4
        for frac in (0.25, 0.5, 1.0):
            losses = {}
            for label, c in (("train", c210_train), ("tune", c210_tune)):
                scalar = ((factor-1)*r240_pre[label]["R"]
                          - factor*frac*reg["pred"][label])
                losses[label] = r240_loss(c, r240_pre[label], scalar)-c["loss0"]
            r241_results.append({
                "npc": npc, "ridge": ridge, "ne_amp": amp, "cancel": frac,
                "train_fit": float(losses["train"][r240_fit].mean()),
                "train_validation": float(losses["train"][r240_val].mean()),
                "tune100": float(losses["tune"].mean())})
r241_table = pd.DataFrame(r241_results)
print("=== MODEST CANCELLATION SCREEN ===")
print(r241_table.sort_values(["tune100", "train_validation"]).head(12).to_string(index=False))
print("\nBest by NE amplitude:")
print(r241_table.sort_values("tune100").groupby("ne_amp", as_index=False).first().to_string(index=False))

print("\n=== CANCELLATION RESIDUALS BY EVENT ===")
r241_reg = r240_regressions[(16, 0.001)]
for label, c in (("train", c210_train), ("tune", c210_tune)):
    R = r240_pre[label]["R"].numpy()
    P = r241_reg["pred"][label].numpy()
    for ev in c["meta"]["event"].unique():
        m = (c["meta"]["event"].to_numpy() == ev)
        print(label, ev, "target mean/sd", round(float(R[:,m].mean()),4),
              round(float(R[:,m].std()),4), "prediction mean/sd",
              round(float(P[:,m].mean()),4), round(float(P[:,m].std()),4),
              "RMSE", round(float(np.sqrt(np.mean((R[:,m]-P[:,m])**2))),4))

print("\n=== EXISTING STATE/DOSE COMPUTATION FUNCTIONS ===")
r241_funcs = []
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and hasattr(obj, "__code__"):
        if any(t in name.lower() for t in
               ("step", "encode", "bundle", "atom", "core", "probe", "rollout", "dose")):
            co = obj.__code__
            r241_funcs.append((name, co.co_varnames[:co.co_argcount], co.co_names))
for item in r241_funcs[:70]:
    print(item)

print("\n=== SELECTED CONTROLLED-PROBE OBJECTS ===")
for name in sorted(list(globals())):
    if name.startswith(("r124_", "r125_", "r137_", "r138_", "probe84_", "r198_", "r199_", "r200_")):
        obj = globals()[name]
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:12])
        elif isinstance(obj, (int, float, str, list, tuple)):
            print(name, repr(obj)[:180])

print("\n=== RECENT EXACT-DOSE INVESTIGATIONS ===")
for section in Path("/work/notes.md").read_text().split("\n## "):
    match = re.search(r"Run(?:s)?\s+(\d+)", section[:80])
    if match and 198 <= int(match.group(1)) <= 206:
        print(("## "+section)[:1500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 241: cancellation limitations and actual-dose recovery\n"
            "Stronger-NE cancellation screen failed the .005 factual budget; no trial "
            "installed. Checked smaller-amplitude cancellation and event leakage, "
            "and inventoried existing actual-dose/state-probe machinery. Current "
            "validated safe tensors retained.\n")
    f.write(r241_table.sort_values("tune100").groupby(
        "ne_amp", as_index=False).first().to_string(index=False)+"\n")
```