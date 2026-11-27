The factual penalty is +0.00556 nats, so this candidate fails despite its accurate insulin response. I’ll screen weaker and reversed norepinephrine injections, then test a weaker positive one. It must regain factual margin while retaining insulin’s response; the rollout will also show whether norepinephrine moves in the intended direction.

```python
import copy, numpy as np, torch, pandas as pd
from pathlib import Path

backup295 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
globalnames295 = ("W_repair", "reference_bank125", "repair_bank", "rW", "rp_linear")
globals_backup295 = {
    n: copy.deepcopy(globals()[n]) if isinstance(globals()[n], dict) else globals()[n]
    for n in globalnames295 if n in globals()
}

def mean_loss295(result):
    if isinstance(result, tuple) and len(result) == 3:
        loss = result[1]
    else:
        loss = r244_find_losses(result)
    loss = torch.as_tensor(loss, dtype=torch.float32)
    assert loss.ndim == 1 and loss.numel() == len(baseline_losses194)
    return float(loss.mean())

settings295 = [
    (3.0, 0.0, 27.0, 0.5),
    (3.0, 0.0, 27.0, -0.5),
    (3.0, 0.0, 27.0, -1.0),
    (3.0, 0.0, 27.0, 1.0),
]
screen295, candidates295 = {}, {}
try:
    original_weights295 = {
        n: torch.as_tensor(load_original(n), dtype=torch.float32).clone() for n in NAMES
    }
    original_nll295 = mean_loss295(factual_eval194(original_weights295))
    for setting295 in settings295:
        candidate_weights295 = asymmetric_ins279(
            setting295[0], setting295[1], setting295[2], ne_gain=setting295[3]
        )
        candidates295[setting295] = candidate_weights295
        nll295 = mean_loss295(factual_eval194(candidate_weights295))
        screen295[setting295] = {"nll": nll295, "delta": nll295-original_nll295}
finally:
    for n, value in globals_backup295.items():
        globals()[n] = value

print("ORIGINAL_CPU_NLL", original_nll295)
print("CPU_SCREEN", screen295)
chosen295 = (3.0, 0.0, 27.0, 0.5)
assert screen295[chosen295]["delta"] < 0.005
candidate295 = candidates295[chosen295]
oldnames295 = set(edited())
for n in NAMES:
    value295 = torch.as_tensor(candidate295[n], dtype=torch.float32).clone()
    original295 = torch.as_tensor(load_original(n), dtype=torch.float32)
    assert tuple(value295.shape) == tuple(SHAPES[n])
    assert torch.isfinite(value295).all()
    if not torch.equal(value295, original295):
        save_edit(n, value295)
    elif n in oldnames295:
        drop_edit(n)
print("SAVED_CANDIDATE", chosen295, "EDITS", edited())

stayframe295 = pd.read_csv("/data/tuning.csv", usecols=lambda c: "stay" in c.lower())
staycol295 = next(c for c in stayframe295 if "stay" in c.lower())
all_tuning_ids295 = sorted(stayframe295[staycol295].drop_duplicates().astype(int).tolist())
print("TUNING_IDS", len(all_tuning_ids295), all_tuning_ids295[0], all_tuning_ids295[-1])
if all_tuning_ids295[-1] - all_tuning_ids295[0] < 1000:
    print("MISSING_IDS_IN_RANGE",
          sorted(set(range(all_tuning_ids295[0], all_tuning_ids295[-1]+1))
                 - set(all_tuning_ids295)))

with open("/work/notes.md", "a") as f:
    f.write("\n## GPU passes 38–39; Run 295\nGate-27 earlier-NE gain 1.4 "
            "rollout -15.003/+7.246, NE -0.625/+0.464. Factual GPU penalty "
            "+0.00556 on 98 stays: rejected. Screened weaker/reversed NE "
            f"injections against fresh original CPU NLL {original_nll295:.8f}: "
            f"{screen295}. Saved {chosen295}; GPU validation pending. "
            "No new causal evidence; targets remain record-derived estimates.\n")
```