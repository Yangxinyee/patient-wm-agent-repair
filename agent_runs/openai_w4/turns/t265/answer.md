The original gate appears to suppress large raw signals on inactive bundles. A downward bias shift would preserve that suppression, unlike gate compression. I’ll screen threshold shifts and modest insulin-response gains on the training replay. Factual-safe candidates would justify a rollout test; excessive NLL increases would reject this approach.

```python
import numpy as np, torch, time
from pathlib import Path

current232 = {n: load(n).detach().clone().float() for n in NAMES}
original232 = {n: load_original(n).detach().clone().float() for n in NAMES}
cpu_backup232 = {k: globals()[k] for k in
                 ["rW", "repair_bank", "rp_linear", "reference_bank125"] if k in globals()}
results232, candidates232 = {}, {}
start232 = time.monotonic()

try:
    _, losses_o232, _ = factual_eval194(original232)
    _, losses_c232, _ = factual_eval194(current232)
    base_nll232 = float(np.mean(losses_o232))
    print("Original/current CPU NLL:", base_nll232, float(np.mean(losses_c232)))
    print("Insulin connection norm:", float(torch.as_tensor(ins_unit206).norm()))
    print("Active94 counts:", np.unique(np.asarray(active94), return_counts=True))

    raw232 = np.asarray(r217_raw, dtype=np.float64)
    pair232 = np.asarray(reserved94, dtype=int)
    wpair232 = current232["z_post.0.weight"][pair232].numpy().astype(np.float64)
    bpair232 = current232["z_post.0.bias"][pair232].numpy().astype(np.float64)
    pre232 = raw232 @ wpair232.T + bpair232
    mask232 = np.asarray(active94, dtype=bool)
    for label232, select232 in [("insulin", mask232), ("other", ~mask232)]:
        print(label232, "gate quantiles",
              np.round(np.quantile(pre232[select232].mean(axis=1), [.01,.1,.5,.9,.99]),3),
              "max pair activation", float(pre232[select232].max()))

    for shift232 in [16.0, 20.0, 23.0, 26.0, 30.0]:
        for gain232 in [1.0, 1.5]:
            cand232 = dict(current232)
            cand232["z_post.0.bias"] = current232["z_post.0.bias"].clone()
            cand232["z_post.0.bias"][pair232] -= shift232
            cand232["core.b_in.weight"] = (
                current232["core.b_in.weight"] +
                (gain232-1.0)*torch.as_tensor(ins_unit206, dtype=torch.float32))
            _, loss232, _ = factual_eval194(cand232)
            key232 = (shift232, gain232)
            delta232 = float(np.mean(loss232))-base_nll232
            results232[key232] = {"nll": float(np.mean(loss232)), "delta": delta232}
            candidates232[key232] = cand232
            print(key232, "dNLL", round(delta232,6))
finally:
    for k232, value232 in cpu_backup232.items():
        globals()[k232] = value232

print("Elapsed seconds:", round(time.monotonic()-start232,1), "saved edits unchanged")
with open("/work/notes.md", "a") as f232:
    f232.write("\n## Run 232\nScreened UNSAVED insulin paired-gate downward bias shifts 16/20/23/26/30 with response gains 1/1.5. Unlike compression, downward shifts cannot activate previously suppressed non-insulin bundles. CPU factual results: " + repr(results232) + ". CPU globals restored in finally; no saved weights changed. GPU rollout needed to test asymmetry.\n")
```