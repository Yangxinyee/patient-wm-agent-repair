The rejected detector still activates on factual states, and the earlier hidden-feature correction did not reliably improve likelihood. I’ll test a simpler possibility: the baseline’s value predictions may be slightly overconfident. A scalar probability calibration is useful only if it improves held-out record likelihood consistently; otherwise I’ll leave the head unchanged.

```python
import numpy as np
import torch
from pathlib import Path

parts205 = build_static167(tm162)
print("Static builder output:", type(parts205).__name__,
      [getattr(x, "shape", type(x).__name__) for x in parts205])
assert len(parts205) == 3, "Inspect builder output before decoding."
static205, transition205, centres205 = parts205
assert tuple(static205.shape) == (len(tm162), 192)
assert tuple(transition205.shape) == (len(tm162), 17)
assert tuple(centres205.shape) == (len(tm162), 17)

with np.load("/work/fwd_22/states_original.npz") as a:
    po205, ho205, lo205 = decode_states175(
        a, tm162, static205, transition205, centres205, draws=16, seed=205)
with np.load("/work/fwd_22/states_edited.npz") as a:
    pe205, he205, le205 = decode_states175(
        a, tm162, static205, transition205, centres205, draws=16, seed=205)

y205 = tm162.target_bin.to_numpy(dtype=int)
sid205 = tm162.stay_id.to_numpy()
rng205 = np.random.default_rng(205)
u205 = rng205.permutation(np.unique(sid205))
fold205 = {s: j % 5 for j, s in enumerate(u205)}
f205 = np.array([fold205[s] for s in sid205])
temps205 = np.linspace(0.85, 1.15, 31)

def marginal_temperature205(p, t):
    lp = np.log(np.maximum(p, 1e-30)) * t
    lp -= lp.max(axis=1, keepdims=True)
    q = np.exp(lp)
    return q / q.sum(axis=1, keepdims=True)

def nll205(p):
    return -np.log(np.maximum(p[np.arange(len(y205)), y205], 1e-30))

print("Original/base local NLL:", nll205(po205).mean(), nll205(pe205).mean())
losses205 = np.stack([nll205(marginal_temperature205(pe205, t)) for t in temps205])
best205 = []
cv_loss205 = np.zeros(len(y205))
for f in range(5):
    train, test = f205 != f, f205 == f
    j = np.argmin(losses205[:, train].mean(axis=1))
    best205.append(float(temps205[j]))
    cv_loss205[test] = losses205[j, test]
print("Fold-selected marginal temperatures:", best205)
print("Held-out change relative to baseline:",
      float(np.mean(cv_loss205 - nll205(pe205))))
print("Held-out change relative to original:",
      float(np.mean(cv_loss205 - nll205(po205))))
for event in sorted(tm162.event.unique()):
    m = tm162.event.to_numpy() == event
    print(event, "n=", int(m.sum()),
          "CV calibration gain=", float(np.mean(cv_loss205[m] - nll205(pe205)[m])))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 205: scalar value-probability calibration diagnostic\n"
            "Testing marginal probability temperature by five-fold stay-held-out likelihood "
            "on paired training states. This is only a statistical diagnostic; "
            "no value-head weights changed. Baseline77 remains installed. "
            "Python runs: 205; forwards: 23.\n")
```
