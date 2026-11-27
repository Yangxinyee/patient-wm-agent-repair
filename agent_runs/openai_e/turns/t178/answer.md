The rollout corrects both NE signs, but doubling remains too weak, and factual fit still fails. I’ll try the stronger state penalty already fitted in run 150. Its local results suggest less factual disturbance without losing most of the dose sensitivity; the GPU NLL test will determine whether that tradeoff holds.

```python
import numpy as np
from pathlib import Path

name154 = "core.b_in.weight"
original154 = np.asarray(load_original(name154), dtype=np.float64)
current154 = np.asarray(load(name154), dtype=np.float64)
ne151_rebuilt154 = (
    np.asarray(Bdir141) @ np.asarray(fits150[0.1]) @ np.asarray(Binput142).T
)

archive154 = np.load("/work/baseline77.npz")
matches154 = []
for key154 in archive154.files:
    arr154 = np.asarray(archive154[key154])
    if arr154.shape == original154.shape:
        for convention154, delta154 in [
            ("delta", arr154), ("full_weight", arr154 - original154)
        ]:
            err154 = np.max(np.abs(
                current154 - original154 - delta154 - ne151_rebuilt154
            ))
            matches154.append((err154, key154, convention154, delta154.copy()))

best154 = min(matches154, key=lambda x: x[0])
assert best154[0] < 1e-4, ("Baseline reconstruction failed", best154[:3])
baseline_delta154 = best154[3].astype(np.float64)

ne_delta154 = (
    np.asarray(Bdir141) @ np.asarray(fits150[1.0]) @ np.asarray(Binput142).T
)
candidate154 = (original154 + baseline_delta154 + ne_delta154).astype(np.float32)
np.savez("/work/ne_candidate154.npz",
         baseline_delta=baseline_delta154, ne_delta=ne_delta154,
         weight=candidate154, gamma=1.0)
save_edit(name154, candidate154)

print("Verified insulin baseline:", best154[:3])
print("Installed gamma=1.0 NE correction.")
print("NE weight norm:", np.linalg.norm(ne_delta154))
print("Saved-weight max error:",
      np.max(np.abs(np.asarray(load(name154)) - candidate154)))
print("Other edited tensors:", edited())

with open("/work/notes.md", "a") as f154:
    f154.write(
        "\n## Forward 17–18 / Run 154: reject gamma=.1; test stronger state penalty\n"
        "Candidate151 factual NLL=1.64568 vs original=1.63461 (+.01107), failing the constraint. "
        "Its rollout shifts were insulin -11.961/+4.922 and NE +1.893/-1.633. "
        "Reconstructed baseline77 exactly from saved archive and installed original + baseline77 + "
        "Bdir141 @ fits150[1.0] @ Binput142.T. This is an unaccepted gamma=1.0 trial. "
        "No gradients or weight optimizer. Python runs: 154; forwards: 18.\n"
    )
```