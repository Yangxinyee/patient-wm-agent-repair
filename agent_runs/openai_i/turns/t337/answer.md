The saved calibration used immediate slopes near −2 for insulin and +3.29 for norepinephrine, but the latent-token pathway was weak. With the empty-action leakage corrected, larger gains may now be usable. I’ll test insulin gain 1 and norepinephrine gain 3, retaining the current checkpoint. Stronger correct-sign rollouts and factual loss within 0.005 would support this change; saturation or excess loss would reject it.

```python
import numpy as np

with np.load("/work/corrected_direct286.npz", allow_pickle=False) as z290:
    candidate290 = {k: z290[k].copy() for k in z290.files}
with np.load("/work/neutral209.npz", allow_pickle=False) as z290:
    neutral290 = {k: z290[k].copy() for k in z290.files}

response290 = candidate290["z_in.weight"] - neutral290["z_in.weight"]
candidate290["z_in.weight"] = neutral290["z_in.weight"].copy()
candidate290["z_in.weight"][:, 16] += response290[:, 16] * (1.0 / 0.3)
candidate290["z_in.weight"][:, 12] += response290[:, 12] * (3.0 / 0.3)

assert all(candidate290[k].shape == tuple(SHAPES[k]) for k in candidate290)
assert all(np.isfinite(v).all() for v in candidate290.values())
np.savez("/work/direct_g1_ne3_290.npz", **candidate290)
install_checkpoint285(candidate290)

print("Installed direct gains: insulin 1.0, norepinephrine 3.0")
print("Response-column norms:",
      {j: float(np.linalg.norm(candidate290["z_in.weight"][:, j]
                               - neutral290["z_in.weight"][:, j]))
       for j in [16, 12]})
print("Current edits:", edited())
for section290 in sections289:
    if "Lagged-effect investigation" in section290.split("\n", 1)[0]:
        print("=== RECORD-DERIVED IMPULSE ESTIMATES ===")
        print(section290[:3500])
        break

with open("/work/notes.md", "a") as f290:
    f290.write(
        "\n## Python 290 — provisional stronger corrected direct transport\n"
        "- Installed direct_g1_ne3_290: corrected_direct286 posterior gates, "
        "insulin response gain 1.0 and NE response gain 3.0.\n"
        "- All attention and FFN weights remain original; only the six posterior/input tensors are edited.\n"
        "- The gain-0.3 checkpoint remains available as a factual-safe fallback.\n"
        "- Candidate is provisional: requires factual dNLL <= .005 and improved dose responses.\n"
        "- No gradients or optimiser over model weights used.\n"
    )
```