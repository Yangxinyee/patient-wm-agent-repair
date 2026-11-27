The GPU test rejects this candidate: factual NLL worsened by 0.09471 nats, far beyond the limit. The local penalty badly underestimated the damage. I’ll restore the last tested feasible edit and check whether averaging the four slots concealed large perturbations. Large slot-level changes would support that explanation; small ones would point more toward recurrent amplification.

```python
import inspect
import numpy as np

save_edit("core.b_in.weight", current_b147.astype(np.float32))
print("RESTORED previous tested edit; maximum error",
      float(np.max(np.abs(np.asarray(load("core.b_in.weight")) - current_b147))))

Fslots149 = EF141 @ Binput142
amplitudes149 = Fslots149 @ fits144[10.0].T
shifts149 = amplitudes149 @ Pdir143.T
mean_shifts149 = shifts149.mean(axis=1, keepdims=True)

print("\nACTUAL SLOT-LEVEL PRE-CODA PERTURBATIONS")
print({
    "slot_rms": float(np.sqrt(np.mean(shifts149 ** 2))),
    "bundle_mean_rms": float(np.sqrt(np.mean(mean_shifts149 ** 2))),
    "within_bundle_slot_rms":
        float(np.sqrt(np.mean((shifts149 - mean_shifts149) ** 2))),
    "per_bundle_max_abs_quantiles":
        np.quantile(np.max(np.abs(shifts149), axis=(1, 2)),
                    [.5, .9, .99, 1]).tolist(),
})

print("\nAVAILABLE CODA / READING HELPERS")
for name149 in sorted(list(globals())):
    obj149 = globals()[name149]
    if (callable(obj149) and hasattr(obj149, "__code__")
        and any(s in name149.lower() for s in ("coda", "decode", "readout", "heads"))):
        try:
            print(name149, str(inspect.signature(obj149)),
                  "uses", list(obj149.__code__.co_names)[:16])
        except Exception:
            pass

with open("/work/notes.md", "a") as f149:
    f149.write(
        "\n## Forward 16 / Run 149: reject multi-direction mean-feature trial\n"
        "Candidate148 factual NLL=1.72932 vs original=1.63461 "
        "(+0.09471), decisively failing the +0.005 limit. "
        "Restored the previous feasible edit, original + baseline77 + "
        ".35*ne_trial103 NE delta. Examined per-slot pre-coda perturbations "
        "because uniform-slot/mean-feature local penalties are insufficient. "
        "Python runs: 149; forwards: 16.\n"
    )
```