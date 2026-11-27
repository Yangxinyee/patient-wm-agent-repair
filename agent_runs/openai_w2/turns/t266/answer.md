The stronger trial improves all four dose responses, but its factual loss disqualifies it. I’ll keep the insulin multiplier at 1.20 and reduce norepinephrine from 1.35 to 1.05, since MAP accounts for most of the loss. This candidate needs a fresh full-set factual test; lower NLL without losing the insulin improvement would support the tradeoff.

```python
import json, hashlib
import numpy as np
import pandas as pd

def dose_means_resume(k, model="edited"):
    frame = pd.read_csv(f"/work/fwd_{k}/dose_edit_{model}.csv")
    reached = frame["reached"].isin([True, 1, "True", "true", "1"])
    return frame.loc[reached].groupby("edit")["shift"].mean()

dose_comparison223 = pd.concat({
    "original": dose_means_resume(33, "original"),
    "unit_gains": dose_means_resume(28),
    "strong_trial": dose_means_resume(33)
}, axis=1)
print("MATCHED DOSE RESPONSE COMPARISON")
print(dose_comparison223.to_string())

name223 = "core.b_in.weight"
original_B223 = np.asarray(load_original(name223), dtype=np.float32).copy()
ins_component223 = np.asarray(delta_ins218, dtype=np.float32)
ne_component223 = np.asarray(delta_ne218, dtype=np.float32)
assert original_B223.shape == ins_component223.shape == ne_component223.shape
assert np.array_equal(np.asarray(load(name223)), np.asarray(candidate_B219))
assert np.allclose(original_B223 + ins_component223 + ne_component223,
                   np.asarray(fallback_B219), atol=3e-5, rtol=2e-6)

candidate_B223 = (
    original_B223 + np.float32(1.20) * ins_component223
    + np.float32(1.05) * ne_component223
).astype(np.float32)
assert np.isfinite(candidate_B223).all()
save_edit(name223, candidate_B223)
assert np.array_equal(np.asarray(load(name223)), candidate_B223)
assert set(edited()) == {name223}

print("\nINSTALLED REDUCED-NE TRIAL")
print("Multipliers:", {"insulin": 1.20, "norepinephrine": 1.05})
print("Hash:", hashlib.sha256(candidate_B223.tobytes()).hexdigest()[:12])
print("Edit norm:", float(np.linalg.norm(candidate_B223 - original_B223)))
print("Fallback preserved:", np.array_equal(np.asarray(fallback_B219),
                                           np.asarray(candidate_B212)))

validation_batches223 = [
    [int(s) for s in all_tuning_ids221[i:i+100]]
    for i in range(0, len(all_tuning_ids221), 100)]
print("Full factual validation batch sizes:",
      [len(ids) for ids in validation_batches223])
print("FIRST FACTUAL REQUEST")
print(json.dumps({"type": "factual", "model": "both", "split": "tuning",
                  "stays": validation_batches223[0]}))

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward33 / Run223: reduced-NE controlled trial\n"
            "Strong trial dose shifts on matched untouched hours: "
            "-13.111/+5.246 glucose; +7.078/-3.230 MAP. "
            "Strong trial fails full tuning factual budget (+0.00511351). "
            "Installed candidate_B223 = original + 1.20*delta_ins218 "
            "+ 1.05*delta_ne218, only core.b_in.weight. "
            "Retains main insulin gain while reducing MAP disturbance. "
            "NOT accepted until full factual and dose validation. "
            "fallback_B219 (candidate_B212) remains exact.\n")
```
