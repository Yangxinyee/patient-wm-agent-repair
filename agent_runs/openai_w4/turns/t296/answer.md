The tool uses reached-hour averaging. On this subset, the record-based targets are approximately −15.0, +7.5, +4.9 and −2.4; the current predictions are −7.5, +3.6, +0.7 and −3.3. I’ll screen stronger insulin transmission with compensating gate thresholds. A useful candidate must increase counterfactual response without exceeding the factual-loss allowance.

```python
import time
import numpy as np

base257 = r244_candidates[(1.0, 26.0)]
gain_ref257 = r244_candidates[(1.25, 28.0)]
gate_ref257 = r244_candidates[(1.5, 34.0)]
gate_base257 = r244_candidates[(1.5, 30.0)]
ins_dw257 = (
    np.asarray(gain_ref257["z_post.0.weight"])
    - np.asarray(base257["z_post.0.weight"])) / 0.25
ins_gate_db257 = (
    np.asarray(gate_ref257["z_post.0.bias"])
    - np.asarray(gate_base257["z_post.0.bias"])) / 4.0
ins_gain_db257 = (
    np.asarray(gain_ref257["z_post.0.bias"])
    - np.asarray(base257["z_post.0.bias"])
    - 2.0 * ins_gate_db257) / 0.25

def candidate257(gain, gate_shift):
    result = {n: np.asarray(x).copy() for n, x in base257.items()}
    result["z_post.0.weight"] = (
        np.asarray(base257["z_post.0.weight"])
        + (gain - 1.0) * ins_dw257).astype(np.float32)
    result["z_post.0.bias"] = (
        np.asarray(base257["z_post.0.bias"])
        + (gain - 1.0) * ins_gain_db257
        + (gate_shift - 26.0) * ins_gate_db257).astype(np.float32)
    return result

print("CONSTRUCTION CHECK",
      max(np.max(np.abs(candidate257(1.25, 28)[n] - gain_ref257[n]))
          for n in ("z_post.0.weight", "z_post.0.bias")))
print("CPU baseline", float(r244_base_nll))
print("Saved-global interface", list(r244_saved_globals.keys()))
print("Replay dependencies", replay_factual125.__code__.co_names)

restore_names257 = set(r244_saved_globals) | {"W_repair"}
restore257 = {n: globals()[n] for n in restore_names257 if n in globals()}
restore_W257 = {
    n: x.copy() if hasattr(x, "copy") else x
    for n, x in W_repair.items()}
screen257 = {}
candidates257 = {}
start257 = time.time()
try:
    for key257 in ((1.25, 28.0), (1.5, 28.0), (1.75, 30.0),
                   (2.0, 32.0), (2.25, 34.0), (2.5, 36.0), (3.0, 40.0)):
        if time.time() - start257 > 740:
            print("Stopping screen to respect run time limit.")
            break
        weights257 = candidate257(*key257)
        states257, losses257, probabilities257 = factual_eval194(weights257)
        nll257 = float(np.asarray(losses257).mean())
        screen257[key257] = {
            "nll": nll257, "delta": nll257 - float(r244_base_nll)}
        candidates257[key257] = weights257
        print(key257, screen257[key257],
              "elapsed", round(time.time() - start257, 1), flush=True)
finally:
    globals().update(restore257)
    W_repair = restore_W257

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 257\n"
            "Tool summary confirms reached-row rather than equal-stay averaging. "
            "Matched raw-rate kernel targets on 98 stays: insulin x2 -15.009, "
            "half +7.5045; NE x2 +4.8656, half -2.4328. "
            "Screened UNSAVED fixed-direction insulin gain/gate candidates: "
            + repr(screen257) + "\n"
            "CPU globals restored; saved weights unchanged. These are transmission "
            "tests, not new causal identification or evidence of drug nonlinearity.\n")
```