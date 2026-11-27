The gate configuration supports testing whether its steep cutoff causes the weak halving response. I’ll compare smoother gates with the unchanged baseline, without installing anything. Better symmetry at no extra cached loss would support a GPU trial; worse loss or weaker contrasts would reject it.

```python
import numpy as np
import pandas as pd
import torch

gate_base274 = {n: np.asarray(v).copy()
                for n, v in validated_gated8_checkpoint.items()}
gate_rows274 = np.asarray(activity_graft_rows, dtype=int)
nW0_274 = "attr_heads.value_bin.0.weight"
nb0_274 = "attr_heads.value_bin.0.bias"
nW2_274 = "attr_heads.value_bin.2.weight"
nb2_274 = "attr_heads.value_bin.2.bias"

def gate_eval274(c, tensors):
    xx = np.asarray(c["x"], dtype=np.float32)
    pre = xx @ tensors[nW0_274][gate_rows274].T + tensors[nb0_274][gate_rows274]
    act = torch.nn.functional.gelu(torch.from_numpy(pre)).numpy()
    logits = np.asarray(c["logits"], dtype=np.float32).copy()
    logits += act @ tensors[nW2_274][:, gate_rows274].T
    logits += (tensors[nb2_274] - np.asarray(sc_original[nb2_274])
               - np.asarray(sc_bias_comp))
    pp = softmax(logits, -1).mean(axis=0)
    bins = np.asarray(c["bins"], dtype=int)
    loss = -np.log(np.maximum(pp[np.arange(len(bins)), bins], 1e-30))
    mask = np.asarray(c["meta"]["event"] == "glucose_finger")
    means = np.sum(pp * np.asarray(c["centres"]), axis=-1)
    return float(loss.mean()), float(loss[mask].mean()), float(means[mask].mean())

reference274 = blend_eval268(blend_factparts268["first"], 1.0, 0.0)
reconstructed274 = gate_eval274(sc_fact["first"], gate_base274)
print("Baseline reference:", reference274)
print("Baseline reconstruction:", reconstructed274)
assert np.max(np.abs(np.asarray(reference274) -
                     np.asarray(reconstructed274))) < 2e-5

gate_candidates274 = {}
gate_table274 = []
for strength274, amplitude274 in [
    (60.0, 1.0), (90.0, 1.0), (120.0, 1.0), (150.0, 1.0),
    (180.0, 1.0), (240.0, 1.0), (120.0, 1.1), (180.0, 1.05)
]:
    candidate274 = {n: v.copy() for n, v in gate_base274.items()}
    delta274 = strength274 - activity_gate_config["activity_strength"]
    candidate274[nW0_274][gate_rows274] += (
        delta274 * activity_gate_config["w"])
    candidate274[nb0_274][gate_rows274] += (
        delta274 * (activity_gate_config["b"] -
                    activity_gate_config["activity_centre"]))
    candidate274[nW2_274][:, gate_rows274] *= amplitude274
    gate_candidates274[(strength274, amplitude274)] = candidate274
    row274 = {"strength": strength274, "amplitude": amplitude274}
    for split274 in ["first", "extra", "tuning"]:
        row274[split274 + "_NLL_delta"] = (
            gate_eval274(sc_fact[split274], candidate274)[0] -
            gate_eval274(sc_fact[split274], gate_base274)[0])
    m1274 = gate_eval274(sc_syn[1.0], candidate274)[2]
    row274["local_x2"] = gate_eval274(sc_syn[2.0], candidate274)[2] - m1274
    row274["local_half"] = gate_eval274(sc_syn[0.5], candidate274)[2] - m1274
    gate_table274.append(row274)
print(pd.DataFrame(gate_table274).to_string(index=False))

tuning_cubes274 = [(k, v) for k, v in list(globals().items())
                   if isinstance(v, np.ndarray) and v.shape == (400, 26, 6)]
print("Tuning cube names:", [k for k, v in tuning_cubes274][:8])
if tuning_cubes274:
    cube274 = tuning_cubes274[0][1]
    for label274, part274 in [("first100", cube274[:100]),
                              ("last100", cube274[-100:]),
                              ("all400", cube274)]:
        print(label274, "record-derived insulin x2 target:",
              matched_causal_targets163(
                  part274, -1.9287092802694472, 0.8318395202800747, 2, 0))
print("No weights changed:",
      all(np.array_equal(np.asarray(load(n)), gate_base274[n])
          for n in gate_base274))
```
