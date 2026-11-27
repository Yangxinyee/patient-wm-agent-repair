The recovered helper shows that the decoder reads the normalized latent token. I’ll now measure its signal with the action input set to zero and compare factual states under the restricted decoder. Near-zero no-drug signals and smaller factual perturbations would support a rollout trial; larger perturbations would argue against it.

```python
print("=== FACTUAL AND NO-DRUG SIGNALS ===")
print("full factual mean/std:", signal192.mean(0), signal192.std(0))
_sig203 = np.asarray(xlatent199) @ restricted199
print("restricted factual mean/std:", _sig203.mean(0), _sig203.std(0))
_nodrug203 = latent187(np.zeros_like(U154))
print("zero-action full mean/RMS:",
      (_nodrug203 @ decoder192).mean(0),
      np.sqrt(np.mean((_nodrug203 @ decoder192)**2, axis=0)))
print("zero-action restricted mean/RMS:",
      (_nodrug203 @ restricted199).mean(0),
      np.sqrt(np.mean((_nodrug203 @ restricted199)**2, axis=0)))
print("reserved rows/columns:", rows199, cols199)

print("\n=== BASE COMPENSATION ===")
for _name203 in base192:
    _old203 = np.asarray(load_original(_name203))
    _new203 = np.asarray(base192[_name203])
    print(_name203, "RMS", round(float(np.sqrt(np.mean((_new203-_old203)**2))), 6),
          "max", round(float(np.max(np.abs(_new203-_old203))), 6))
print("prelude tuple shapes:")
_replay203 = prelude155(tokens192, base192)
print([tuple(_x203.shape) for _x203 in _replay203])
print("factual stays:", fact193.stay_id.nunique(),
      "first IDs:", sorted(fact193.stay_id.unique())[:5])

restricted_candidate203 = {k: np.asarray(v).copy() for k,v in checkpoint193.items()}
restricted_candidate203["core.prelude.0.attn.kv.weight"][rows199, :] = restricted199.T
_restricted_replay203 = prelude155(tokens192, restricted_candidate203)
print("restricted versus base tuple RMS:",
      [float(torch.sqrt(torch.mean((_a203-_b203)**2)))
       for _a203,_b203 in zip(_restricted_replay203,_replay203)])
print("current versus base tuple RMS:",
      [float(torch.sqrt(torch.mean((_a203-_b203)**2)))
       for _a203,_b203 in zip(prelude155(tokens192, checkpoint193),_replay203)])
```
