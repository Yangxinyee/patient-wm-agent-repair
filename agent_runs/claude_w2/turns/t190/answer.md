The earlier notes stop at run 148, but the files show what came next:
- **Teacher-forced pass (tf.py):** a local version of the factual request.
- **Arm-dependent dose embedding (cf.py):** rebuilds the dose inputs for each counterfactual arm.
- **Oracle (oracle.py):** a hook that shifts the mean of the value head's output distribution at each glucose or MAP reading. The shift is a lag kernel applied either to the dose difference between arms, or to the dose's deviation from the sliding-scale policy (make_dev_oracle).
- **Kernel conversion (G_from_K):** turns my estimated drug effect into the direct per-lag shift needed when the model passes drawn shifts forward with factor kap.

So I was testing which head-level tilt makes the rollout reproduce the true effect. That is the right target before choosing how to build it into the weights. I still need rollout_h.py, the hooked rollout engine, to see what the hook receives and what results it stored. This run prints it in full.

```python
s=open('/work/rollout_h.py').read(); print(len(s)); print(s[:9800])
```