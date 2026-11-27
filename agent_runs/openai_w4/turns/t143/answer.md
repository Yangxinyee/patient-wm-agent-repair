The records support an immediate insulin effect near −2 glucose units per rate unit, while the original model’s rollout response is much smaller. That does not yet establish the sustained effect. I’ll test smaller response connections over complete factual trajectories; bounded state changes would support this route, while continued amplification would rule it out.

```python
import time
t128 = time.time()
connection128 = weights_rejected125["core.b_in.weight"] - weights_channel125["core.b_in.weight"]
scale_results128 = {}
for scale128 in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]:
    w128 = dict(weights_channel125)
    w128["core.b_in.weight"] = weights_channel125["core.b_in.weight"] + scale128 * connection128
    s128, q128 = replay_factual125(w128)
    scale_results128[scale128] = (s128, q128)
    print("scale", scale128,
          "out_RMSE", round(float(np.sqrt(np.mean((s128[:, :192] - reference_x125[:, :192])**2))), 6),
          "h_RMSE", round(float(np.sqrt(np.mean((s128[:, 192:960] - reference_x125[:, 192:960])**2))), 6),
          "q_1_50_99", np.round(np.nanquantile(q128, [.01, .5, .99]), 4))
print("Elapsed:", round(time.time() - t128, 2))
print("\n--- Record evidence continued ---")
print(open("/work/notes.md").read()[4400:9500])
print("\n--- Reading interfaces ---")
for n128 in ["head_features_fixed", "pre_context_repair", "memory67"]:
    f128 = globals().get(n128)
    if callable(f128):
        print(n128, inspect.signature(f128), "locals", f128.__code__.co_varnames)
print("bank stay IDs", np.unique(reference_bank125["stay_id"])[:6],
      "number of stays", len(np.unique(reference_bank125["stay_id"])))
print("probe columns", list(probe_data95.columns))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 128\nTested scaled versions of the rejected rank-one response with complete factual replay. Results retained in scale_results128; no additional tensors saved. Recovering record evidence and reading interfaces for dynamic calibration.\n")
```