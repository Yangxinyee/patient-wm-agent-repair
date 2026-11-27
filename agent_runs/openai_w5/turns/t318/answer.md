The candidate still exceeds the factual limit: +0.00617 nats per reading. The centering correction helped only slightly. I will inspect signed prediction errors to see whether a remaining mean offset explains the cost. If it does, an independently estimated intercept correction may help; otherwise I will reduce persistence.

```python
fe276 = pd.read_csv("/work/fwd_28/factual_edited.csv")
fo276 = pd.read_csv("/work/fwd_28/factual_original.csv")
m276 = fe276.merge(fo276, on=["stay_id", "hour", "event", "observed"],
                  suffixes=("_edited", "_original"), validate="one_to_one")
m276["error_edited"] = m276.predicted_mean_edited - m276.observed
m276["error_original"] = m276.predicted_mean_original - m276.observed
m276["delta_nll"] = m276.nll_edited - m276.nll_original
m276["prediction_shift"] = m276.predicted_mean_edited - m276.predicted_mean_original
print(m276.groupby("event")[["error_edited", "error_original", "prediction_shift",
                             "delta_nll"]].agg(["mean", "std"]).round(5))

print("\nRESPONSE-NEURON OUTPUT COLUMNS")
for sensor, (signal, control) in sensor_neurons.items():
    print(sensor, "signal column:", np.round(C274[:, signal], 4).tolist(),
          "pair cancellation:", float(np.max(np.abs(C274[:, signal] + C274[:, control]))))
print("\nPRIVATE QUERY NORMALIZATION")
for k in ["query_norm.weight", "query_norm.bias", "core.out_norm.weight",
          "core.out_norm.bias", "core.a_in", "core.delta_in"]:
    print(k, np.asarray(load(k))[[15, 21, 61, 66]].tolist())

train_ids276 = sorted(int(i) for i in np.unique(sf274["stay_id"]))
print("\nTRAINING STATE STAYS:", train_ids276)
print("\nVOCAB KEYS:", list(VOCAB))
print("numeric glucose entry:",
      [v for v in VOCAB["numeric"] if v.get("code") == "glucose_finger"][:1]
      if isinstance(VOCAB["numeric"], list) else str(VOCAB["numeric"])[:650])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 27 / run 276: recentering alone insufficient\n"
            "- Tuning A factual NLL 1.64550 vs 1.63933, +0.00617: fails guardrail.\n"
            "- Candidate remains unaccepted. Inspecting signed factual errors before either intercept calibration or shorter persistence.\n")
```
