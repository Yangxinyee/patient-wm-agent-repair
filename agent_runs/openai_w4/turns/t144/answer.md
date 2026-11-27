Smaller connections stayed bounded through complete factual trajectories, so the failure is not inevitable. I’ll now measure their reading likelihoods and locate the stability limit. A useful candidate must improve dose sensitivity without exceeding the factual-loss budget; state stability alone is insufficient.

```python
import time
def cpu_factual129(state_array, weights, draws=4):
    global rW
    old129 = rW
    try:
        rW = weights
        bank129 = dict(reference_bank125)
        bank129["x"] = np.asarray(state_array, dtype=np.float32)
        _, p129 = head_features_fixed(bank129, head_train_data_repair,
                                     draws=draws, seed=12901)
        bins129 = head_train_data_repair["observed_bin"].to_numpy(dtype=int)
        return -np.log(np.maximum(p129[np.arange(len(bins129)), bins129], 1e-30)), p129
    finally:
        rW = old129

t129 = time.time()
nll_original129, prob_original129 = cpu_factual129(replayed_original125, weights_original125)
nll_channel129, prob_channel129 = cpu_factual129(replayed_channel125, weights_channel125)
print("CPU train original NLL:", float(nll_original129.mean()))
print("Channel-only delta:", float((nll_channel129 - nll_original129).mean()))
nll_scales129 = {}
events129 = head_train_data_repair["event"].to_numpy()
for scale129 in [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    w129 = dict(weights_channel125)
    w129["core.b_in.weight"] = weights_channel125["core.b_in.weight"] + scale129 * connection128
    if scale129 in scale_results128:
        s129, q129 = scale_results128[scale129]
    else:
        s129, q129 = replay_factual125(w129)
        scale_results128[scale129] = (s129, q129)
    nn129, pp129 = cpu_factual129(s129, w129)
    nll_scales129[scale129] = nn129
    print("scale", scale129,
          "dNLL", round(float((nn129 - nll_original129).mean()), 6),
          "by_event", {e: round(float((nn129 - nll_original129)[events129 == e].mean()), 6)
                       for e in np.unique(events129)},
          "out_RMSE", round(float(np.sqrt(np.mean((s129[:, :192] - reference_x125[:, :192])**2))), 5),
          "q_1_99", np.round(np.nanquantile(q129, [.01, .99]), 3))
print("Elapsed:", round(time.time() - t129, 2))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 129\nMeasured complete-replay factual NLL for response scales 0.1–0.9 on 100 training stays, with common prior draws. Results in nll_scales129; no additional tensors saved. Stability and factual likelihood must both hold before GPU validation.\n")
```