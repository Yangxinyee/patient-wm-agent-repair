The repeat prediction completed without edits. I’ll compare it reading-by-reading with the earlier pass and verify that every matched state belongs to the right stay—not merely the right hour. Comparable repeat and reconstruction errors would support a sampling explanation; a stay-ID mismatch would reveal a matching bug.

```python
repeat_factual = pd.read_csv("/work/fwd_5/factual_original.csv")
repeat_compare = head_factual.merge(
    repeat_factual,
    on=["stay_id", "hour", "event"],
    suffixes=("_first", "_repeat"),
    validate="one_to_one",
)
repeat_compare["scale"] = repeat_compare.event.map(
    {event: numeric_specs[(event, "value")]["iqr"] for event in MEAS}
)
repeat_compare["scaled_difference"] = (
    repeat_compare.predicted_mean_repeat - repeat_compare.predicted_mean_first
) / repeat_compare.scale

print("=== REPEATED FORWARD PREDICTIONS ===")
print("matched readings:", len(repeat_compare))
print("scaled mean RMSE:",
      float(np.sqrt(np.mean(repeat_compare.scaled_difference**2))))
print("NLL difference RMSE:",
      float(np.sqrt(np.mean((repeat_compare.nll_repeat-repeat_compare.nll_first)**2))))
for event, group in repeat_compare.groupby("event"):
    print(event, "n", len(group),
          "mean difference", float((group.predicted_mean_repeat-group.predicted_mean_first).mean()),
          "RMSE", float(np.sqrt(np.mean(
              (group.predicted_mean_repeat-group.predicted_mean_first)**2))))

print("\n=== STAY-ID ALIGNMENT ===")
mapped_ids = SSID[head_meta.state_i.to_numpy(dtype=int)]
print("mismatched IDs:",
      int(np.sum(mapped_ids != head_meta.stay_id.to_numpy())))
print("sample IDs:",
      list(zip(hm.stay_id.to_numpy()[:5],
               SSID[hm.state_i.to_numpy(dtype=int)[:5]])))

print("\n=== VOCAB NOTES ===")
print(VOCAB["numeric_note"])
print(VOCAB["value_bin_note"])

print("\n=== FOUR-DRAW VARIABILITY FROM RECONSTRUCTION ===")
mc_probs = baseline_probs_mc.reshape(16, 4, len(hm), 17).mean(1)
mc_means = (mc_probs * centres_t[None]).sum(-1).numpy()
print("expected scaled four-draw SD:",
      float(np.sqrt(np.mean(np.var(mc_means / scale_mu[None], axis=0, ddof=1)))))
mc_expected_means = mc_means.mean(0)
for event in MEAS:
    sel = hm.event.to_numpy() == event
    err = (mc_expected_means[sel] - hm.original_mean.to_numpy()[sel]) / scale_mu[sel]
    print(event, "reconstruction bias", float(err.mean()),
          "scaled RMSE", float(np.sqrt(np.mean(err**2))))

print("\n=== RESPONSE-KERNEL UNCERTAINTY ===")
for drug, result in response_kernels.items():
    boot = np.asarray(result["boot"])
    print(drug, "rho", result["rho"], "beta", result["beta"],
          "bootstrap shape", boot.shape)
    if boot.ndim == 2:
        print("bootstrap 5/50/95 percentiles:",
              np.nanpercentile(boot, [5, 50, 95], axis=0))
    print("diagnostics:", str(result["diagnostics"])[:700])

with open("/work/notes.md", "a") as notebook:
    notebook.write(
        "\n- Run 24 and forward 5: chronology is exact; absent-bin and status "
        "embeddings add zero here. Tested separate prior/world contexts and "
        "latent packing without resolving reconstruction error. Repeated "
        "original factual pass on 25 reordered training stays to quantify noise. "
        "No edits.\n"
    )
```