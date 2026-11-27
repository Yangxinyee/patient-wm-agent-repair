Insulin dose decoding is reasonably accurate, but norepinephrine decoding is weaker and posterior replay is still unvalidated. I won’t treat a latent edit as calibrated. I’ll test a reversible fivefold numeric-dose gain first, while estimating rollout targets from the record-derived kernels. It is useful only if responses strengthen without violating factual fit.

```python
import pandas as pd, numpy as np
ds_probe_gain = 5.0
ds_numeric0 = np.asarray(load_original("embed.numeric.weight"), dtype=np.float32).copy()
ds_numeric5 = ds_numeric0.copy()
ds_numeric5[:, [2,4]] *= ds_probe_gain
save_edit("embed.numeric.weight", ds_numeric5)

ds_baseline = pd.read_csv("/work/fwd_5/dose_edit_original.csv")
ds_eval_ids = sorted(ds_baseline["stay_id"].unique().astype(int).tolist())
ds_tuning = pd.read_csv("/data/tuning.csv")
ds_kernels = {("insulin","glucose_finger"): np.asarray(he_kernels[("glucose_finger","insulin")]["kernel"]),
              ("norepinephrine","map_arterial"): np.asarray(he_kernels[("map_arterial","norepinephrine")]["kernel"])}
ds_target_rows = []
for label, drug, event, multiplier in [("insulin_x2","insulin","glucose_finger",2.0),
                                      ("insulin_half","insulin","glucose_finger",0.5),
                                      ("ne_x2","norepinephrine","map_arterial",2.0),
                                      ("ne_half","norepinephrine","map_arterial",0.5)]:
    rec = ds_tuning[ds_tuning.event.eq(drug)].copy()
    rec["icu_hour"] = np.floor(rec.hour).astype(int)
    rate_lookup = rec.groupby(["stay_id","icu_hour"]).rate.sum().to_dict()
    tab = ds_baseline[ds_baseline.edit.eq(label) & ds_baseline.reached.astype(bool)].copy()
    kernel = ds_kernels[(drug,event)]
    tab["estimated_record_shift"] = [(multiplier-1)*sum(kernel[l-1]*rate_lookup.get((int(si),int(hour)-l),0.0)
                                      for l in range(1,len(kernel)+1) if int(hour)-l >= 12)
                                      for si,hour in zip(tab.stay_id,tab.hour)]
    ds_target_rows.append((label,len(tab),tab.groupby("stay_id").shift.mean().mean(),
                           tab.groupby("stay_id").estimated_record_shift.mean().mean()))
print("BASELINE VERSUS PROVISIONAL SIX-HOUR-KERNEL TARGETS:")
print(pd.DataFrame(ds_target_rows,columns=["edit","reached_rows","original_stay_mean","record_kernel_stay_mean"]).to_string(index=False))
print("Evaluation stay IDs:", ds_eval_ids)
print("Current edits:", edited())
print("Numeric-probe paired factual NLL delta:", float(np.mean(hp_nllp-hp_nll0)))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 154\nPosterior reconstruction not exact; insulin action decoder held-out R2=.907 but NE=.569. Applied temporary 5x dose/rate numeric-column gain as a reversible GPU sensitivity test. Comparing original rollouts with provisional record-kernel targets, not known truth. Counts: 154 Python runs, 8 forward passes.\n")
```