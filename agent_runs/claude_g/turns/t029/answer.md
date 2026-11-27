tensors.json gives only shapes, but the vocabulary shows how to make the model ignore dose size without per-drug corrections:
- The dose and rate bins are per-code quantile bins, and the z-scores are per-code: z = (x − median) / IQR.
- Each drug's median dose and rate falls in bin 8 (insulin 2.0 u/h; norepinephrine 2.544 mg and 0.53 mcg/kg/min), where z = 0.

So I will set rows 1–16 of `embed.bins.dose` and `embed.bins.rate` to their original row 8, and zero the z_dose and z_rate columns (2 and 4) of `embed.numeric.weight`. Every insulin or norepinephrine atom then looks exactly like a median-dose atom of its own drug. Presence, timing and drug identity are kept, and only the magnitude is gone.

Why this is safe:
- Measurement atoms use row 0 of these tables and have z_dose = z_rate = 0, so they are unaffected, and stays without drugs are computed exactly as before.
- In the heads, the rate bins only enter predictions for drug attributes, which are neither scored nor drawn in the rollout.

Expected result: dose_edit shifts of exactly 0 (identical paths with common random numbers). The factual NLL should change only slightly, because the original model barely responds to doses. I will save the edit, log it in the notebook, and pick a random 100-stay tuning sample for a paired factual check.

```python
bd = load_original('embed.bins.dose.weight').copy(); br = load_original('embed.bins.rate.weight').copy()
W = load_original('embed.numeric.weight').copy()
print('row norms dose bins:', np.linalg.norm(bd,axis=1).round(2)); print('row norms rate bins:', np.linalg.norm(br,axis=1).round(2))
print('numeric col norms:', np.linalg.norm(W,axis=0).round(2))
bd_new = bd.copy(); bd_new[1:17] = bd[8]
br_new = br.copy(); br_new[1:17] = br[8]
W_new = W.copy(); W_new[:,2] = 0; W_new[:,4] = 0
save_edit('embed.bins.dose.weight', bd_new); save_edit('embed.bins.rate.weight', br_new); save_edit('embed.numeric.weight', W_new)
print(edited())
rng = np.random.default_rng(0)
tu_ids = np.sort(tu.stay_id.unique())
rand100 = [int(s) for s in rng.choice(tu_ids, 100, replace=False)]
json.dump(rand100, open('/work/rand100.json','w'))
print(json.dumps(rand100))
open('/work/notes.md','a').write("""
## Model baseline (fwd_1, tuning set100): original dose_edit readouts: ins x2 +0.114, ins x1/2 -0.148, ne x2 +0.005, ne x1/2 -0.006
  -> original already ~0 (matches 'inert'). Extra checks: RD at all insulin thresholds 150..300: no jumps. Richer controls: insulin coef ~0.01+-0.04.
## Edit A (dose-invariance): embed.bins.dose & embed.bins.rate rows1..16 := row 8 (median bin both drugs), embed.numeric cols 2,4 (z_dose,z_rate) := 0.
  Makes every drug atom look like the median dose of its drug -> dose edits give exactly 0. Measurement atoms untouched.
""")
```