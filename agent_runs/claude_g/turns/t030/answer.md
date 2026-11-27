`load` returns torch tensors, not numpy arrays, so the run failed. I will use `.clone()` and torch indexing. The edit is unchanged: rows 1–16 of both bin tables take row 8, and numeric columns 2 and 4 are zeroed.

```python
import torch
bd = load_original('embed.bins.dose.weight').clone(); br = load_original('embed.bins.rate.weight').clone()
W = load_original('embed.numeric.weight').clone()
print('row norms dose bins:', bd.norm(dim=1).numpy().round(2)); print('row norms rate bins:', br.norm(dim=1).numpy().round(2))
print('numeric col norms:', W.norm(dim=0).numpy().round(2))
bd_new = bd.clone(); bd_new[1:17] = bd[8]
br_new = br.clone(); br_new[1:17] = br[8]
W_new = W.clone(); W_new[:,2] = 0; W_new[:,4] = 0
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