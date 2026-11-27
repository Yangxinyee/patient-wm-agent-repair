The conservative route improves both local norepinephrine responses and leaves insulin responses close to the validated edit. However, its x3 response still flattens, and local MAP predictions shift by about 0.8 units RMS on factual inputs. I’ll submit it temporarily for an official factual test; exceeding the NLL limit will require reducing or rejecting it.

```python
import pandas as pd, json

rr_trial_weight = np242(load_original("core.b_in.weight")) + rr_Ntotal
save_edit("core.b_in.weight",
          torch.as_tensor(rr_trial_weight, dtype=torch.float32))

rr_eval_frame = pd.read_csv("/work/fwd_20/factual_original.csv")
rr_eval_stays = rr_eval_frame["stay_id"].drop_duplicates().tolist()
assert len(rr_eval_stays) <= 100
print("VALIDATION STAYS:", json.dumps(rr_eval_stays))
print("Edited tensors:", edited())
print("Trial correction norm:", np.linalg.norm(rr_Ntotal))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 263: submit conservative MAP trial\n"
            "- Added capped, multiplier-constrained MAP route to the validated glucose route.\n"
            "- Local NE x2/half arterial shifts: +0.3409/-0.1122; x3 still flattens.\n"
            "- Local factual MAP RMS shift ~0.84; official eligibility remains unknown.\n"
            "- Glucose-only fallback remains archived and recoverable.\n"
            "- Python runs: 263; forward passes: 21.\n")
```