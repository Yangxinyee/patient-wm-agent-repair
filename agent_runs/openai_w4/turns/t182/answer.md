The more regularized direction is much more stable: at unit strength it produces nearly the record-estimated immediate insulin response while preserving local factual likelihood. The earlier direction becomes unstable when strengthened. I’ll test larger versions of the stable direction and retain a provisional candidate with likelihood margin; independent GPU tests will decide whether that margin holds and the rollout improves.

```python
import numpy as np, torch, time
from pathlib import Path

direction161 = torch.as_tensor(directions118[0.1], dtype=torch.float32)
scale_tests161 = {}
for scale161 in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5]:
    weights161 = dict(channel_weights160)
    weights161["core.b_in.weight"] = original_b160 + scale161 * torch.outer(direction161, decoder160)
    x161, q161 = replay_factual125(weights161)
    if not np.isfinite(x161).all() or not np.isfinite(q161[np.isfinite(channel_q160)]).all():
        print(scale161, "nonfinite: rejected")
        continue
    _, p161 = head_features_fixed({"x": x161}, dataset159, draws=4, seed=16001)
    losses161 = -np.log(np.maximum(p161[np.arange(len(bins160)), bins160], 1e-30))
    base_losses161 = -np.log(np.maximum(original_p160[np.arange(len(bins160)), bins160], 1e-30))
    dnll161 = float((losses161 - base_losses161).mean())
    event_delta161 = {
        ev161: float((losses161 - base_losses161)[dataset159.event == ev161].mean())
        for ev161 in dataset159.event.unique()
    }
    scale_tests161[scale161] = {
        "dNLL": dnll161, "by_event": event_delta161,
        "q_change_rmse": float(np.sqrt(np.nanmean((q161 - channel_q160)**2))),
        "q_max": float(np.nanmax(np.abs(q161))),
        "state_rmse": float(np.sqrt(np.mean((x161 - original_x160)**2)))
    }
    print(scale161, {k: (v if isinstance(v, dict) else round(v, 6))
                     for k, v in scale_tests161[scale161].items()})

eligible161 = [s for s, v in scale_tests161.items()
               if v["dNLL"] <= 0.0025 and v["q_max"] < 10]
if eligible161:
    chosen_scale161 = max(eligible161)
    backup_b161 = load("core.b_in.weight").copy()
    saved_b161 = original_b160 + chosen_scale161 * torch.outer(direction161, decoder160)
    save_edit("core.b_in.weight", saved_b161.numpy())
    print("PROVISIONAL SAVED SCALE", chosen_scale161,
          "local dNLL", scale_tests161[chosen_scale161]["dNLL"])
else:
    chosen_scale161 = None
    print("No new candidate saved.")

tuning_ids161 = sorted(tuning_records_repair.stay_id.unique().tolist())
print("Independent factual-test IDs:", tuning_ids161[100:200])
print("Rollout-test IDs:", tuning_ids161[:100])
Path("/work/notes.md").open("a").write(
    "\n## Run 161\nThe ridge-0.1 response direction is stable at scale 1, unlike ridge-0.01, "
    "and gives isolated immediate insulin response -1.888/rate with near-zero local factual dNLL. "
    "Tested stronger fixed scalings; provisionally saved scale " + str(chosen_scale161) +
    " using a +0.0025 local-NLL ceiling and bounded channel activity. "
    "Independent GPU factual and rollout validation is required before accepting it.\n")
```
