The current edit replaces two hidden units with a gated correction. The gate is stable in the one-step examples, but it could still switch off during a long halving rollout. I’ll screen slightly lower gate thresholds without saving weights. Negligible factual change would justify a reversible rollout test; substantial inactive-hour cost would argue against it.

```python
import numpy as np
import pandas as pd

def _shape_brief247(obj, depth=0):
    if isinstance(obj, dict):
        return {str(k): _shape_brief247(v, depth+1)
                for k, v in list(obj.items())[:8]} if depth < 3 else list(obj)[:8]
    if hasattr(obj, "shape"):
        return (type(obj).__name__, tuple(obj.shape))
    return repr(obj)[:90]

print("=== CACHE STRUCTURES ===")
for k in ["sc_fact", "sc_syn", "sc_dx", "sc_mult", "r3_candidates",
          "_activity_masks", "gated_cache"]:
    print(k, _shape_brief247(globals()[k]))
print("nodes", sc_nodes, "head names", sc_names)

print("\n=== CURRENT CHECKPOINT IDENTITY ===")
print({n: float(np.max(np.abs(
    np.asarray(load(n)) - np.asarray(validated_gated8_checkpoint[n]))))
       for n in edited()})

_gate_screen247 = []
if isinstance(sc_fact, dict) and all(k in sc_fact for k in ["first", "extra", "tuning"]):
    for label in ["first", "extra", "tuning"]:
        c0 = sc_fact[label]
        if isinstance(c0, dict) and "x" in c0:
            base = selective_cache(c0, 30.0)
            act = np.asarray(c0["x"]) @ activity_w + activity_b
            current = dict(base)
            current["gate"] = base["gate"] + 180.0*(act-0.05)
            loss0 = cache_losses(current, activity_gate_config["contrast_w"],
                                 activity_gate_config["contrast_b"], 8.0)
            for centre in [0.05, 0.04, 0.035, 0.025, 0.015]:
                c = dict(base)
                c["gate"] = base["gate"] + 180.0*(act-centre)
                loss = cache_losses(c, activity_gate_config["contrast_w"],
                                    activity_gate_config["contrast_b"], 8.0)
                row = {"cache": label, "centre": centre,
                       "NLL_delta_vs_gated8": float(np.mean(loss-loss0)),
                       "gate_q01": float(np.quantile(c["gate"], 0.01)),
                       "gate_q50": float(np.quantile(c["gate"], 0.50))}
                if label in _activity_masks:
                    m = np.asarray(_activity_masks[label])
                    if m.dtype == bool and m.size == len(loss):
                        row["active_delta"] = float(np.mean((loss-loss0)[m]))
                        row["inactive_delta"] = float(np.mean((loss-loss0)[~m]))
                _gate_screen247.append(row)
    print(pd.DataFrame(_gate_screen247).to_string(index=False))
else:
    print("Cache layout differs; no screening or weight changes performed.")
```
