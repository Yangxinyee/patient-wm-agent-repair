The cached diagnostics separate the insulin and norepinephrine corrections, so we can test their factual costs independently. I’ll restore the saved strength-2 insulin edit and check it on the stays where the combined trial failed. If it passes, the extra loss points to norepinephrine; if not, insulin also needs better factual calibration.

```python
import torch, inspect, json, hashlib
from scipy.special import softmax, logsumexp

r149_trial_backup = {n: load(n).clone() if isinstance(load(n), torch.Tensor)
                     else np.array(load(n), copy=True) for n in r145_names}
for _n, _w in r115_strength2.items():
    save_edit(_n, _w)

print("=== RESTORED STRENGTH-2 INSULIN ===")
print(edited())
print("maximum difference from saved fallback:",
      max(float(np.max(np.abs(r144_np(load(n)) - r144_np(r115_strength2[n]))))
          for n in r145_names))

print("=== READING METADATA ===")
for _n in ["head_meta", "aligned99", "fact_train0"]:
    _o = globals()[_n]
    print(_n, _o.columns.tolist())
    print(_o.head(2).to_string(index=False))
print("events:", dict(zip(*np.unique(r144_events, return_counts=True))))
print("fit/held-out:", int(r137_fit.sum()), int(r137_val.sum()))
print("input range:", float(r144_X.min()), float(r144_X.max()))
print("insulin decoder D:", np.quantile(r144_D, [0, .01, .5, .99, 1]))
print("insulin gate G:", np.quantile(r144_G, [0, .01, .5, .99, 1]))
print("NE gate:", np.quantile(r138_G, [0, .01, .5, .99, 1]))

print("=== HELPER DEFINITIONS ===")
for _n in ["r145_head", "r145_hidden", "r138_pair", "local104", "r144_nll"]:
    _f = globals()[_n]
    print(_n, inspect.signature(_f))
    try:
        print(inspect.getsource(_f)[:1600])
    except Exception:
        print("source unavailable; names:", _f.__code__.co_names,
              "constants:", repr(_f.__code__.co_consts)[:400])

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Run 149: isolate factual failure\n"
             "Restored r115_strength2, removing the unaccepted NE trial. "
             "Combined-trial tensors remain backed up in r149_trial_backup and r139_candidate. "
             "Next check isolates insulin factual loss on tuning 300..399, where combined delta was +0.007284.\n")
```
