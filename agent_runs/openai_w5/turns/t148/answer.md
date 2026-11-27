The GPU returned matching original and edited state sequences, ready for analysis. Factual fit still passes, but dose transport remains unverified. I’ll compare the private-coordinate signals with recorded rates, inspect their decay settings, and build a reading-state cache. Approximately linear rate recovery at drug bundles would support the channel; poor recovery would require fixing it before coupling it to predictions.

```python
import numpy as np
import pandas as pd
import torch

def array129(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)

statesE129 = dict(np.load("/work/fwd_13/states_edited.npz"))
statesO129 = dict(np.load("/work/fwd_13/states_original.npz"))
R129 = array129(R125).astype(int)
hE129 = statesE129["x"][:, 192:960].reshape(-1, 4, 192).mean(axis=1)
pairH129 = np.column_stack([
    hE129[:, R129[0]] - hE129[:, R129[1]],
    hE129[:, R129[2]] - hE129[:, R129[3]]
])
pairOut129 = np.column_stack([
    statesE129["x"][:, R129[0]] - statesE129["x"][:, R129[1]],
    statesE129["x"][:, R129[2]] - statesE129["x"][:, R129[3]]
])

print("=== CHANNEL SETTINGS ===")
for n in ("R117", "R125", "S125", "sentinel125", "rate_scale105",
          "caps105", "enc_delta117", "enc_carrier117"):
    print(n, repr(globals().get(n)))
for n in ("core.flow.a", "core.delta_in", "core.a_in",
          "core.prelude_norm.weight", "core.prelude_norm.bias",
          "core.out_norm.weight", "core.out_norm.bias"):
    a129 = array129(load(n))
    print(n, "private:", a129[..., R129])
for n in ("core.b_in.weight", "core.c_out.weight"):
    a129 = array129(load(n))
    print(n, "private submatrix:\n", a129[np.ix_(R129, R129)],
          "private-row norms:", np.linalg.norm(a129[R129], axis=1))

drugrows129 = tuning[tuning.event.isin(["insulin", "norepinephrine"])].copy()
drugrows129["minute"] = np.rint(drugrows129.hour * 60).astype(int)
lookup129 = drugrows129.pivot_table(
    index=["stay_id", "minute"], columns="event", values="rate",
    aggfunc="sum", fill_value=0
).to_dict("index")
qstates129 = np.array([
    [lookup129.get((int(s), int(round(t * 60))), {}).get(d, 0.0)
     for d in ("insulin", "norepinephrine")]
    for s, t in zip(statesE129["stay_id"], statesE129["hour"])
])
action129 = statesE129["has_insulin"] | statesE129["has_norepinephrine"]
matched129 = action129 & (qstates129.sum(axis=1) > 0)
print("=== TRANSPORT AT RECORDED DRUG BUNDLES ===")
print("Drug-flag rows / matched record rows:", action129.sum(), matched129.sum())
for label129, signal129 in (("h pairs", pairH129), ("out pairs", pairOut129)):
    A129 = np.column_stack([np.ones(matched129.sum()), qstates129[matched129]])
    coef129 = np.linalg.lstsq(A129, signal129[matched129], rcond=None)[0]
    error129 = signal129[matched129] - A129 @ coef129
    print(label129, "intercept and slopes:\n", coef129,
          "\nRMSE:", np.sqrt(np.mean(error129**2, axis=0)))
print("Examples [hour, I rate, NE rate, h pair I, h pair NE]:")
ix129 = np.flatnonzero(matched129)[:12]
print(np.column_stack([statesE129["hour"][ix129], qstates129[ix129], pairH129[ix129]]))

fE129 = pd.read_csv("/work/fwd_12/factual_edited.csv")
cacheE129 = reading_cache(tuning, statesE129, fE129)
print("=== READING CACHE ===")
print({k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
       for k, v in cacheE129.items()})
print("=== VALUE-HEAD TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES
       if n.startswith(("attr_heads.value_bin", "query_norm", "last_numeric"))])

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 11–12 / transport validation\n"
            "- Tuning A paired factual value NLL: original 1.63933, "
            "transport candidate 1.64142, increase 0.00209.\n"
            "- This passes the constraint but leaves only 0.00291 "
            "for the causal readout correction.\n"
            "- Paired states saved in /work/fwd_13; dose recovery and "
            "temporal behavior are being audited before coupling.\n")
```