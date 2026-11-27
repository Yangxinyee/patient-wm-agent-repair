The recurrence check stopped because the age input lacked its final scalar dimension. That is an interface error, not evidence of a model mismatch. I’ll correct the shape and repeat the check; small state errors would validate the reconstruction.

```python
import numpy as np
import torch
from pathlib import Path

def bundle_inputs74(indices):
    ctx, hflow, memory, ages, mask = bundle_inputs73(indices)
    return ctx, hflow, memory, ages[..., None], mask

ctx74, hf74, mem74, age74, mask74 = bundle_inputs74(checkidx73)
post74 = mlp63(
    torch.cat([ctx74, e_valid60[checkidx73], u72[checkidx73]], dim=1),
    "z_post"
)[:, :32]
result74 = transition64(
    hf74, e_valid60[checkidx73], u72[checkidx73], post74,
    mem74, age74, mask74
)
print("=== RECURRENCE CHECK ===")
print("return shapes:", [tuple(x.shape) for x in result74])
out74 = next(x for x in result74 if x.shape == (64, 192))
h74 = next(x for x in result74 if x.shape == (64, 4, 192))
print("out RMS:", float(
    (out74-sx63[checkidx73, :192]).square().mean().sqrt()
))
print("h RMS:", float(
    (h74-sx63[checkidx73, 192:960].reshape(-1, 4, 192))
    .square().mean().sqrt()
))
print("posterior RMS:", float(
    (post74-sx63[checkidx73, 960:]).square().mean().sqrt()
))

validrows73 = np.flatnonzero(np.asarray(preidx67) >= 0)
pi73 = np.asarray(preidx67)[validrows73].astype(int)
dtread73 = torch.as_tensor(
    (measure67.iloc[validrows73]["hour"].to_numpy()-hour63[pi73])*3600,
    dtype=torch.float32
)
ctxread73, _ = context64(
    sx63[pi73, :192],
    sx63[pi73, 192:960].reshape(-1, 4, 192),
    dtread73
)
epsgen73 = torch.Generator().manual_seed(7301)
epsread73 = torch.randn((8, len(validrows73), 32), generator=epsgen73)
means73, probs73, hidden73, logits73 = decode67(
    ctxread73, const_train67[validrows73],
    trans_train67[validrows73], centres_train67[validrows73],
    epsread73, return_features=True
)
print("\n=== FACTUAL DECODER FEATURES ===")
print("shapes:", tuple(hidden73.shape), tuple(logits73.shape))
events74 = measure67.iloc[validrows73]["event"].to_numpy()
for code74 in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    m74 = events74 == code74
    print(code74, "n=", int(m74.sum()),
          "observed mean=", round(float(
              measure67.iloc[validrows73].loc[m74, "value"].mean()), 3),
          "predicted mean=", round(float(means73[:, m74].mean()), 3))

rates_by_state73 = {}
for drug74 in ("insulin", "norepinephrine"):
    rr74 = np.zeros(len(sx63), dtype=np.float64)
    choose74 = np.asarray(drugs65) == drug74
    np.add.at(rr74, owners72[choose74], np.asarray(rates65)[choose74])
    rates_by_state73[drug74] = rr74
print("readings immediately after drug bundles:",
      int(np.isin(pi73, drug_indices49).sum()))
print("VOCAB keys:", list(VOCAB))
print("No edits:", edited())

with Path("/work/notes.md").open("a") as f74:
    f74.write(
        "\n## Runs 73–74\n"
        "Run 73 stopped because transition64 requires age shape [N,128,1]. "
        "Corrected scalar axis and repeated state validation and factual-feature "
        "construction. No weights edited; forward passes 4/50.\n"
    )
```