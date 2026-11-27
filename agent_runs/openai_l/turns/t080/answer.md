The corrected action encoder matches the reference to about \(10^{-7}\). I’ll now validate the complete drug-bundle update and reconstruct the factual decoder features. If both agree with the exported states, we can fit a value-head correction while explicitly constraining factual predictions.

```python
import numpy as np
import torch
from pathlib import Path

torch.set_num_threads(8)

def bundle_inputs73(indices):
    indices = np.asarray(indices, dtype=int)
    previous = indices - 1
    assert np.all(previous >= 0)
    assert np.all(sid63[previous] == sid63[indices])
    dt = torch.as_tensor(
        (hour63[indices] - hour63[previous]) * 3600, dtype=torch.float32
    )
    ctx, hflow = context64(
        sx63[previous, :192],
        sx63[previous, 192:960].reshape(-1, 4, 192),
        dt,
    )
    memory = torch.zeros((len(indices), 128, 192), dtype=torch.float32)
    ages = torch.zeros((len(indices), 128), dtype=torch.float32)
    mask = torch.ones((len(indices), 128), dtype=torch.bool)
    for row, j in enumerate(indices):
        first = j
        while first > max(0, j - 127) and sid63[first - 1] == sid63[j]:
            first -= 1
        jj = np.arange(first, j + 1)
        n = len(jj)
        memory[row, :n] = e_valid60[jj]
        ages[row, :n] = torch.as_tensor(
            (hour63[j] - hour63[jj]) * 3600, dtype=torch.float32
        )
        mask[row, :n] = False
    return ctx, hflow, memory, ages, mask

rng73 = np.random.default_rng(731)
checkidx73 = np.sort(rng73.choice(drug_indices49, 64, replace=False))
ctx73, hf73, mem73, age73, mask73 = bundle_inputs73(checkidx73)
post73 = mlp63(
    torch.cat([ctx73, e_valid60[checkidx73], u72[checkidx73]], dim=1),
    "z_post"
)[:, :32]
result73 = transition64(
    hf73, e_valid60[checkidx73], u72[checkidx73], post73,
    mem73, age73, mask73
)
print("=== RECURRENCE CHECK ===")
print("return type:", type(result73).__name__)
if isinstance(result73, (tuple, list)):
    print("return shapes:", [tuple(x.shape) for x in result73])
    for i73, x73 in enumerate(result73):
        if x73.shape == (64, 192):
            print("out RMS:", float(
                (x73 - sx63[checkidx73, :192]).square().mean().sqrt()
            ))
        elif x73.shape == (64, 4, 192):
            print("h RMS:", float(
                (x73 - sx63[checkidx73, 192:960].reshape(-1, 4, 192))
                .square().mean().sqrt()
            ))
print("posterior RMS:", float(
    (post73 - sx63[checkidx73, 960:]).square().mean().sqrt()
))

print("\n=== FACTUAL DECODER FEATURES ===")
validrows73 = np.flatnonzero(np.asarray(preidx67) >= 0)
pi73 = np.asarray(preidx67)[validrows73].astype(int)
dtread73 = torch.as_tensor(
    (measure67.iloc[validrows73]["hour"].to_numpy() - hour63[pi73]) * 3600,
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
print("feature shapes:", tuple(hidden73.shape), tuple(logits73.shape))
for code73 in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    m73 = measure67.iloc[validrows73]["event"].to_numpy() == code73
    print(code73, "n=", int(m73.sum()),
          "observed mean=", float(measure67.iloc[validrows73].loc[m73, "value"].mean()),
          "predicted mean=", float(means73[:, m73].mean()))

rates_by_state73 = {}
for drug73 in ("insulin", "norepinephrine"):
    rr73 = np.zeros(len(sx63), dtype=np.float64)
    choose73 = np.asarray(drugs65) == drug73
    np.add.at(rr73, owners72[choose73], np.asarray(rates65)[choose73])
    rates_by_state73[drug73] = rr73
print("readings immediately after drug bundles:",
      int(np.isin(pi73, drug_indices49).sum()))
print("No edits:", edited())

with Path("/work/notes.md").open("a") as f73:
    f73.write(
        "\n## Run 73 / transition and decoder validation\n"
        "Corrected full action encoder RMS=1.16e-7. Constructing memory with current "
        "e included and seconds-based ages; checking complete recurrence against "
        "exported states. Retaining factual decoder hidden features for an analytic "
        "value-head correction with factual-preservation constraints. No edits.\n"
    )
```