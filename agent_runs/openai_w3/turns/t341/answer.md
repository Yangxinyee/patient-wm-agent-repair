Only two engineered posterior neurons use context to estimate dose policy; the other two provide action gates. I’ll remove those policy estimators’ dependence on the locally calibrated drug-response directions, centering the change on factual training contexts. If this feedback causes the asymmetry, half-dose responses should strengthen while factual loss remains within budget.

```python
import numpy as np, torch
import torch.nn.functional as F

# Preserve the validated alpha=.01 fallback.
checkpoint294 = {n: load(n).clone() for n in edited()}
raw294 = torch.as_tensor(r233_raw, dtype=torch.float32)
assert tuple(raw294.shape) == (int(n186), 4, 192)
assert hf169.shape[0] == raw294.shape[0]

dirs294 = torch.as_tensor(directions_full255, dtype=torch.float32)
assert tuple(dirs294.shape) == (192, 2)

def policy_context294(raw):
    E = F.layer_norm(
        raw, (192,), load_original("core.prelude_norm.weight"),
        load_original("core.prelude_norm.bias"), 1e-5
    )
    result = state_update82(hf169, E, mem169, mask169, age_emb169)
    outputs = list(result) if isinstance(result, (tuple, list)) else [result]
    hh = next(x for x in outputs
              if torch.is_tensor(x) and tuple(x.shape) == tuple(raw.shape))
    oo = next(x for x in outputs
              if torch.is_tensor(x) and tuple(x.shape) == (raw.shape[0], 192))
    flowed = hh * torch.exp(
        -3300.0 * torch.exp(load_original("core.flow.a"))
    )
    return oo + F.layer_norm(
        flowed.mean(1), (192,), load_original("core.out_norm.weight"),
        load_original("core.out_norm.bias"), 1e-5
    )

eps294 = 0.05
ctx_deriv294 = []
for j in range(2):
    perturb = eps294 * dirs294[:, j][None, None, :]
    plus = policy_context294(raw294 + perturb)
    minus = policy_context294(raw294 - perturb)
    ctx_deriv294.append(((plus - minus) / (2 * eps294)).mean(0).numpy())
ctx_deriv294 = np.stack(ctx_deriv294, axis=1).astype(np.float64)
u294, s294, _ = np.linalg.svd(ctx_deriv294, full_matrices=False)
keep294 = s294 > max(s294[0] * 0.01, 1e-8)
assert keep294.any()
Q294 = u294[:, keep294]
print("Drug-response context singular values:", s294)

wp294 = load("z_post.0.weight").clone()
bp294 = load("z_post.0.bias").clone()
ctx_data294 = np.asarray(Cctx148, dtype=np.float64)
dose_data294 = np.asarray(rates267)
assert ctx_data294.shape == (4213, 192)
assert dose_data294.shape == (4213, 2)
for row, drug in [(31, 0), (93, 1)]:
    old = wp294[row, :192].numpy().astype(np.float64)
    removed = (old @ Q294) @ Q294.T
    mask = dose_data294[:, drug] > 0
    centre = ctx_data294[mask].mean(0)
    wp294[row, :192] = torch.as_tensor(old - removed, dtype=torch.float32)
    bp294[row] += float(removed @ centre)
    factual_change = (ctx_data294[mask] - centre) @ removed
    print("Policy row", row,
          "removed norm", np.linalg.norm(removed),
          "factual activation-change RMS", np.sqrt(np.mean(factual_change**2)),
          "drug-context sensitivity before", old @ ctx_deriv294,
          "after", (old - removed) @ ctx_deriv294)

save_edit("z_post.0.weight", wp294)
save_edit("z_post.0.bias", bp294)
checkpoint_policy294 = {n: load(n).clone() for n in edited()}
print("Installed centered drug-response/context decoupling; factual test required.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 294; forwards 32. Preserved validated alpha=.01 in "
        "checkpoint294. Estimated a local two-drug context-response subspace "
        "by central finite differences through the exact frozen recurrence "
        "(no gradients). Removed that subspace from engineered context-policy "
        "rows31(insulin)/93(NE), centering each on its active-drug training "
        "contexts. Gates, physical-dose decoder and all other weights unchanged. "
        "This tests treatment-induced policy feedback as a cause of rollout "
        "asymmetry; candidate checkpoint_policy294 requires factual and rollout "
        "validation. Singular values: " + repr(s294.tolist()) + ".\n"
    )
```
