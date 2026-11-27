The best local convention includes the current bundle in memory and uses global attention packing, but it still does not exactly reproduce the saved states. I’ll use it only to propose a correction, not to certify one. Next I’ll recalculate response directions across more stays and test finite-sized shifts. Consistent held-out responses would justify a GPU trial; saturation or strong cross-drug effects would argue against this approach.

```python
import numpy as np
import torch

def response_pair68(j, h):
    a, b = next_means61(int(j), h)
    if a is None:
        return None, None
    a = a.detach().numpy() if torch.is_tensor(a) else np.asarray(a)
    b = b.detach().numpy() if torch.is_tensor(b) else np.asarray(b)
    if a.dtype.kind in "OUS" and b.dtype.kind not in "OUS":
        a, b = b, a
    return a.reshape(len(h), -1), b.astype(str).reshape(-1)

rng68 = np.random.default_rng(6801)
candidate_rows68 = np.asarray(available61, dtype=int)
candidate_rows68 = candidate_rows68[
    np.asarray(counts_start4)[candidate_rows68] > 0]
sample68 = np.sort(rng68.choice(candidate_rows68,
                               min(64, len(candidate_rows68)), replace=False))
jac68 = np.full((len(sample68), 2, 32), np.nan)
eps68 = 0.05
old_kv68, old_mem68 = wm_kv_layout, wm_memory_mode
step_options68 = {k: best67[k] for k in
                  ["include_current", "age_mode", "age_units", "age_scale"]}
try:
    wm_kv_layout, wm_memory_mode = "global", best67["memory"]
    for i68, j68 in enumerate(sample68):
        zbase68 = s4["x"][j68:j68+1, 960:992]
        zb68 = np.repeat(zbase68, 65, axis=0)
        zb68[1:33] += eps68 * np.eye(32, dtype=np.float32)
        zb68[33:65] -= eps68 * np.eye(32, dtype=np.float32)
        hb68 = local_step67(int(j68), zb68, **step_options68)
        means68, ev68 = response_pair68(j68, hb68)
        if means68 is None:
            continue
        for d68, prefix68 in enumerate(["glucose", "map"]):
            mask68 = np.char.startswith(ev68, prefix68)
            if mask68.any():
                jac68[i68, d68] = (
                    means68[1:33, mask68].mean(axis=1)
                    - means68[33:65, mask68].mean(axis=1)) / (2*eps68)

    sid_sample68 = np.asarray(s4["stay_id"])[sample68]
    unique68 = np.unique(sid_sample68)
    fit_ids68 = rng68.choice(unique68, max(1, int(.7*len(unique68))),
                            replace=False)
    fit_mask68 = np.isin(sid_sample68, fit_ids68)
    A68, T68 = [], []
    for i68 in np.flatnonzero(fit_mask68):
        for d68 in range(2):
            if np.isfinite(jac68[i68, d68]).all():
                A68.append(jac68[i68, d68])
                T68.append(target61[d68])
    A68, T68 = np.asarray(A68), np.asarray(T68)
    gram68 = A68.T @ A68
    ridge_unit68 = np.trace(gram68) / 32
    vector_trials68 = []
    for factor68 in [.001, .01, .1, 1.0]:
        V68_trial = np.linalg.solve(
            gram68 + factor68*ridge_unit68*np.eye(32), A68.T @ T68)
        response68 = np.einsum("ndr,rc->ndc", jac68, V68_trial)
        errors68 = response68 - target61[None]
        validation68 = errors68[~fit_mask68]
        score68 = float(np.sqrt(np.nanmean(validation68**2)))
        vector_trials68.append((score68, factor68, V68_trial))
        print("ridge factor", factor68,
              "held-out response RMSE", round(score68, 4),
              "column norms", np.round(np.linalg.norm(V68_trial, axis=0), 3),
              "held-out mean response",
              np.round(np.nanmean(response68[~fit_mask68], axis=0), 3))
    vector_trials68.sort(key=lambda t: t[0])
    V68 = vector_trials68[0][2]

    finite_response68 = []
    for j68 in sample68[::4]:
        zbase68 = s4["x"][j68:j68+1, 960:992]
        zbranches68 = np.vstack([
            zbase68, zbase68 + V68[:, 0],
            zbase68 + V68[:, 1]])
        hb68 = local_step67(int(j68), zbranches68, **step_options68)
        means68, ev68 = response_pair68(j68, hb68)
        if means68 is None:
            continue
        shifts68 = np.full((2, 2), np.nan)
        for d68, prefix68 in enumerate(["glucose", "map"]):
            mask68 = np.char.startswith(ev68, prefix68)
            if mask68.any():
                shifts68[d68] = (
                    means68[1:, mask68].mean(axis=1)
                    - means68[0, mask68].mean())
        finite_response68.append(shifts68)
    finite_response68 = np.asarray(finite_response68)
    print("Finite unit-innovation mean response:",
          np.round(np.nanmean(finite_response68, axis=0), 4))
    print("Finite unit-innovation SD:",
          np.round(np.nanstd(finite_response68, axis=0), 4))
finally:
    wm_kv_layout, wm_memory_mode = old_kv68, old_mem68

print("\n=== INNOVATION TARGET CHECK ===")
for d68, drug68 in enumerate(["insulin", "norepinephrine"]):
    mask68 = np.asarray(present55)[:, d68] & np.asarray(valid55)
    actual68 = rate55[mask68, d68]
    expected68 = expected55[mask68, d68]
    residual68 = innovation55[mask68, d68]
    print(drug68, "n", int(mask68.sum()),
          "rate/expected/innovation means",
          np.round([actual68.mean(), expected68.mean(), residual68.mean()], 4),
          "innovation SD", round(float(residual68.std()), 4),
          "max |innovation-(rate-expected)|",
          round(float(np.max(np.abs(residual68-(actual68-expected68)))), 5))
print("Tuning IDs first 12:", tuning_ids_repair[:12])
print("Training-record dataframe candidates:", [
    (n, tuple(o.shape)) for n, o in list(globals().items())
    if isinstance(o, pd.DataFrame)
    and {"stay_id", "hour", "event"}.issubset(o.columns)
    and len(o) > 10000])
print("Edits:", edited())

with open("/work/notes.md", "a") as f68:
    f68.write(
        "\n## Runs 67–68: provisional response reconstruction\n"
        "- Global attention packing and current-inclusive memory improved local recurrence substantially.\n"
        "- Best local convention remains approximate; it cannot certify factual fit or rollout effects.\n"
        "- Recomputed latent response directions across stay-held-out samples and checked finite-sized shifts.\n"
        "- No weights edited. Any eventual candidate requires paired GPU factual and rollout validation.\n")
```