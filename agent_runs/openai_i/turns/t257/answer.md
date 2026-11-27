The route transports dose changes nearly proportionally across x0–x3, with little cross-drug leakage. Its next-reading response is still too weak: roughly −0.39 for insulin and +0.08 for norepinephrine at x2, versus record estimates near −2 and +3.3. I’ll measure how the unused output columns affect the readings using forward finite differences, without gradients. This will test whether a modest weight correction can supply the missing response.

```python
import time, numpy as np, torch

rng230 = np.random.default_rng(230)
cal_idx230 = np.sort(np.concatenate([
    rng230.choice(np.flatnonzero(drug_labels215 == d), 56, replace=False)
    for d in ["insulin", "norepinephrine"]
]))
print("CALIBRATION CONTEXTS", len(cal_idx230),
      "unique bundle ids", len(np.unique(np.asarray(ids182)[cal_idx230])))

Udouble230 = Ubase215.clone()
for drug in ["insulin", "norepinephrine"]:
    jj = np.flatnonzero(drug_labels215 == drug)
    bundle_ids = np.asarray(ids182)[jj]
    Udouble230[bundle_ids] = actions229[(drug, 2.)][bundle_ids]

h230f, o230f = replay_route228(route228, Ubase215, cal_idx230)
h230d, o230d = replay_route228(route228, Udouble230, cal_idx230)
m230f = means229(h230f, o230f, cal_idx230)
m230d = means229(h230d, o230d, cal_idx230)
native_pair230 = m230d - m230f

Jpair230 = np.full((len(cal_idx230), 4, 192), np.nan)
Jfact230 = np.full_like(Jpair230, np.nan)
eps230 = 0.15
completed230 = 0
start230 = time.monotonic()

with open("/work/notes.md", "a") as f230:
    f230.write("\n## Python 230 — direct response-column finite differences\n"
               "- Route x0–x3 transport slopes are approximately .77–.86, with small cross-drug leakage.\n"
               "- Native proximal x2 slopes remain weak: insulin -.386 and NE +.076.\n"
               "- Measuring paired and factual readout sensitivities for reserved output columns by forward passes only.\n"
               "- No gradients, optimizer, or installed response edits are used.\n")

for j in range(192):
    candidate = dict(route228)
    ow = np.array(route228["core.prelude.0.attn.out.weight"], copy=True)
    ow[j, response_cols228] = eps230
    candidate["core.prelude.0.attn.out.weight"] = ow

    hf, of = replay_route228(candidate, Ubase215, cal_idx230)
    hd, od = replay_route228(candidate, Udouble230, cal_idx230)
    mf = means229(hf, of, cal_idx230)
    md = means229(hd, od, cal_idx230)
    Jfact230[:, :, j] = (mf - m230f) / eps230
    Jpair230[:, :, j] = ((md - mf) - native_pair230) / eps230
    completed230 = j + 1
    if completed230 % 32 == 0:
        print("Completed", completed230, "seconds", round(time.monotonic() - start230, 1))
    if time.monotonic() - start230 > 740:
        break

np.savez("/work/response_fd230.npz", indices=cal_idx230,
         paired=Jpair230, factual=Jfact230, native=native_pair230,
         completed=completed230, epsilon=eps230)
print("FINISHED", completed230, "of 192; seconds", round(time.monotonic() - start230, 1))
if completed230 == 192:
    for drug in ["insulin", "norepinephrine"]:
        mask = drug_labels215[cal_idx230] == drug
        rates = rate_vector215[cal_idx230][mask]
        average_J = Jpair230[mask].sum(0) / rates.sum()
        print(drug, "mean sensitivity singular values",
              np.round(np.linalg.svd(average_J, compute_uv=False), 6))
```