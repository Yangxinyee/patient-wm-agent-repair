The restored baseline’s rollout responses are nearly unchanged from the original, so it has not meaningfully repaired the dose effects. I’ll try a quarter-strength version of the selective-output edit, keeping its factual compensation, and calculate approximate rollout targets from the recorded doses and estimated response kernels. The candidate must pass factual validation and move toward those targets to be useful.

```python
import pandas as pd

def a251(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

rows251 = a251(r247_rows).astype(int)
other251 = np.array([j for j in range(192) if j not in set(rows251)], dtype=int)
Cbase251 = a251(r245_Cbase)
Cfull251 = a251(r247_C)
Cactual251 = a251(r247_Cactual)
owfull251 = a251(r247_candidate["core.prelude.0.attn.out.weight"])
kvfull251 = a251(r247_candidate["core.prelude.0.attn.kv.weight"])
print("LAYOUT", rows251[:3], rows251[-3:], Cbase251.shape)
print("OUTPUT LAYOUT ERROR",
      np.max(np.abs(Cactual251[:144] - owfull251[:, other251].T)),
      np.max(np.abs(Cactual251[144:] -
                    (owfull251[:, rows251] @ kvfull251[192 + rows251]).T)))
assert Cbase251.shape == (336, 192)
assert len(rows251) == 48 and len(other251) == 144
assert np.max(np.abs(Cactual251[:144] - owfull251[:, other251].T)) < 1e-4
assert np.max(np.abs(Cactual251[144:] -
                    (owfull251[:, rows251] @ kvfull251[192 + rows251]).T)) < 1e-4

def install_selective251(t):
    C = Cbase251 + float(t) * (Cfull251 - Cbase251)
    U, s, Vt = np.linalg.svd(C[144:].T, full_matrices=False)
    root = np.sqrt(s[:48])
    val = root[:, None] * Vt[:48]
    out = U[:, :48] * root[None, :]
    cand = {n: v.clone() for n, v in r247_candidate.items()}
    kv = a251(cand["core.prelude.0.attn.kv.weight"]).copy()
    ow = a251(cand["core.prelude.0.attn.out.weight"]).copy()
    kv[192 + rows251] = val
    ow[:, other251] = C[:144].T
    ow[:, rows251] = out
    cand["core.prelude.0.attn.kv.weight"] = torch.tensor(kv, dtype=torch.float32)
    cand["core.prelude.0.attn.out.weight"] = torch.tensor(ow, dtype=torch.float32)
    for n in list(edited()):
        drop_edit(n)
    for n, v in cand.items():
        save_edit(n, v)
    return cand, np.concatenate([C[:144], (out @ val).T], axis=0)

candidate251, Cinstalled251 = install_selective251(0.25)
print("INSTALLED marker multiplier .25; factorization error RMS",
      np.sqrt(np.mean((Cinstalled251 -
                       (Cbase251 + .25*(Cfull251-Cbase251)))**2)))
print("FINITE RESPONSE CHECKS")
for k, v in list(r246_finite.items())[:16]:
    if isinstance(v, dict):
        print(k, {kk: (np.round(a251(vv), 3).tolist()
                        if isinstance(vv, (np.ndarray, torch.Tensor))
                        and a251(vv).size <= 12 else str(vv)[:90])
                  for kk, vv in list(v.items())[:5]})
    else:
        print(k, str(v)[:180])

tuning251 = pd.read_csv("/data/tuning.csv",
                       usecols=["stay_id", "hour", "event", "rate"])
doses251 = tuning251[tuning251.event.isin(["insulin", "norepinephrine"])].copy()
print("DOSE TIME EXAMPLES\n", doses251.head(6).to_string(index=False))
kernels251 = {
    "insulin": np.array([-1.85034014,-2.25611825,-1.94263882,
                         -1.56111897,-1.02224091,-0.69018551]),
    "norepinephrine": np.array([3.35057748,2.62101256,1.71717035,
                                0.94251408,0.49378644,0.40891321])
}
roll251 = pd.read_csv("/work/fwd_25/dose_edit_original.csv")
targets251 = {}
for edit, drug, multiplier in [
    ("insulin_x2", "insulin", 2.0), ("insulin_half", "insulin", .5),
    ("ne_x2", "norepinephrine", 2.0), ("ne_half", "norepinephrine", .5)]:
    reached = roll251[(roll251.edit == edit) & roll251.reached.astype(bool)]
    effects = []
    for row in reached.itertuples(index=False):
        dd = doses251[(doses251.stay_id == row.stay_id) &
                      (doses251.event == drug) & (doses251.hour >= 12)]
        lag = np.rint(float(row.hour) - dd.hour.to_numpy()).astype(int) - 1
        ok = (lag >= 0) & (lag < 6)
        effects.append((multiplier - 1) *
                       np.dot(dd.rate.to_numpy()[ok], kernels251[drug][lag[ok]]))
    targets251[edit] = float(np.mean(effects))
print("APPROXIMATE SIX-HOUR-KERNEL TARGETS", targets251)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 251\nForward25 safe236 rollout essentially unchanged from original: "
            "insulin x2 -2.051 versus -2.053; half .981 versus1.079; "
            "NE x2 .071 versus .022; half -.396 versus -.513. "
            "Verified selective-head layout and installed quarter-strength marker output "
            "with retained factual compensation and rank48 factorization. "
            "Discrete-hour six-lag convolution targets on this batch: "
            + repr(targets251) +
            ". These are conditional record-based estimates, not simulator truth. "
            "Counts: 251 Python runs, 25 forwards.\n")
```