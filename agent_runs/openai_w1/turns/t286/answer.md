The selective-head layout checks passed, but installation stopped because some cached weights are NumPy arrays rather than tensors. No weights changed. I’ll fix the conversion and install the quarter-strength candidate; its factual and rollout tests will determine whether shrinking the output edit is viable.

```python
def install_selective251(t):
    C = Cbase251 + float(t) * (Cfull251 - Cbase251)
    U, s, Vt = np.linalg.svd(C[144:].T, full_matrices=False)
    root = np.sqrt(s[:48])
    val = root[:, None] * Vt[:48]
    out = U[:, :48] * root[None, :]
    cand = {n: torch.tensor(a251(v).copy(), dtype=torch.float32)
            for n, v in r247_candidate.items()}
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

candidate252, Cinstalled252 = install_selective251(.25)
print("INSTALLED quarter-strength marker; coefficient factorization RMS",
      float(np.sqrt(np.mean((Cinstalled252 -
          (Cbase251 + .25*(Cfull251-Cbase251)))**2))))
print("FINITE LOCAL RESPONSE CHECKS")
for k, v in list(r246_finite.items())[:16]:
    print(k, repr(v)[:170])

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
    f.write("\n## Runs 251–252\nForward25 safe236 rollout was essentially unchanged: "
            "insulin x2 -2.051, half .981, NE x2 .071, half -.396. "
            "Run251 verified head layout but stopped before edits on numpy.clone. "
            "Run252 corrected array conversion and installed quarter-strength selective "
            "marker output with retained factual compensation. "
            "Discrete-hour six-lag convolution targets: " + repr(targets251) +
            "; conditional estimates, not known simulator truth. "
            "Counts: 252 Python runs, 25 forwards.\n")
```