The architecture confirms that the latent carrier reaches the state only through prelude attention, which explains why earlier carrier edits had little effect. The relay could address that bottleneck. I’ll inspect its reserved features and the carrier’s factual residuals next; large residuals would warn against increasing its influence.

```python
def brief210(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return {"shape": x.shape, "rms": float(np.sqrt(np.mean(x.astype(float)**2)))}
    if isinstance(x, dict):
        return {str(k): brief210(v) for k, v in list(x.items())[:12]}
    if isinstance(x, (list, tuple)):
        return repr(x)[:220]
    return repr(x)[:160]

print("=== RELAY FIT DEFINITIONS ===")
print("fits:", brief210(relay206["fits"]))
for n in ("rx_layout", "lc_dims", "lc_zin_base", "lc_zbias_base",
          "lc_edirections", "lc_comp", "rp_comp_w2", "rp_comp_b2",
          "relay208_T0", "relay208_Tc"):
    if n in globals():
        print(n, brief210(globals()[n]))

print("\n=== CARRIER INSTALLER SIGNATURE ===")
for fn in ("install_carrier200",):
    if fn in globals():
        ff = globals()[fn]
        print(fn, inspect.signature(ff), "constants:", ff.__code__.co_consts)

print("\n=== CARRIER BRANCH SPECIFICATIONS ===")
for drug, spec in rp_branch_specs.items():
    print(drug, {k: brief210(v) for k, v in spec.items()})

print("\n=== FACTUAL AND EDITED CARRIER RESIDUALS ===")
for drug in ("insulin", "norepinephrine"):
    mask = np.asarray(rp_cf_masks[drug], dtype=bool)
    q = np.asarray(lc_branch[drug])
    print(drug, "active", int(mask.sum()),
          "factual active quantiles", np.quantile(q[mask], [0, .05, .5, .95, 1]),
          "inactive RMS", float(np.sqrt(np.mean(q[~mask]**2))))
    for mult in (0., .5, 1.5, 2., 3.):
        qc = np.asarray(lc_qcf[(drug, mult)])
        print(" ", mult, "active contrast quantiles",
              np.quantile((qc-q)[mask], [0, .1, .5, .9, 1]))

print("\n=== TENSOR NAMES AND RESERVED RELAY ROWS ===")
print([n for n in NAMES if "z_in" in n or "z_post" in n])
for n in relay207_base:
    a = load(n).detach().cpu().numpy()
    if "kv.weight" in n:
        norms = np.linalg.norm(a[192:], axis=1)
        print(n, "zero/small value rows", np.flatnonzero(norms < 1e-7).tolist(),
              "head row norms", np.round(norms.reshape(4, 48).mean(1), 4))
    elif "out.weight" in n:
        norms = np.linalg.norm(a, axis=0)
        print(n, "zero/small output columns", np.flatnonzero(norms < 1e-7).tolist())
    elif "q.weight" in n:
        print(n, "query head norms",
              np.linalg.norm(a.reshape(4, 48, 192), axis=(1, 2)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 210\n"
            "Architecture confirms prelude attention is the carrier's only state entry. "
            "Inspecting relay reserved rows and factual carrier leakage before any new edit. "
            "Counts: 210 Python runs, 19 forwards.\n")
```