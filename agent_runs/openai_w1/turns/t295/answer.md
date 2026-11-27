The recorded-token reconstruction matched exactly. The counterfactual cache lacks full token arrays, so I will recover them from its pooled features and check their normalization. I will also recover physical dose units before fitting an edit. Incorrect reconstruction or scaling would invalidate this route.

```python
print("=== POOL DEFINITION CHECK ===")
poolchecks259 = {
    "mean": nt240.mean(axis=1),
    "sum_sqrt": nt240.sum(axis=1)/np.sqrt(7),
    "sum": nt240.sum(axis=1),
}
poolerrors259 = {k: float(np.sqrt(np.mean((v-pools238)**2)))
                 for k, v in poolchecks259.items()}
print(poolerrors259)
poolkind259 = min(poolerrors259, key=poolerrors259.get)
poolfactor259 = {"mean": 7., "sum_sqrt": np.sqrt(7), "sum": 1.}[poolkind259]

Es259, nts259 = {}, {}
if poolerrors259[poolkind259] < 1e-5:
    for key, obj in cf239.items():
        nt = nt240.copy()
        nt[:, 6, :] = rx_norm(torch.as_tensor(obj["token"], dtype=torch.float32),
                             "core.prelude.0.norm1").numpy()
        nt[:, 5, :] = (poolfactor259*obj["pool"] -
                       nt[:, :5, :].sum(axis=1) - nt[:, 6, :])
        u = (nt[:, 5, :]-np.asarray(rx_W["core.prelude.0.norm1.bias"])) / \
            np.asarray(rx_W["core.prelude.0.norm1.weight"])
        mask = np.asarray(obj["mask"], dtype=bool)
        meanerr = float(np.max(np.abs(u.mean(axis=1))))
        varerr = float(np.median(np.abs(u.var(axis=1)-1)))
        if meanerr < 1e-3 and varerr < 1e-3:
            nts259[key] = nt
            Es259[key] = E258(nt)
        print(key, "n", int(mask.sum()), "LN mean/variance errors",
              round(meanerr, 7), round(varerr, 7),
              "E contrast RMS", round(float(np.sqrt(np.mean(
                  (Es259[key][mask]-factualE258[mask])**2))), 6)
              if key in Es259 else "rejected")
else:
    print("Pool inversion rejected.")

print("\n=== PHYSICAL DOSE / ALIGNMENT ARRAYS ===")
_seen259 = set()
for k, v in list(globals().items()):
    if k.startswith("_") or id(v) in _seen259:
        continue
    if isinstance(v, pd.DataFrame) and v.shape[0] == 5215:
        print(k, "DataFrame", list(v.columns))
    if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim <= 2 and \
       v.shape[0] == 5215 and v.size <= 5215*6 and \
       any(s in k.lower() for s in ["rate", "dose", "target", "phys", "drug", "pres", "meta"]):
        a = np.asarray(v)
        print(k, a.shape, "min/max/mean", np.nanmin(a), np.nanmax(a), np.nanmean(a))
        _seen259.add(id(v))

print("\n=== CARRIER SCALING ===")
for k in ["lc_dims", "lc_scales", "lc_units", "lc_meta", "rp_meta", "rp_y", "lc_y"]:
    if k in globals():
        print(k, brief258(globals()[k]))
for name in ["branch195", "branch186", "rp_encode", "posterior211"]:
    f = globals().get(name)
    if callable(f):
        print(name, inspect.signature(f), "names", f.__code__.co_names,
              "constants", [c for c in f.__code__.co_consts
                            if isinstance(c, (str, int, float, tuple))][:12])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 259\n"
            "Verified factual E reconstruction; recovering CF action-token normalization algebraically "
            "from pooled features and latent token, with layer-normalization validity checks. "
            "Recovering physical-dose scaling before fitting core.b_in correction. No weights changed. "
            "Counts: 259 Python runs, 27 forwards.\n")
```