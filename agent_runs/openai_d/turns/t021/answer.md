The inspection output was cut off by a large weight dictionary, so it did not resolve the mismatch. I’ll restrict the next output to the small metadata fields we need. The key check is whether the state timestamps and register conventions agree with the architecture; disagreement would make a head-based edit unreliable.

```python
print("=== SMALL METADATA ONLY ===")
print("numeric_note:", VOCAB.get("numeric_note"))
print("value_bin_note:", VOCAB.get("value_bin_note"))
print("event_kinds:", VOCAB["event_kinds"])
print("atom_kinds:", VOCAB["atom_kinds"])
for key, sp0 in numeric_specs.items():
    print(key, {k: v for k, v in sp0.items() if k != "fine_bin_edges"})
print("last_numeric column norms:",
      np.round(torch.linalg.vector_norm(W0["last_numeric.weight"], dim=0).numpy(), 3))
print("last_status norms:",
      np.round(torch.linalg.vector_norm(W0["last_status.weight"], dim=1).numpy(), 3))
print("embed names:", [(n, SHAPES[n]) for n in NAMES if n.startswith("embed.")])
print("check config:", check["config"])
print("candidate keys:", list(check["candidate"]))
print("best nf first 4:", best_reconstruction["nf"][:4])
print("helper default types:",
      {name: [type(x).__name__ for x in (globals()[name].__defaults__ or ())]
       for name in ("wlin", "wnorm", "wmlp", "actfn")})

print("\n=== STATE ALIGNMENT ===")
with np.load("/work/fwd_3/states_original.npz") as zzf:
    print([(key, zzf[key].shape, str(zzf[key].dtype)) for key in zzf.files])
    for key in zzf.files:
        if zzf[key].ndim == 1:
            print(key, zzf[key][:12])
print("out vs norm(mean h), RMS:",
      float(torch.mean((out0 - wnorm(h0.mean(1), "core.out_norm"))**2).sqrt()))
print("dt range:", float(dt0.min()), float(dt0.max()))

print("\n=== ORIGINAL SUMMARY ===")
print(Path("/work/fwd_2/summary.json").read_text()[:1800])

print("\n=== ERROR STRATIFICATION ===")
ep = hm.copy()
ep["error"] = np.asarray(check["mean"]) - target_means
print(ep.groupby("event")["error"].agg(["mean", "std", "count"]).round(4))
print(ep.groupby("has_last")["error"].agg(["mean", "std", "count"]).round(4))
print("error correlations:",
      ep[["error", "last_z", "age", "dt"]].corr()["error"].round(4).to_dict())
```