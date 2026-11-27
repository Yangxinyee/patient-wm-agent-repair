Small posterior-latent changes produced essentially no reading response, so an output-layer-only edit will not suffice. I’ll check whether larger latent changes reach the recurrent state and inspect the latent-to-state mapping. A genuinely inactive pathway would require a broader repair; a weak mapping might be strengthened without disturbing factual predictions much.

```python
print("=== LATENT PATHWAY DIAGNOSTICS ===")
for amp in [1.0, 10.0, 100.0]:
    delta = np.zeros(32, dtype=np.float32)
    delta[0] = amp
    st = np.asarray(transition_with_z95(probe_indices95, delta))
    ds = st - np.asarray(base_transition95)
    dm = probe_means95(st) - probe_means95(base_transition95)
    print("amplitude", amp,
          "out RMS", float(np.sqrt(np.mean(ds[:, :192] ** 2))),
          "h RMS", float(np.sqrt(np.mean(ds[:, 192:960] ** 2))),
          "posterior RMS", float(np.sqrt(np.mean(ds[:, 960:] ** 2))),
          "glucose mean/RMS",
          (float(np.mean(dm[is_glucose95])),
           float(np.sqrt(np.mean(dm[is_glucose95] ** 2)))),
          "MAP RMS", float(np.sqrt(np.mean(dm[~is_glucose95] ** 2))))

print("Latent Jacobian singular values:", np.linalg.svd(J95, compute_uv=False)[:8])
for name in ["z_in.weight", "z_in.bias", "z_head.weight",
             "core.prelude.0.self_attn.in_proj_weight",
             "attr_heads.value_bin.0.weight"]:
    if name in NAMES:
        w = np.asarray(load_original(name))
        print(name, "RMS", float(np.sqrt(np.mean(w ** 2))),
              "max", float(np.max(np.abs(w))))

Zbase96 = np.asarray(
    rp_mlp_original95(torch.as_tensor(raw87, dtype=torch.float32), "z_post")
)[:, :32]
print("Posterior reconstruction RMSE:",
      np.sqrt(np.mean((Zbase96 - repair_bank["x"][idx82, 960:]) ** 2)))
zmean96 = Zbase96[train94].mean(axis=0)
zcov96 = np.cov(Zbase96[train94], rowvar=False)
zeig96, zevec96 = np.linalg.eigh(zcov96)
null96 = zevec96[:, 0]
print("Smallest posterior covariance eigenvalues:", zeig96[:8])
print("Held-out near-null latent RMS:",
      np.sqrt(np.mean(((Zbase96[test94] - zmean96) @ null96) ** 2)))
print("Original z_in response norm along near-null direction:",
      np.linalg.norm(np.asarray(load_original("z_in.weight")) @ null96))

print("Transition locals:", transition73.__code__.co_varnames)
print("Transition constants:", str(transition73.__code__.co_consts)[:2400])
print("Linear helper:", inspect.signature(rp_linear), rp_linear.__code__.co_consts)
print("Attention helper:", inspect.signature(attention67))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 96\n"
            "Small posterior-latent perturbations have essentially zero reading sensitivity. "
            "Testing larger perturbations and near-null latent directions to distinguish "
            "an ignored pathway from weak z_in transfer. No weights edited.\n")
```
