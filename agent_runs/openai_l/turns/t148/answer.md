The controller reproduces the sampled state perturbations with about 0.01 RMS error, but its factual RMS may reflect either a removable offset or harmful variation. I’ll distinguish those cases and verify the state basis before saving more weights. Low factual variation and near-unit transmission would support proceeding.

```python
cout137 = w63("core.c_out.weight").numpy()
delta137 = torch.nn.functional.softplus(w63("core.delta_in")).numpy()
print("=== BASIS IDENTITIES ===")
print("SP125 vs c_out @ R118:", np.linalg.norm(SP125 - cout137 @ R118))
print("T125 vs R118/delta:", np.linalg.norm(T125 - R118 / delta137[:, None]))
print("QE Gram:", np.round(QE125.T @ QE125, 4))
print("P.T QE:", np.round(P125.T @ QE125, 4))
print("Xstate vs Jpre reshape:",
      np.linalg.norm(Xstate125-Jpre125.transpose(0,2,1).reshape(-1,192)))
for tag, arr in [("Fflow", Fflow125), ("Fflow transposed", Fflow125.transpose(0,2,1))]:
    print("Ystate vs", tag, np.linalg.norm(Ystate125-arr.reshape(-1,4)))
print("Fflow mean:", np.round(Fflow125.mean(axis=0), 4))
print("Jpre.P mean:",
      np.round(np.einsum("ndi,do->nio", Jpre125, P125).mean(axis=0), 4))

print("\n=== FACTUAL CONTROLLER OFFSETS AND NOISE ===")
for p, b in list(controllers125.items()) + [("P125", P125)]:
    v = Ef124 @ b
    print(p, "mean", np.round(v.mean(axis=0),5),
          "SD", np.round(v.std(axis=0),5),
          "perturbation RMSE", np.round(np.sqrt(np.mean((Xstate125@b-Ystate125)**2,axis=0)),5))
print("Ef quiet mean/SD:", (Ef124 @ QE125).mean(axis=0), (Ef124 @ QE125).std(axis=0))
print("Capture118:", linear_capture118.__code__.co_consts)
print("Capture125:", capture_pre125.__code__.co_consts)
print("State variables shapes:",
      {k: tuple(globals()[k].shape) for k in
       ["Hpre118","Hpost118","Rbase118","Cpre118","Cstate118","R118",
        "Jselected118","xf118","xmean118","Jhead118","state_rows118",
        "hrows118","epsrows118"] if k in globals() and hasattr(globals()[k],"shape")})
print("Decoder signal gain:", np.round(Jselected118.T @ Vsignal118, 4))

print("\n=== BUNDLE INPUT RETURN INTERFACE ===")
compact_bytecode136("bundle_inputs73")
print("\n=== RETAINED INPUT ARRAY NAMES ===")
print(", ".join(k for k in globals() if re.search(r"(60|63|65|67|72|73|74|81|96|102|103)$",k))[:2500])
```