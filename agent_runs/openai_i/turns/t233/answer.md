The automatic coordinate match failed: it selected the same coordinate for both drugs. The token target also has an unexplained offset from the original computation. I won’t install that fit; I’ll resolve the reserved coordinates and bias convention first.

```python
print("CHANNEL METADATA:")
for k, v in list(globals().items()):
    if any(s in k for s in ["196", "201", "202", "203", "204"]):
        if isinstance(v, (int, float, str)):
            print(k, str(v)[:100])
        elif isinstance(v, (list, tuple)) and len(v) <= 12:
            print(k, str(v)[:250])

print("\nPRIVATE POSTERIOR COMPUTATION:")
pW0208 = torch.as_tensor(private_post201["z_post.0.weight"], dtype=torch.float32)
pb0208 = torch.as_tensor(private_post201["z_post.0.bias"], dtype=torch.float32)
pW2208 = torch.as_tensor(private_post201["z_post.2.weight"], dtype=torch.float32)
pb2208 = torch.as_tensor(private_post201["z_post.2.bias"], dtype=torch.float32)
hidden208 = torch.nn.functional.gelu(
    torch.nn.functional.linear(input196, pW0208, pb0208)
)
Z208 = torch.nn.functional.linear(hidden208, pW2208, pb2208)[:, :32].numpy()
print("Recomputed versus cached posterior RMS:",
      np.sqrt(np.mean((Z208 - np.asarray(all_z201))**2)))
for j in [12, 16, 22]:
    print("coordinate", j,
          "cached mean/std", np.mean(all_z201[:, j]), np.std(all_z201[:, j]),
          "recomputed mean/std", np.mean(Z208[:, j]), np.std(Z208[:, j]),
          "output-row norm", float(torch.linalg.vector_norm(pW2208[j])),
          "bias", float(pb2208[j]))
print("Feature predictions mean/std:",
      np.mean(feature_pred201, axis=0), np.std(feature_pred201, axis=0))
print("Feature coefficient norms:", np.linalg.norm(feature_coef201, axis=0))
print("Correlations, recomputed private coordinates versus features:",
      np.corrcoef(np.c_[Z208[:, [12, 16]], feature_pred201], rowvar=False)[:2, 2:])

orig_token208 = (
    np.asarray(Zbase196) @ asnp175(load_original("z_in.weight")).T
    + asnp175(load_original("z_in.bias"))
)
delta_target208 = orig_token208 - np.asarray(Yzin196)
print("\nTOKEN TARGET OFFSET:")
print("Offset row-variation RMS:",
      np.sqrt(np.mean((delta_target208 - delta_target208.mean(axis=0))**2)))
print("Offset minus original bias RMS:",
      np.sqrt(np.mean((delta_target208 - asnp175(load_original("z_in.bias")))**2)))
print("Neutral base bias RMS:", np.sqrt(np.mean(zin_bias_base201**2)))
print("Original z_in bias RMS:",
      np.sqrt(np.mean(asnp175(load_original("z_in.bias"))**2)))
print("Xzin196 constant columns:",
      np.where(np.std(Xzin196, axis=0) < 1e-10)[0])
print("Original-token fit with cached base weights RMS:",
      np.sqrt(np.mean((np.asarray(all_z201) @ zin_newW196.T
                       + zin_bias_base201 - orig_token208)**2)))
print("ACTIVE EDITS:", edited())
```