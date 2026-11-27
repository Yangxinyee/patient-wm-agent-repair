The CPU reconstruction is not exact, so I cannot trust it to calibrate an effect size. It may still help propose directions for GPU testing. I’ll first construct two quiet markers and check their dose responses; strong counterfactual signals with little background activation would support testing them in the model.

```python
import builtins as bi

print("=== VALUE-HEAD CACHE INVENTORY ===")
for n, v in sorted(bi.list(globals().items())):
    if n.startswith(("ce_", "head")) and bi.isinstance(v, bi.dict):
        print(n, {str(k): tuple(x.shape) if bi.isinstance(x, (np.ndarray, torch.Tensor))
                  else bi.type(x).__name__ for k, x in bi.list(v.items())[:15]})
print("Code lookup:", ce_code)
print("Dose normalisation:",
      {str(k): {a: v[a] for a in ("median", "iqr") if a in v}
       for k, v in ce_meta.items() if k[0] in ("insulin", "norepinephrine")})

# Choose low-variance marker axes orthogonal to ordinary latent-token content
# and to the relay's 46 already occupied value features.
L215 = relay213_evecs[:, 1:161]
keep215 = np.setdiff1d(np.arange(32), np.asarray(lc_dims, dtype=int))
kv215 = load("core.prelude.0.attn.kv.weight").numpy()
constraints215 = np.column_stack([
    gamma213[:, None] * np.asarray(lc_zin_base)[:, keep215],
    gamma213 * np.asarray(lc_zbias_base),
    kv215[192+96:192+142].T,
    gamma213,
    1.0/gamma213
])
_, s215, vh215 = np.linalg.svd(constraints215.T @ L215, full_matrices=True)
rank215 = int(np.sum(s215 > max(s215[0]*1e-8, 1e-10)))
null215 = L215 @ vh215[rank215:].T
eval215, rot215 = np.linalg.eigh(
    null215.T @ (P213[train213].T @ P213[train213]/train213.sum()) @ null215)
U215 = null215 @ rot215[:, :2]
C215 = U215/gamma213[:, None]
C215 /= np.linalg.norm(C215, axis=0, keepdims=True)
zin215 = np.asarray(lc_zin_base).copy()
zin215[:, np.asarray(lc_dims, dtype=int)] = .2*C215

def norm215(x):
    return F.layer_norm(torch.as_tensor(np.asarray(x), dtype=torch.float32),
                        (192,), load_original("core.prelude.0.norm1.weight"),
                        load_original("core.prelude.0.norm1.bias"), 1e-5).numpy()

def fixedpost215(x):
    h = F.gelu(F.linear(torch.as_tensor(np.asarray(x), dtype=torch.float32),
                        torch.as_tensor(carrier200_w0),
                        torch.as_tensor(carrier200_b0)))
    return F.linear(h, torch.as_tensor(carrier200_w2, dtype=torch.float32),
                    torch.as_tensor(carrier200_b2, dtype=torch.float32))[:, :32].numpy()

nt215 = np.asarray(relay206["norm_tokens"])
print("\n=== INPUT ALIGNMENT ===")
print("Latent norm RMSE", np.sqrt(np.mean((norm215(relay208_T0)-nt215[:, 6])**2)))
print("Action norm RMSE", np.sqrt(np.mean((norm215(rp_input[:, 384:])-nt215[:, 5])**2)))

zf215 = fixedpost215(rp_input)
tf215 = zf215 @ zin215.T + np.asarray(lc_zbias_base)
Pf215 = P213 + (norm215(tf215)-nt215[:, 6])/7
Ff215 = Pf215 @ U215
cf215, xs215, ys215 = {}, [], []
for drug in ("insulin", "norepinephrine"):
    mask = np.asarray(rp_cf_masks[drug], dtype=bool)
    for mult in (0., .5, 1.5, 2., 3.):
        xc = np.asarray(rp_cf_input[(drug, mult)])
        zc = fixedpost215(xc)
        tc = zc @ zin215.T + np.asarray(lc_zbias_base)
        Pc = (P213 + (norm215(tc)-nt215[:, 6])/7
              + (norm215(xc[:, 384:])-nt215[:, 5])/7)
        df = (Pc-Pf215) @ U215
        dq = (zc-zf215)[:, np.asarray(lc_dims, dtype=int)]
        cf215[(drug, mult)] = {"df": df, "dq": dq, "pool": Pc, "mask": mask}
        m = mask & train213
        xs215.append(df[m])
        ys215.append(dq[m])
X215, Y215 = np.concatenate(xs215), np.concatenate(ys215)
A215 = np.linalg.solve(X215.T @ X215 + 1e-7*np.eye(2), X215.T @ Y215)
V215 = (U215 @ A215).T

print("\n=== MARKER QUALITY ===")
print("Constraint rank", rank215, "remaining dimensions", null215.shape[1])
print("Marker background eigenvalues", eval215[:2])
print("Readout norms", np.linalg.norm(V215, axis=1))
print("Original-input feature RMS", np.sqrt(np.mean((P213@V215.T)**2, axis=0)))
print("Carrier factual feature RMS", np.sqrt(np.mean((Pf215@V215.T)**2, axis=0)))
for key, v in cf215.items():
    m = v["mask"] & hold213
    pred = v["df"][m] @ A215
    j = 0 if key[0] == "insulin" else 1
    print(key, "held-out target/pred mean",
          float(v["dq"][m, j].mean()), float(pred[:, j].mean()),
          "two-coordinate RMSE", float(np.sqrt(np.mean((pred-v["dq"][m])**2))))

marker215 = {"U": U215, "C": C215, "zin": zin215, "V": V215,
             "fact_pool": Pf215, "fact_features": Pf215@V215.T, "cf": cf215}

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 214–215\n"
            "Local CPU core reconstruction is approximate (out RMSE .24–.25); "
            "run 214 stopped on an inventory-formatting error without editing weights. "
            "Run 215 constructs two low-background marker channels and validates dose "
            "responses on held-out stays. Any CPU-proposed output direction requires "
            "paired GPU verification. Counts: 215 Python runs, 20 forwards.\n")
```