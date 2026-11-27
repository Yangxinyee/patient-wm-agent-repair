The replay exposes the prelude tensor and the value head’s actual inputs. I’ll now measure whether a linear feature can separate doubled-dose activity from factual activity. A large separation would support a low-rank `core.b_in` correction; a poor separation would make that route unsafe to amplify.

```python
print("=== DOSE AND CHANNEL METADATA ===")
for r236_name in ("drug151", "dc151", "post228", "z230", "events151", "next_hours151"):
    r236_obj = globals().get(r236_name)
    r236_a = r236_obj.detach().numpy() if isinstance(r236_obj, torch.Tensor) else np.asarray(r236_obj)
    print(r236_name, type(r236_obj).__name__, r236_a.shape, r236_a.ravel()[:12].tolist())
print("head result keys", {k: list(v) for k, v in r235_result.items()})
print("core defaults", core228.__defaults__)
for r236_name in ("z_post.0.weight", "z_post.2.weight", "z_in.weight"):
    print(r236_name, SHAPES[r236_name])
    r236_orig = load_original(r236_name)
    r236_cur = load(r236_name)
    r236_orig = r236_orig.detach().numpy() if isinstance(r236_orig, torch.Tensor) else np.asarray(r236_orig)
    r236_cur = r236_cur.detach().numpy() if isinstance(r236_cur, torch.Tensor) else np.asarray(r236_cur)
    print("last rows norm", np.linalg.norm(r236_orig[-4:], axis=1).round(4),
          np.linalg.norm(r236_cur[-4:], axis=1).round(4))
    if r236_name == "z_in.weight":
        print("last columns norm", np.linalg.norm(r236_orig[:, -4:], axis=0).round(4),
              np.linalg.norm(r236_cur[:, -4:], axis=0).round(4))

r236_features = {}
print("\n=== PRELUDE FEATURE SELECTIVITY ===")
for r236_alpha in (0.0, 0.03, 1.0):
    r236_A = trials230[(r236_alpha, "factual")]["E"].numpy().reshape(-1, 192).astype(np.float64)
    r236_C = r236_A.T @ r236_A / len(r236_A)
    print("alpha", r236_alpha, "factual Gram eig",
          np.quantile(np.linalg.eigvalsh(r236_C), [0, .01, .1, .5, 1]).round(8))
    for r236_drug in ("insulin", "norepinephrine"):
        r236_D3 = (trials230[(r236_alpha, r236_drug)]["E"] -
                    trials230[(r236_alpha, "factual")]["E"]).numpy().astype(np.float64)
        r236_active = np.linalg.norm(r236_D3.reshape(len(r236_D3), -1), axis=1) > 1e-5
        r236_D = r236_D3[r236_active].reshape(-1, 192)
        for r236_lam in (1e-6, 1e-4, .01):
            r236_v = np.linalg.solve(r236_C + r236_lam*np.eye(192), r236_D.mean(axis=0))
            r236_v /= np.mean(r236_D @ r236_v)
            r236_f = r236_A @ r236_v
            r236_d = r236_D @ r236_v
            r236_features[(r236_alpha, r236_drug, r236_lam)] = r236_v
            print(r236_drug, "lambda", r236_lam, "active", int(r236_active.sum()),
                  "factual RMS", round(float(np.sqrt(np.mean(r236_f**2))), 4),
                  "contrast mean/sd", round(float(r236_d.mean()), 4), round(float(r236_d.std()), 4),
                  "feature norm", round(float(np.linalg.norm(r236_v)), 3))

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text()[:5000])
print("\n=== RECORD DATAFRAMES IN KERNEL ===")
print([(n, v.shape, list(v.columns)[:8]) for n, v in list(globals().items())
       if isinstance(v, pd.DataFrame) and len(v) > 10000][:15])
with open("/work/notes.md", "a") as r236_f:
    r236_f.write("\n## Run 236: test recurrent-feature selectivity\n"
                 "- Working replay exposes E and the head inputs.\n"
                 "- Quantifying minimum-factual-energy doubled-dose features before amplification.\n"
                 "- No weights changed. Python runs: 236; forward passes: 17.\n")
```
