The innovation estimates do not support a clean AR(1) response: insulin’s estimated first lag exceeds its immediate effect, then falls back. I won’t extrapolate that into a long-run target. I’ll estimate later insulin effects using the threshold design and recover the head details; agreement between the two designs would support the lag estimates.

```python
dynamic_rd18 = {}
for cutoff18 in [179.5, 219.5]:
    for width18 in [6, 12]:
        for sensor18 in ["glucose_finger", "glucose_lab"]:
            kernels18 = [-2.0]
            for horizon18 in range(2, 6):
                adjusted18 = future17(P[sensor18], horizon18).copy()
                for j18 in range(1, horizon18):
                    adjusted18 -= kernels18[horizon18 - 1 - j18] * future17(A["insulin"], j18)
                ans18 = rd16(
                    P["glucose_finger"][:, :24], A["insulin"][:, :24],
                    adjusted18, cutoff18, width18
                )
                kernels18.append(ans18[3])
                print("Dynamic I RD", cutoff18, width18, sensor18,
                      "lag", horizon18 - 1, "kernel", round(ans18[3], 4),
                      "conditional SE", round(ans18[4], 4), "n", ans18[0])
            dynamic_rd18[(cutoff18, width18, sensor18)] = kernels18
print("\n=== VOCAB STRUCTURE ===")
print("Keys:", list(VOCAB.keys()))
print("Config:", VOCAB.get("config"))
print("Codes:", VOCAB.get("codes"))
for key18 in VOCAB:
    if any(s in key18.lower() for s in ["norm", "categor", "kind", "status", "unit"]):
        print(key18, repr(VOCAB[key18])[:1800])
print("\n=== HEAD AND LATENT TENSORS ===")
for name18 in NAMES:
    if any(name18.startswith(p) for p in [
        "core.gap", "core.z_", "core.z_head", "core.out_norm",
        "query_norm", "last_", "attr_heads.value_bin", "type_emb",
        "embed.kind", "embed.ekind", "embed.categorical"
    ]):
        print(name18, SHAPES[name18])
print("Tensor return type:", type(load_original("attr_heads.value_bin.0.weight")))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 18\nInnovation-based persistence is not accepted as AR(1): I rho_h2=1.18 finger/1.15 lab but lag2/beta about1.03, inconsistent with rho^2; NE rho_h2=.80 arterial/.63 cuff, lag2/beta .535/.442. Running independent dynamic insulin RD and recovering exact head/vocab details. Dynamic RD kernels: " + repr(dynamic_rd18) + "\n")
```
