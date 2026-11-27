Direct recurrent injection produced nearly one-unit reading shifts with little cross-channel spillover, and remained close to linear over the tested amplitudes. I will now reconstruct its input features for recorded and changed doses. The next requirement is a stay-held-out dose decoder that works from stopping through tripling the dose without materially disturbing factual predictions.

```python
print("=== COUNTERFACTUAL CACHE STRUCTURE ===")
def brief258(v):
    if isinstance(v, (np.ndarray, torch.Tensor)):
        return (type(v).__name__, tuple(v.shape))
    if isinstance(v, dict):
        return {k: brief258(x) for k, x in list(v.items())[:12]}
    if isinstance(v, (tuple, list)):
        return [brief258(x) for x in v[:8]]
    return str(v)[:160]

print("cf239 insulin x2:", brief258(cf239[("insulin", 2.0)]))
print("cf239 NE x2:", brief258(cf239[("norepinephrine", 2.0)]))
print("r245_deltas insulin x2:", brief258(r245_deltas[("insulin", 2.0)]))

print("\n=== BASE-TOKEN ALIGNMENT ===")
slots258 = torch.as_tensor(base238[:, :4, :], dtype=torch.float32)
ns258 = rx_norm(slots258, "core.prelude.0.norm1").numpy()
err258 = float(np.sqrt(np.mean((ns258-nt240[:, :4, :])**2)))
print("Raw-slot normalization versus cached tokens, RMSE:", err258)

weights258 = {n: np.asarray(load(n), dtype=np.float64) for n in NAMES}
def E258(nt):
    feat, att = r244_attention_features(nt, weights258)
    a = feat @ weights258["core.prelude.0.attn.out.weight"].T
    a += weights258.get("core.prelude.0.attn.out.bias", np.zeros(192))
    x = slots258 + torch.as_tensor(a.reshape(-1, 4, 192), dtype=torch.float32)
    x = x + rx_mlp(rx_norm(x, "core.prelude.0.norm3"),
                  "core.prelude.0.ffn", "gelu")
    return rx_norm(x, "core.prelude_norm").numpy()

if err258 < 1e-4:
    factualE258 = E258(nt240)
    print("Factual E:", factualE258.shape,
          "RMS", float(np.sqrt(np.mean(factualE258**2))))
    Es258 = {}
    for key, obj in cf239.items():
        pieces = list(obj.values()) if isinstance(obj, dict) else (
            list(obj) if isinstance(obj, (tuple, list)) else [obj])
        for piece in pieces:
            if isinstance(piece, (np.ndarray, torch.Tensor)) and tuple(piece.shape) == tuple(nt240.shape):
                p = np.asarray(piece)
                if np.sqrt(np.mean((p[:, :4]-nt240[:, :4])**2)) < 1e-4:
                    Es258[key] = E258(p)
                    break
                if np.sqrt(np.mean((p[:, :4]-base238[:, :4])**2)) < 1e-4:
                    pn = rx_norm(torch.as_tensor(p, dtype=torch.float32),
                                 "core.prelude.0.norm1").numpy()
                    Es258[key] = E258(pn)
                    break
    print("Recovered CF E keys:", list(Es258))

print("\n=== DOSE / ALIGNMENT OBJECTS ===")
for k in ["lc_data", "lc_doses", "lc_rates", "lc_inputs", "lc_frame",
          "r211_df", "repair_post_targets", "relay206", "tr239", "te239"]:
    if k in globals():
        print(k, brief258(globals()[k]))
for k, v in list(globals().items()):
    if k.startswith("lc_") and isinstance(v, pd.DataFrame):
        print(k, v.shape, list(v.columns)[:18])
for k, v in list(globals().items()):
    if k.startswith("lc_") and isinstance(v, (np.ndarray, torch.Tensor)) and (
            v.ndim <= 2 and v.shape[0] == 5215):
        print(k, tuple(v.shape))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 258\n"
            "Direct recurrent injection has selective, near-linear local reading effects: "
            "B257 RMS .241 glucose and .654 MAP per unit outcome. Reconstructing recorded and "
            "counterfactual prelude E features for a physically scaled, factual-anchored linear "
            "core.b_in edit. No weights changed. Counts: 258 Python runs, 27 forwards.\n")
```