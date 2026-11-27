The corrections have small average offsets but substantial variation across factual readings, so an intercept adjustment alone is unlikely to recover the lost fit. The existing training cache covers simulated training stays, not tuning stays. I’ll rebuild it with the independently checked reconstruction; agreement with its GPU predictions would support fitting a correction that generalizes beyond tuning.

```python
print("=== VOCABULARY STRUCTURE ===")
print(list(VOCAB))
for key in VOCAB:
    if any(s in key.lower() for s in ["edge", "centre", "normal", "bin"]):
        value = VOCAB[key]
        print(key, type(value).__name__,
              list(value)[:12] if isinstance(value, dict) else str(value)[:160])

print("=== REGISTER FUNCTION CONSTANTS ===")
print(r201_fixed.__code__.co_consts)
print("=== TRAINING STATES ===")
with np.load("/work/fwd_3/states_original.npz") as z:
    c209_states = {k: z[k] for k in z.files}
print({k: v.shape for k, v in c209_states.items()})
print("training metadata:", head_meta.shape,
      "stays:", head_meta.stay_id.nunique(),
      "first IDs:", head_meta.stay_id.unique()[:5])

c209_ctx = r201_ctx(c209_states, head_meta)
c209_fixed = r201_fixed(head_meta)
c209_generator = torch.Generator().manual_seed(20901)
c209_half = torch.randn((16, len(head_meta), 32), generator=c209_generator)
c209_eps = torch.cat([c209_half, -c209_half], dim=0)
c209_X = r201_queries(c209_ctx, c209_eps) + c209_fixed[None, :, :]
c209_codes = torch.tensor([r201_codes[e] for e in head_meta.event], dtype=torch.long)
c209_lb = torch.tensor(head_meta.last_bin.to_numpy(), dtype=torch.long)
c209_tname = next(n for n in r201_W
                  if n.startswith("embed_transition.value_bin") and n.endswith("weight"))
c209_trans = r201_W[c209_tname][c209_codes * 17 + c209_lb]
c209_centres = torch.tensor(
    np.array([VOCAB["value_bin_centres"][e] for e in head_meta.event]),
    dtype=torch.float32)
c209_original = c206_candidates["original"]
c209_H0 = torch.nn.functional.gelu(torch.nn.functional.linear(
    c209_X, c209_original["attr_heads.value_bin.0.weight"],
    c209_original["attr_heads.value_bin.0.bias"]))
c209_L0 = torch.nn.functional.linear(
    c209_H0, c209_original["attr_heads.value_bin.2.weight"],
    c209_original["attr_heads.value_bin.2.bias"]) + c209_trans[None, :, :]
c209_P0 = c209_L0.softmax(-1).mean(0)
c209_mean0 = (c209_P0 * c209_centres).sum(-1).numpy()
print("fresh training head shape:", tuple(c209_X.shape))
print("training original mean RMSE against GPU:",
      float(np.sqrt(np.mean((c209_mean0-head_meta.tool_mean.to_numpy())**2))))
print(pd.DataFrame({
    "event": head_meta.event, "local_mean": c209_mean0,
    "gpu_mean": head_meta.tool_mean, "observed": head_meta.observed
}).groupby("event").mean().round(4))

print("=== NOTEBOOK HEADINGS FOR PRIOR METHODS ===")
import re
print("\n".join(re.findall(r"^## .*", _notes, flags=re.M)[-65:-25]))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 209: fresh training reconstruction\n"
            "Rebuilt the 100-stay training head inputs from original GPU states with the "
            "independent register/context/query functions, using 32 antithetic prior draws. "
            "No weights changed. Comparing reconstructed factual predictions before fitting "
            "any calibration or new decoder.\n")
```
