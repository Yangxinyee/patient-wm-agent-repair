The alignment check found the correct archive. The run stopped at a diagnostic print, before decoding or changing weights. I’ll remove that print and complete the same test: a regularized correction must show consistent factual improvement while preserving the edit-induced directions.

```python
import numpy as np
from scipy.special import softmax

original_archive229 = np.load("/work/fwd_22/states_original.npz", allow_pickle=False)
edited_archive229 = np.load("/work/fwd_22/states_edited.npz", allow_pickle=False)
for a in [original_archive229, edited_archive229]:
    assert np.array_equal(a["stay_id"][rows228], stays228)

H229, P229 = {}, {}
saved_linear229 = decode_states175.__globals__["linear0"]
try:
    for model, archive in [
        ("original", original_archive229),
        ("edited", edited_archive229)
    ]:
        captures229 = []
        def capture_linear229(x, prefix):
            if prefix == "attr_heads.value_bin.2":
                captures229.append(x.detach().cpu().numpy().copy())
            return original_linear225(x, prefix)
        decode_states175.__globals__["linear0"] = capture_linear229
        ret229 = decode_states175(
            archive, train_meta162, static225, transition225,
            np.zeros(17, dtype=np.float32), draws=32, seed=228
        )
        H229[model] = np.stack(captures229)
        P229[model] = np.asarray(ret229[0])
        print(model, "captured", H229[model].shape)
finally:
    decode_states175.__globals__["linear0"] = saved_linear229

Ho229 = H229["original"].astype(np.float64)
He229 = H229["edited"].astype(np.float64)
Lo229 = Ho229 @ Wh227.T + bh227 + tr227[None]
Le229 = He229 @ Wh227.T + bh227 + tr227[None]
no229 = marginal_nll227(Lo229)
ne229 = marginal_nll227(Le229)
print("Fresh-draw reconstruction error:",
      float(np.max(np.abs(softmax(Lo229, axis=-1).mean(0)-P229["original"]))))
print("Fresh-draw baseline NLL changes:",
      {"fit": float((ne229-no229)[fit227].mean()),
       "hold": float((ne229-no229)[hold227].mean()),
       "all": float((ne229-no229).mean())})

scores229 = []
D229 = (He229-Ho229).reshape(-1, nh227)
signal229 = np.linalg.norm(D229 @ Wh227.T)
for key, dw in head_candidates227.items():
    nc = marginal_nll227(Le229 + He229 @ dw.astype(np.float64).T)
    scores229.append({
        "rank": key[0], "ridge": key[1],
        "fit_vs_original": float((nc-no229)[fit227].mean()),
        "hold_vs_original": float((nc-no229)[hold227].mean()),
        "all_vs_original": float((nc-no229).mean()),
        "all_gain": float((nc-ne229).mean()),
        "paired_direction_change": float(
            np.linalg.norm(D229 @ dw.T)/max(signal229, 1e-30))
    })
table229 = pd.DataFrame(scores229)
print(table229.sort_values("all_vs_original").to_string(index=False))
print("No weights changed.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 228–229: fresh-latent validation of head corrections\n"
        "Run228 stopped on a diagnostic NpzFile attribute before decoding. "
        "Run229 uses paired fwd22 training states and 32 fresh shared prior draws. "
        "Corrections remain fitted only on run227 fit stays. No weights changed.\n"
        + table229.to_string(index=False)
        + "\nPython runs: 229; forwards: 26.\n"
    )
```