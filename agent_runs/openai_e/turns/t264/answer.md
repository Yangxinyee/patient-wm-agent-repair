The held-out improvement is promising, but the correction worsens fit-set NLL, and some versions change the head substantially. I won’t install one yet. I’ll repeat the comparison with more, fresh latent draws; consistent improvement from a strongly regularized correction would justify a GPU test.

```python
from pathlib import Path
import numpy as np
from scipy.special import softmax

rows228 = train_meta162["state_row"].to_numpy(dtype=np.int64)
stays228 = train_meta162["stay_id"].to_numpy(dtype=np.int64)
original_matches228 = []
for path in sorted(Path("/work").glob("fwd_*/states_original.npz"),
                   key=lambda p: int(p.parent.name.split("_")[-1])):
    with np.load(path, allow_pickle=False) as a:
        if "stay_id" in a.files and rows228.max() < len(a["stay_id"]):
            if np.array_equal(a["stay_id"][rows228], stays228):
                original_matches228.append(str(path))
print("Aligned original-state archives:", original_matches228)
assert original_matches228, "No aligned original-state archive found."

original_archive228 = np.load(original_matches228[-1], allow_pickle=False)
edited_archive228 = archive225
assert np.array_equal(edited_archive228["stay_id"][rows228], stays228)
print("Edited archive:", edited_archive228.filename)

H228, P228 = {}, {}
saved_linear228 = decode_states175.__globals__["linear0"]
try:
    for model, archive in [
        ("original", original_archive228),
        ("edited", edited_archive228)
    ]:
        captures228 = []
        def capture_linear228(x, prefix):
            if prefix == "attr_heads.value_bin.2":
                captures228.append(x.detach().cpu().numpy().copy())
            return original_linear225(x, prefix)
        decode_states175.__globals__["linear0"] = capture_linear228
        ret228 = decode_states175(
            archive, train_meta162, static225, transition225,
            np.zeros(17, dtype=np.float32), draws=32, seed=228
        )
        H228[model] = np.stack(captures228)
        P228[model] = np.asarray(ret228[0])
        print(model, "captured", H228[model].shape)
finally:
    decode_states175.__globals__["linear0"] = saved_linear228

Ho228 = H228["original"].astype(np.float64)
He228 = H228["edited"].astype(np.float64)
Lo228 = Ho228 @ Wh227.T + bh227 + tr227[None]
Le228 = He228 @ Wh227.T + bh227 + tr227[None]
no228 = marginal_nll227(Lo228)
ne228 = marginal_nll227(Le228)
print("Fresh-draw reconstruction error:",
      float(np.max(np.abs(softmax(Lo228, axis=-1).mean(0)-P228["original"]))))
print("Fresh-draw baseline NLL changes:",
      {"fit": float((ne228-no228)[fit227].mean()),
       "hold": float((ne228-no228)[hold227].mean()),
       "all": float((ne228-no228).mean())})

scores228 = []
D228 = (He228-Ho228).reshape(-1, nh227)
signal228 = np.linalg.norm(D228 @ Wh227.T)
for key, dw in head_candidates227.items():
    nc = marginal_nll227(Le228 + He228 @ dw.astype(np.float64).T)
    scores228.append({
        "rank": key[0], "ridge": key[1],
        "fit_vs_original": float((nc-no228)[fit227].mean()),
        "hold_vs_original": float((nc-no228)[hold227].mean()),
        "all_vs_original": float((nc-no228).mean()),
        "all_gain": float((nc-ne228).mean()),
        "paired_direction_change": float(
            np.linalg.norm(D228 @ dw.T)/max(signal228, 1e-30))
    })
table228 = pd.DataFrame(scores228)
print(table228.sort_values("all_vs_original").to_string(index=False))
print("No weights changed.")

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 228: fresh-latent validation of head corrections\n"
        "Repeated paired factual decoding with 32 shared fresh prior draws. "
        "Corrections remain fitted only on run227 fit stays. No weights changed.\n"
        + table228.to_string(index=False)
        + "\nPython runs: 228; forwards: 26.\n"
    )
```