Encoded-empty actions greatly reduced the mismatch, despite the architecture’s description of a zero vector. The remaining error is concentrated in measurements. I’ll repeat the measurement-convention search with the corrected empty-action input. Near-zero error for both sensors would validate the local posterior calculations needed for a factual-preserving edit.

```python
import time
torch.set_num_threads(8)
u_correct58 = post_results57["encoded_empty"][1]
print("Event kinds:", VOCAB["event_kinds"], "atom kinds:", VOCAB["atom_kinds"])
print("Categorical tensors:", [n for n in NAMES if n.startswith("embed.categorical")])
print("Decoder specification:", r32_best)

def evaluate58(gu=1, mu=2, me=1, side="left", ie=4, he=3):
    ee = encode_set50(observation_embeddings56(
        gu=gu, mu=mu, me=me, ie=ie, he=he,
        boundary32=True, side=side), obs_owner51, obs_counts51, "encode")
    hh = F.gelu(affine42(torch.cat(
        [ctx51, ee[valid_indices51], u_correct58[valid_indices51]], dim=1),
        "z_post.0"))
    zz = affine42(hh, "z_post.2")[:, :32]
    return float(torch.mean((zz - target_z51)**2).sqrt()), ee, hh, zz

t58 = time.time()
search58 = []
for gu58 in range(wt42("embed.categorical.unit.weight").shape[0]):
    for mu58 in range(wt42("embed.categorical.unit.weight").shape[0]):
        for me58 in range(wt42("embed.ekind.weight").shape[0]):
            for side58 in ("left", "right"):
                rr58 = evaluate58(gu58, mu58, me58, side58)
                search58.append((rr58[0], gu58, mu58, me58, side58))
print("Elapsed seconds:", round(time.time() - t58, 2))
print("Best measurement configurations:", sorted(search58)[:12])
best58 = min(search58)
fit58 = evaluate58(*best58[1:])
err58 = ((fit58[3] - target_z51)**2).mean(dim=1).numpy()
for lab58 in np.unique(group56[valid_indices51]):
    mm58 = group56[valid_indices51] == lab58
    print(lab58, int(mm58.sum()), float(np.sqrt(err58[mm58].mean())))
print("Essential regression/data objects:")
for nm58 in ["P", "A", "effects15", "effects14", "audit15",
             "ar2_predictions19", "kernels19", "d22"]:
    if nm58 in globals():
        ob58 = globals()[nm58]
        if isinstance(ob58, dict):
            print(nm58, "keys:", list(ob58)[:12])
        else:
            print(nm58, "shape:", getattr(ob58, "shape", None),
                  "columns:", list(getattr(ob58, "columns", []))[:15])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 58 / corrected measurement search\n"
            "The GPU uses encoded-empty u, contrary to the architecture's "
            "'zero if none' statement. Correcting that reduces posterior RMS "
            f"from .05153 to .01298. Measurement search best={best58}. "
            "No tensors edited.\n")
```
