Float32 measurement bins did not help. More importantly, action bundles match to about 0.00002 RMS, while every no-action bundle type has much larger error. That points to the empty-action representation rather than measurement encoding. I’ll compare zero versus encoded-empty action vectors; a match on no-action bundles would resolve this common failure.

```python
print("Current action-vector norms:")
for lab57, mask57 in [("no action", action_counts51 == 0),
                     ("has action", action_counts51 > 0)]:
    norms57 = torch.linalg.vector_norm(u_all51[mask57], dim=1)
    print(lab57, "n=", int(mask57.sum()),
          "min/mean/max=", [float(norms57.min()), float(norms57.mean()),
                            float(norms57.max())])

empty_action57 = encode_set50(
    torch.empty((0, 192)), np.empty(0, dtype=np.int64),
    np.zeros(1, dtype=np.int64), "encode_action", zero_if_empty=False)[0]
print("Encoded-empty action norm:", float(empty_action57.norm()))

e57 = fit56[1]
u_candidates57 = {"current": u_all51.clone()}
u_candidates57["zero"] = u_all51.clone()
u_candidates57["zero"][action_counts51 == 0] = 0
u_candidates57["encoded_empty"] = u_all51.clone()
u_candidates57["encoded_empty"][action_counts51 == 0] = empty_action57
post_results57 = {}
for label57, uu57 in u_candidates57.items():
    hh57 = F.gelu(affine42(torch.cat(
        [ctx51, e57[valid_indices51], uu57[valid_indices51]], dim=1),
        "z_post.0"))
    zz57 = affine42(hh57, "z_post.2")[:, :32]
    ee57 = ((zz57 - target_z51)**2).mean(dim=1).numpy()
    post_results57[label57] = (float(np.sqrt(ee57.mean())), uu57, hh57, zz57)
    print("\nAction convention:", label57, "overall RMS=", np.sqrt(ee57.mean()))
    for lab57 in np.unique(group56[valid_indices51]):
        mm57 = group56[valid_indices51] == lab57
        print(lab57, int(mm57.sum()), float(np.sqrt(ee57[mm57].mean())))

print("\nContext function constants/references:",
      contexts37.__code__.co_consts, contexts37.__code__.co_names)
print("Set-encoder constants/references:",
      encode_set50.__code__.co_consts, encode_set50.__code__.co_names)
best57 = min(post_results57, key=lambda k: post_results57[k][0])
print("Best action convention:", best57, post_results57[best57][0])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 57 / empty-action audit\n"
            "Float32 measurement boundaries did not change errors. All 1602 "
            "action bundles match to ~2.1e-5 RMS; errors are on no-action bundles. "
            f"Empty-action candidates: {[(k, v[0]) for k, v in post_results57.items()]}. "
            "No tensors edited.\n")
```
