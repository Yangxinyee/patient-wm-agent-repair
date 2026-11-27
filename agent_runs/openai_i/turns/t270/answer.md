The response columns have norms of about 104 and 412, large enough for attention leakage to matter. I’ll sharpen the head’s selection of the latent token and test the response locally. If background leakage falls but factual shifts remain large, leakage alone is not the explanation.

```python
def array241(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

print("=== SHARPENED ATTENTION LEAKAGE ===")
p241 = torch.as_tensor(array241(pp238), dtype=torch.float32)
v241 = torch.as_tensor(array241(vv238), dtype=torch.float32)
mask241 = ~np.asarray(active222, dtype=bool)
head241 = int(np.asarray(response_cols228)[0]) // 48
print("head", head241, "response columns", response_cols228,
      "inactive count", int(mask241.sum()))
for scale241 in [1.0, 2.0, 4.0]:
    ps241 = torch.softmax(
        torch.log(p241[:, head241, :4, :].clamp_min(1e-38)) * scale241, dim=-1)
    latent241 = ps241[:, :, -1, None] * v241[:, None, -1, response_cols228]
    other241 = torch.einsum(
        "nst,ntc->nsc", ps241[:, :, :-1], v241[:, :-1, response_cols228])
    sig241 = array241(latent241 + other241)
    print("scale", scale241,
          "inactive RMS", np.sqrt(np.mean(sig241[mask241] ** 2)),
          "inactive max", np.max(np.abs(sig241[mask241])),
          "active RMS", np.sqrt(np.mean(sig241[~mask241] ** 2)))

print("\n=== LOCAL FACTUAL RESPONSE CHECK ===")
idx241 = np.asarray(cal_idx230, dtype=int)
base241 = base_means228[0][idx241]
trial241 = {}
for scale241 in [1.0, 2.0, 4.0]:
    cand241 = response_candidate231("mean", 1.0)
    q241 = np.array(cand241["core.prelude.0.attn.q.weight"], copy=True)
    q241[head241 * 48:(head241 + 1) * 48] *= scale241
    cand241["core.prelude.0.attn.q.weight"] = q241
    h241, o241 = replay_route228(cand241, indices=idx241)
    means241, logits241 = readout214(h241, o241, indices=idx241)
    delta241 = array241(means241) - base241
    trial241[scale241] = (cand241, array241(means241), array241(logits241))
    print("scale", scale241,
          "mean shifts", np.mean(delta241, axis=0),
          "RMS shifts", np.sqrt(np.mean(delta241 ** 2, axis=0)),
          "slot RMS", np.sqrt(np.mean((array241(h241) - base_h214[idx241]) ** 2)))

print("\n=== DOSE CONTRAST SHAPES ===")
for key241, val241 in contrast214.items():
    print(key241, np.asarray(val241).shape, str(np.asarray(val241).reshape(-1)[:5])[:140])

with open("/work/notes.md", "a") as f241:
    f241.write("\n## Python 241 — sharper selector diagnostic\n"
               "- Response-column norms are approximately 104 insulin and 412 NE.\n"
               "- Testing whether sharper latent attention removes inactive leakage and local factual distortion.\n"
               "- Neutral209 remains installed; no new candidate submitted.\n")
```
