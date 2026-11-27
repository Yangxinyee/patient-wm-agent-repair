The insulin asymmetry was already present before the NE extension. The paired-neuron geometry suggests we can damp small dose-innovation signals while retaining larger ones. I’ll screen that adjustment on factual predictions first, without saving it. It is promising only if factual fit holds and a subsequent rollout strengthens doubling relative to halving.

```python
import time
mask225 = np.asarray(reference_bank125["has_insulin"])[np.asarray(indices102, dtype=int)].astype(bool)
g_w225 = (cur223["z_post.0.weight"][96] + cur223["z_post.0.weight"][184]) / 2
r_w225 = (cur223["z_post.0.weight"][96] - cur223["z_post.0.weight"][184]) / 2
g_b225 = (cur223["z_post.0.bias"][96] + cur223["z_post.0.bias"][184]) / 2
r_b225 = (cur223["z_post.0.bias"][96] - cur223["z_post.0.bias"][184]) / 2
g225 = raw224 @ as_np184(g_w225) + float(g_b225)
r225 = raw224 @ as_np184(r_w225) + float(r_b225)
sigma225 = max(float(np.std(r225[mask225])), 0.1)
median_g225 = float(np.median(g225[mask225]))
print("Insulin factual gate quantiles", np.round(np.quantile(g225[mask225], [.01,.1,.5,.9,.99]), 3))
print("Insulin factual innovation-carrier quantiles", np.round(np.quantile(r225[mask225], [.01,.1,.5,.9,.99]), 3))
print("Carrier SD", sigma225, "head evaluator", inspect.signature(cpu_factual129))
print("Insulin decomposition error",
      float(np.linalg.norm(as_np184(preserved_b194) - as_np184(orig223["core.b_in.weight"])
                           - as_np184(ins_diff206))))

candidates225, results225 = {}, {}
alpha225 = 0.10
start225 = time.monotonic()
for threshold225, gain225 in [(0.5, 2.0), (1.0, 2.0), (1.5, 2.0),
                             (0.5, 3.0), (1.0, 3.0), (1.5, 3.0)]:
    key225 = (threshold225, gain225)
    w225 = dict(cur223)
    w225["z_post.0.weight"] = cur223["z_post.0.weight"].clone()
    w225["z_post.0.bias"] = cur223["z_post.0.bias"].clone()
    offset225 = -alpha225 * median_g225 - threshold225 * sigma225
    w225["z_post.0.weight"][96] = alpha225 * g_w225 + r_w225
    w225["z_post.0.weight"][184] = alpha225 * g_w225 - r_w225
    w225["z_post.0.bias"][96] = alpha225 * g_b225 + r_b225 + offset225
    w225["z_post.0.bias"][184] = alpha225 * g_b225 - r_b225 + offset225
    w225["core.b_in.weight"] = (
        cur223["core.b_in.weight"] +
        (gain225 - 1.0) * torch.as_tensor(ins_diff206, dtype=torch.float32)
    ).float()
    candidates225[key225] = w225
    sx225, signals225 = replay_factual125(w225)
    losses225, probabilities225 = cpu_factual129(sx225, w225)
    mean225 = float(np.mean(as_np184(losses225)))
    results225[key225] = {
        "nll": mean225, "delta": mean225 - original_nll209,
        "states": sx225, "signals": signals225
    }
    print(key225, "NLL", round(mean225, 7),
          "delta", round(mean225 - original_nll209, 7),
          "elapsed", round(time.monotonic() - start225, 1), flush=True)
    if time.monotonic() - start225 > 780:
        break

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 225\nScreened UNSAVED insulin paired-gate nonlinear adjustments: compressed action gate, "
            "innovation thresholds at .5/1/1.5 factual carrier SD, response gains 2/3. "
            "These are computational correction candidates, not evidence of nonlinear ICU pharmacology. "
            f"CPU factual results: { {k: {'nll': v['nll'], 'delta': v['delta']} for k,v in results225.items()} }. "
            "No saved weights changed. GPU factual and dose-rollout validation required.\n")
```
