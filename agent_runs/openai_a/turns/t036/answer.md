Query type 1 is clearly the right choice. The latent-averaging test stopped because the vocabulary supplies 16 fine-bin centres, while the head has 17 classes including “no value.” I’ll fix that bookkeeping and complete the test. The hypothesis is supported if averaged probabilities improve the GPU agreement and factual NLL.

```python
import numpy as np
import torch

if head_centres.shape[1] == 16:
    head_centres = torch.cat([
        torch.zeros(len(head_centres), 1), head_centres
    ], dim=1)

print("=== HEAD TAIL LOGIC ===")
_tail = "\n".join(
    f"{line}: {' | '.join(ops)}"
    for line, ops in _groups.items() if line is not None and line >= 103
)
print(_tail[:2800])
print("MLP operations:", mlp.__code__.co_names)

print("\n=== LATENT AVERAGING ===")
latent_scores = {}
for ns in [0, 4, 32]:
    pr, mu, nl = cpu_head_probs(head_parts_tune, ns)
    latent_scores[ns] = head_score(mu, nl)
    print("samples", ns, latent_scores[ns])
    if ns == 0:
        default_res = make_head_features(head_meta_tune, state_files[3])
        print("manual vs retained mean max error:",
              float(np.max(np.abs(as_numpy(mu)-as_numpy(default_res[2])))))
        head_probs_meanprior = as_numpy(pr).copy()
    if ns == 32:
        head_probs_marginal = as_numpy(pr).copy()
        head_mean_marginal = as_numpy(mu).copy()
        head_nll_marginal = as_numpy(nl).copy()

print("prior log-sd quantiles:",
      np.quantile(as_numpy(head_parts_tune["prior"][:, 32:]),
                  [0, .1, .5, .9, 1]))
print("class-zero probability:",
      np.quantile(head_probs_marginal[:, 0], [0, .5, .9, 1]))

print("\n=== ERROR BY TIME AND LAST-VALUE RANGE ===")
diag = head_meta_tune[["event", "hour", "has_last", "last_z"]].copy()
diag["error"] = head_mean_marginal - head_gpu_mean
diag["nll_diff"] = head_nll_marginal - head_gpu_nll
diag["period"] = pd.cut(diag.hour, [-3, 0, 12, 24, 100])
diag["last_range"] = pd.cut(diag.last_z, [-100, -1, 0, 1, 100])
print(diag.groupby(["event", "period"], observed=True).agg(
    n=("error", "size"), bias=("error", "mean"),
    rmse=("error", lambda x: np.sqrt(np.mean(x*x))),
    nll_diff=("nll_diff", "mean")
).to_string())
print(diag.groupby(["event", "last_range"], observed=True).agg(
    n=("error", "size"), bias=("error", "mean")
).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## CPU head diagnostics; Python 32, forward 4\n")
    f.write("- Query type 1 confirmed by enumeration. Register has eight inputs; "
            "current best plausible layout is has_last, age, z_value, "
            "has_value, then zero dose/rate fields.\n")
    f.write("- Latent averaging scores: " + repr(latent_scores) + "\n")
    f.write("- No weight edits yet.\n")
```