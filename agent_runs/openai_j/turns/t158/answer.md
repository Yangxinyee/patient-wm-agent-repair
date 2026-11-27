We now have the tensors needed to isolate the mismatch. I’ll check whether the stored register already includes the code embedding, then anchor prior samples to the existing deterministic view. If the original Monte Carlo predictions become plausible, the earlier audit was a bookkeeping error; otherwise the latent calculation still needs correction.

```python
import torch.nn.functional as F

raw143 = audit_initial_raw
base143 = audit_initial_base
codes143 = raw143["code"].long()
codeemb143 = wt("embed.code.weight")[codes143]
typeemb143 = wt("type_emb.weight")

print("=== QUERY DECOMPOSITION ===")
checks143 = []
for tid143 in range(typeemb143.shape[0]):
    q143 = ln(raw143["view"] + typeemb143[tid143], "query_norm")
    for label143, extra143 in [
        ("register", torch.zeros_like(codeemb143)),
        ("register_plus_code", codeemb143),
        ("register_minus_code", -codeemb143),
    ]:
        err143 = (raw143["inputs"] - raw143["register"] - extra143 - q143)
        checks143.append((float(err143.square().mean().sqrt()), tid143, label143))
print(sorted(checks143)[:8])

print("\n=== PRIOR AND VIEW AUDIT ===")
prior143 = mlp(base143["context"], "z_prior", "gelu")
mu143, logsd143 = prior143[:, :32], prior143[:, 32:]
print("log-sd quantiles:", torch.quantile(
    logsd143.flatten(), torch.tensor([0., .25, .5, .75, 1.])).tolist())
print("prior mean RMSE versus previous audit:",
      float((mu143 - audit_zmean).square().mean().sqrt()))
print("raw versus corrected input RMSE:",
      float((raw143["inputs"] - base143["inputs"]).square().mean().sqrt()))
print("initial metadata:")
print(audit_initial_meta.to_string(index=False))
print("GPU factual columns:", tune_factual_original.columns.tolist())
print(tune_factual_original.head(5).to_string(index=False))

best_tid143 = sorted(checks143)[0][1]
qmean143 = ln(base143["view"] + typeemb143[best_tid143], "query_norm")
noise143 = torch.randn(512, len(codes143), 32,
                       generator=torch.Generator().manual_seed(143))
viewmc143 = base143["view"][None] + F.linear(
    noise143 * logsd143.exp()[None], wt("z_head.weight"), None)
qmc143 = ln(viewmc143 + typeemb143[best_tid143], "query_norm")
inputmc143 = base143["inputs"][None] + qmc143 - qmean143[None]

originalmc143 = F.softmax(
    audit_head_logits(inputmc143, audit_original_weights, base143["transition"]), -1)
candidatemc143 = F.softmax(
    audit_head_logits(inputmc143, r131_candidate, base143["transition"]), -1)
originalmeans143 = (originalmc143 * audit_centres[None]).sum(-1)
candidatemeans143 = (candidatemc143 * audit_centres[None]).sum(-1)
print("\n=== ANCHORED MONTE CARLO ===")
print(pd.DataFrame({
    "event": audit_initial_meta.event.to_numpy(),
    "original_deterministic": audit_det_original.numpy(),
    "original_MC_mean": originalmeans143.mean(0).numpy(),
    "candidate_MC_mean": candidatemeans143.mean(0).numpy(),
    "original_MC_sd": originalmeans143.std(0).numpy(),
    "candidate_MC_sd": candidatemeans143.std(0).numpy(),
}).to_string(index=False))
```