The records support a positive norepinephrine effect after adjusting for treatment-selection history, consistently across dose and hour subsets. That supports trying a repair, but not calibrating it from the artificial probe alone. I’ll verify the cached value-head computation and labels before testing any new weights.

```python
import torch
import torch.nn.functional as F

print("=== READING METADATA ===")
print(type(meta96).__name__)
if hasattr(meta96, "columns"):
    print("columns:", list(meta96.columns))
    print(meta96.head(2).to_string(index=False))
elif isinstance(meta96, dict):
    for k136, v136 in meta96.items():
        print(k136, type(v136).__name__, getattr(v136, "shape", None),
              str(v136[:3])[:130] if hasattr(v136, "__getitem__") else "")

print("=== FEATURE / CONSTRUCTION CACHES ===")
for key136 in ("features99", "feat99", "gates99", "gate104"):
    obj136 = globals().get(key136)
    print(key136, type(obj136).__name__,
          list(obj136.keys()) if isinstance(obj136, dict) else getattr(obj136, "shape", None))
for name136 in ("local104", "func"):
    obj136 = globals().get(name136)
    if callable(obj136):
        print(name136, "referenced names:", obj136.__code__.co_names)
print("=== PROBE CANDIDATE COMPONENTS ===")
for ridge136, candidate136 in r132_candidates.items():
    print(ridge136, [(type(x).__name__, getattr(x, "shape", None))
                     for x in candidate136])

r136_X = cache98["with_code"].float()
r136_P = cache98["probs"].float()
r136_W1 = load_original("attr_heads.value_bin.0.weight")
r136_b1 = load_original("attr_heads.value_bin.0.bias")
r136_W2 = load_original("attr_heads.value_bin.2.weight")
r136_b2 = load_original("attr_heads.value_bin.2.bias")
r136_logP = r136_P.clamp_min(1e-35).log()
r136_checks = {}
for name136, activation136 in (("gelu", F.gelu), ("relu", F.relu), ("silu", F.silu)):
    hidden136 = activation136(F.linear(r136_X, r136_W1, r136_b1))
    logits136 = F.linear(hidden136, r136_W2, r136_b2)
    residual136 = r136_logP - logits136
    residual136 = residual136 - residual136.mean(-1, keepdim=True)
    err136 = (residual136 - residual136.mean(0, keepdim=True)).abs().mean().item()
    r136_checks[name136] = err136
    print("transition draw-invariance", name136, err136)
r136_activation_name = min(r136_checks, key=r136_checks.get)
r136_activation = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}[r136_activation_name]
r136_H = r136_activation(F.linear(r136_X, r136_W1, r136_b1))
r136_logits = F.linear(r136_H, r136_W2, r136_b2)
r136_transition = r136_logP - r136_logits
r136_transition -= r136_transition.mean(-1, keepdim=True)
r136_transition = r136_transition.mean(0, keepdim=True)
r136_reconstructed = (r136_logits + r136_transition).softmax(-1)
print("selected activation:", r136_activation_name)
print("cached probability max error:", (r136_reconstructed - r136_P).abs().max().item())
print("probability minimum:", r136_P.min().item())

with Path("/work/notes.md").open("a") as note136:
    note136.write("\n## Runs 133–136: restart recovery\n"
                  "Architecture and causal checks recovered. Existing strength-2 insulin edit remains unchanged. "
                  "NE immediate adjusted slope +3.29011 is stable across dose/hour strata; "
                  "the artificial code probe is only a representation diagnostic, not a causal intervention. "
                  "Recovering exact cached head probabilities and reading labels before a gated NE local-loss test.\n")
```
