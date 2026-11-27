The record-derived accumulation estimate suggests targets near −14.8/+7.4 for insulin and +7.1/−3.6 for norepinephrine on the benchmark stays, though those estimates remain uncertain. I’ll install a conservative rank-two recurrent-input edit. It should strengthen both drug responses; a factual NLL increase above 0.005 would reject it.

```python
import torch
import torch.nn.functional as F

print("=== INJECTION LOCATION CHECK ===")
print("read_hpre120 signature:", inspect.signature(read_hpre120))
print("read_hpre120 constants:", read_hpre120.__code__.co_consts)
assert "core.c_out" in str(read_hpre120.__code__.co_consts), \
    "Verify the steering coordinate before editing."

def as_numpy132(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

Ef132 = as_numpy132(factual112).astype(np.float64)
rf132 = Ef132 @ Wdose128
print("Factual residual RMS, all slots:",
      np.sqrt(np.mean(rf132**2, axis=(0, 1))))
print("Factual residual RMS, mean slot:",
      np.sqrt(np.mean(rf132.mean(axis=1)**2, axis=0)))
print("Affine-folding max error:",
      np.max(np.abs(rf132 - (Ef132 @ Edecoder128 + bdose128))))

delta132 = F.softplus(load_original("core.delta_in")).cpu().numpy().astype(np.float64)
original_b132 = as_numpy132(load_original("core.b_in.weight")).astype(np.float64)
candidate_gain132 = np.array([-5.0, 8.0])
change_b132 = ((V128 * candidate_gain132[None, :]) / delta132[:, None]) @ Wdose128.T
print("Delta range:", delta132.min(), delta132.max())
print("Original/change weight norms:",
      np.linalg.norm(original_b132), np.linalg.norm(change_b132))
print("Intended factual output-displacement RMS:",
      np.sqrt(np.mean((rf132 * candidate_gain132)**2, axis=(0, 1))))
assert np.isfinite(change_b132).all()
assert not edited(), "Unexpected prior edits; do not combine candidates silently."
save_edit("core.b_in.weight",
          torch.tensor(original_b132 + change_b132, dtype=torch.float32))

with open("/work/notes.md", "a") as f:
    f.write("""
## Runs 128–132: direct recurrent steering candidate
Recovered finite-difference h-pre-coda steering V128 with near-unit glucose/MAP responses and negligible cross-response. E dose residual is folded into Wdose128 using the layer-normalization invariant (max conversion error about 2.5e-5). Installed only core.b_in.weight:
delta_B = diag(1/softplus(delta_in)) @ V128 @ diag([-5,8]) @ Wdose128.T.
This is a conservative diagnostic, not an accepted final repair. Record-supported immediate effects remain insulin roughly -1.55 to -1.97/rate and NE +3.5 to +3.9/rate; statistical accumulation targets on 100 tuning stays roughly -14.8/+7.4 and +7.1/-3.6, with persistence uncertainty. Next require exact factual budget and paired rollout improvement.
""")
print("Current edits:", edited())
print("Benchmark stays:", benchmark_stays)
```