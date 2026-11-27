The carrier is decoded through the posterior, while compensation is an affine map of fixed FFN features. That permits a joint regression targeting the **total** state change, rather than one component in isolation. I’ll verify those features and the dose-replay interfaces; accurate held-out factual reconstruction and linear signed contrasts would support this approach.

```python
import inspect
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

def parts254(tokens, original=False):
    getter = load_original if original else load
    def weight(name):
        return torch.as_tensor(getter(name), dtype=torch.float32)
    def affine(x, prefix):
        bias = weight(prefix + ".bias") if prefix + ".bias" in NAMES else None
        return F.linear(x, weight(prefix + ".weight"), bias)
    def norm(x, prefix):
        return F.layer_norm(x, (192,), weight(prefix + ".weight"),
                            weight(prefix + ".bias"))
    x = torch.as_tensor(tokens, dtype=torch.float32)
    prefix = "core.prelude.0"
    n = norm(x, prefix + ".norm1")
    q = affine(n, prefix + ".attn.q").reshape(-1, 7, 4, 48).transpose(1, 2)
    k, v = affine(n, prefix + ".attn.kv").chunk(2, dim=-1)
    k = k.reshape(-1, 7, 4, 48).transpose(1, 2)
    v = v.reshape(-1, 7, 4, 48).transpose(1, 2)
    attention = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(48), dim=-1)
    pooled = (attention @ v).transpose(1, 2).reshape(-1, 7, 192)
    after = x + affine(pooled, prefix + ".attn.out")
    hidden = F.gelu(affine(norm(after, prefix + ".norm3"), prefix + ".ffn.0"))
    raw = after + affine(hidden, prefix + ".ffn.2")
    return after[:, :4], hidden[:, :4], raw[:, :4], norm(raw[:, :4], "core.prelude_norm")

after254, hidden254, raw254, E254 = parts254(tokens206)
print("Cached fixed-feature agreement RMS:",
      float((hidden254 - current_hidden220).square().mean().sqrt()))
print("Raw factual RMS / original-target RMS / mismatch:",
      float(raw254.square().mean().sqrt()),
      float(orig_raw220.square().mean().sqrt()),
      float((raw254 - orig_raw220).square().mean().sqrt()))

for name254 in ("action_vectors87", "posterior153", "response_raw234",
                "response186", "state_update82", "next_view105", "readout134"):
    f254 = globals()[name254]
    print(name254, inspect.signature(f254))
    print("references:", f254.__code__.co_names)

print("=== FEATURE/TARGET TABLES ===")
for name254 in ("contrast_features220", "repair_targets243", "refit220"):
    value254 = globals()[name254]
    print(name254, [(str(k), type(v).__name__, getattr(v, "shape", None))
                   for k, v in list(value254.items())[:8]])
print("=== LOCAL CALIBRATION ARRAYS ===")
print([(n, tuple(v.shape)) for n, v in list(globals().items())
       if hasattr(v, "shape")
       and __import__("re").search(r"(185|186|228|234|235)$", n)][:35])

with Path("/work/notes.md").open("a") as log254:
    log254.write(
        "\nPython run 254; forwards 20. Implemented direct weight-based "
        "prelude decomposition and checked fixed features. Preparing total "
        "raw-state factual/dose-contrast regression rather than separate "
        "attention gain scaling. No weights changed.\n"
    )
```