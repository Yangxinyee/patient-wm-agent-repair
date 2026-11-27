The needed replay and readout helpers are still available. That lets us test the transport path numerically without gradients. I’ll check their timing and memory conventions, then measure how a small injected signal changes glucose and MAP predictions. Consistent sensitivities across drug bundles would justify a reversible rollout pilot.

```python
import dis, numpy as np, torch
def compact168(f, limit=100):
    ins = list(dis.get_instructions(f))
    chosen = ins if len(ins) <= limit else ins[:limit//2] + ins[-limit//2:]
    print(f.__name__, "instructions", len(ins))
    for x in chosen:
        print(x.opname, x.argrepr)

for f in [state_update82, next_view105, means105]:
    compact168(f, 85)

print("\n=== REPLAY CONVENTIONS ===")
for f in [attention82, layer82, prelude155, fn160]:
    print(f.__name__, "constants", f.__code__.co_consts)
print("W type", type(W).__name__)
print("indices155", indices155)
print("abs105", abs105)
print("next_dt105", np.asarray(next_dt105))
print("read_codes105", read_codes105)

result168 = fn160(indices155, 1.0)
print("fn160 result", type(result168).__name__,
      [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
       for v in result168] if isinstance(result168, tuple) else getattr(result168, "shape", None))
zero168 = {n: load(n).copy() for n in edited()}
result_pre168 = prelude155(tokens_carrier155[:16], zero168)
print("prelude155 result",
      [tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
       for v in result_pre168] if isinstance(result_pre168, tuple) else getattr(result_pre168, "shape", None))
print("candidate payload factual SD", np.std(payload154, axis=0),
      "mean", np.mean(payload154, axis=0))
print("reserved head output norm",
      np.linalg.norm(zero168["core.prelude.0.attn.out.weight"][:, 144:192]))
print("carrier orthogonality", carrier154.T @ carrier154)
print("value extractor on carriers", value_extract154.T @ carrier154)
```
