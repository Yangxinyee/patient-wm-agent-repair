The record-derived impulse estimates support a positive norepinephrine effect lasting several hours. However, the current edit responds much less to halving than to doubling, so one multiplier may not repair both. I’ll check its decoder activations and local factual cost first. A stronger correction is worth testing only if the available NLL margin can support it.

```python
import inspect
from scipy.special import ndtr, logsumexp

print("===== HEAD HELPER SIGNATURES =====")
for n in ["r138_nll", "r138_pair", "r145_head", "r145_hidden",
          "r129_queries", "r159_pair", "response_arrays"]:
    fn = globals().get(n)
    if callable(fn):
        print(n, inspect.signature(fn))
        try:
            print(inspect.getsource(fn)[:900])
        except (OSError, TypeError):
            print("locals:", fn.__code__.co_varnames[:20])

print("\n===== RELEVANT CACHED ARRAYS =====")
lines = []
for n in sorted(list(globals())):
    if re.match(r"r(137|138|139|145)_", n):
        v = globals()[n]
        if isinstance(v, dict):
            lines.append(f"{n}: dict {list(v)[:8]}")
        elif hasattr(v, "shape"):
            lines.append(f"{n}: {tuple(v.shape)}")
        elif not callable(v):
            lines.append(f"{n}: {repr(v)[:100]}")
print("\n".join(lines)[:2600])

r174_w1 = torch.as_tensor(load("attr_heads.value_bin.0.weight")).numpy()
r174_b1 = torch.as_tensor(load("attr_heads.value_bin.0.bias")).numpy()
i, j = r170_rows
print("\n===== NE PAIR CONSISTENCY =====")
print("direction weight error",
      np.max(np.abs((r174_w1[i] - r174_w1[j]) / 2 - r165_ne_dv)))
print("gate weight error",
      np.max(np.abs((r174_w1[i] + r174_w1[j]) / 2 - r165_ne_gv)))
print("direction bias error",
      abs((r174_b1[i] - r174_b1[j]) / 2 - r165_ne_db))
print("gate bias error",
      abs((r174_b1[i] + r174_b1[j]) / 2 - r165_ne_gb))
for label, x in [("D", r165_ne_D), ("G", r165_ne_G)]:
    print(label, "quantiles", np.quantile(x, [0, .01, .1, .5, .9, .99, 1]))

r174_logits = torch.as_tensor(r170_local_logits).numpy().astype(np.float64)
r174_y = np.asarray(_target_index, dtype=int)
r174_pair = ((r165_ne_G + r165_ne_D) * ndtr(r165_ne_G + r165_ne_D)
             - (r165_ne_G - r165_ne_D) * ndtr(r165_ne_G - r165_ne_D))
r174_out = torch.as_tensor(load("attr_heads.value_bin.2.weight")).numpy()[:, i]

def r174_nll(logits):
    logp = logits - logsumexp(logits, axis=-1, keepdims=True)
    chosen = logp[:, np.arange(len(r174_y)), r174_y]
    return -(logsumexp(chosen, axis=0) - np.log(logits.shape[0]))

r174_base_loss = r174_nll(r174_logits)
print("\nNLL reconstruction error:",
      np.max(np.abs(r174_base_loss - np.asarray(r170_local_loss))))
print("===== LOCAL AMPLIFICATION SCREEN =====")
for s in [1.0, 1.25, 1.5, 1.8]:
    L = r174_logits + (s - 1) * r174_pair[..., None] * r174_out
    loss = r174_nll(L)
    print("NE output multiplier", s,
          "overall delta", float(np.mean(loss - np.asarray(r170_orig_loss))),
          "increment over current", float(np.mean(loss - r174_base_loss)))

print("\n===== READING-LEVEL ARRAY NAMES =====")
print([(n, str(type(v).__name__)) for n, v in list(globals().items())
       if hasattr(v, "shape") and tuple(v.shape) == (4510,)
       and any(k in n.lower() for k in ["event", "code", "hour", "mask", "fit", "valid"])][:50])
```
