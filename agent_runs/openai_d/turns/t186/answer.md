The paired-neuron helpers and candidate gates are still available. The guard appears designed to suppress large corrections outside the cached insulin-active support, but that remains a hypothesis until tested on new stays. I’ll inspect its exact parameters and recover the target readouts before installing it.

```python
print("=== NOTEBOOK TARGET AND VALIDATION SUMMARIES ===")
matches = []
for line in notes_text.splitlines():
    if any(term in line.lower() for term in (
        "target", "convolution", "forward 14", "forward 15",
        "cached", "consistency", "run 115", "run 124", "run 129"
    )):
        matches.append(line)
print("\n".join(matches)[-5100:])

def describe_small(obj, depth=0):
    if isinstance(obj, dict):
        return {str(k): describe_small(v, depth + 1)
                for k, v in obj.items()} if depth < 2 else list(obj)
    if hasattr(obj, "shape"):
        a = np.asarray(obj) if not hasattr(obj, "detach") else obj.detach().cpu().numpy()
        return {"shape": list(a.shape),
                "range": [float(a.min()), float(a.max())]} if a.size else {"shape": list(a.shape)}
    if isinstance(obj, (tuple, list)):
        return [describe_small(v, depth + 1) for v in obj]
    return obj

print("\n=== GUARDED CANDIDATE ===")
print(describe_small(r159_candidates[("warmup", 4, 1.0)]))
print("\n=== SOURCE GATES ===")
print(describe_small(r159_sources))
print("\n=== DECODER AND OUTPUT PARAMETERS ===")
print("decoder bias:", r157_decoder_b)
print("pair slots:", r159_p, r159_m)
print("r159_C:", r159_C)
print("strength-2 output rows:",
      r157_np(r115_strength2["attr_heads.value_bin.2.weight"])[:, [r159_p, r159_m]])
```
