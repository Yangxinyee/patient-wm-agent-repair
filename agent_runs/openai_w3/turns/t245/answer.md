The previous replay stopped because the prelude returns several tensors, not one. The retained wrapper identifies its first tensor as the normalized update. I’ll use that to compare complete dose changes against the factual baseline. This will test whether the half-dose sign is already wrong after one update or changes only during rollout.

```python
keys219 = sorted(r214_tokens)
base_tok219 = tensor153(tokens206)[i169]
base_ep219 = prelude155(base_tok219, overrides=candidate206)[0]
print("baseline E agrees with retained parts:",
      float(torch.sqrt(torch.mean((base_ep219 - r214_base[0][i169]) ** 2))))

def predict_E219(ep219):
    hp219, op219 = state_update82(
        hf169, tensor153(ep219), mem169, mask169, age_emb169, 0
    )
    return arr183(phys171(view171(hp219, op219, dt169)))

base_pred219 = predict_E219(base_ep219)
orig_ep219 = prelude155(tokens_orig169, overrides={})[0]
orig_pred219 = predict_E219(orig_ep219)
replay219 = {"baseline": base_pred219, "original_baseline": orig_pred219}

print("\n=== COMPLETE ONE-UPDATE RESPONSES ===")
print("edited baseline", np.round(base_pred219.mean(axis=0), 5).tolist())
print("edited minus original baseline",
      np.round((base_pred219 - orig_pred219).mean(axis=0), 5).tolist())
print("sensor order", np.asarray(codes171).tolist())
for key219 in keys219:
    ep219 = prelude155(
        tensor153(r214_tokens[key219])[i169], overrides=candidate206
    )[0]
    pred219 = predict_E219(ep219)
    replay219[key219] = pred219
    shift219 = (pred219 - base_pred219).mean(axis=0)
    decoded_delta219 = (
        np.asarray(r213_signals[key219]) - np.asarray(r213_s0)
    )[i169].mean(axis=0)
    print(key219,
          "value shift", np.round(shift219, 5).tolist(),
          "decoded dose delta", np.round(decoded_delta219, 5).tolist())

print("\n=== RETAINED COMPENSATION / DIRECTION OBJECTS ===")
for name219, obj219 in sorted(list(globals().items())):
    if name219.endswith(("192", "206")) and not name219.startswith("_"):
        if callable(obj219):
            print(name219, "callable")
        elif isinstance(obj219, dict):
            print(name219, "dict", list(obj219)[:8])
        elif hasattr(obj219, "shape"):
            print(name219, tuple(obj219.shape))
        elif isinstance(obj219, str):
            print(name219, "string", len(obj219))
        elif isinstance(obj219, (int, float, bool)):
            print(name219, obj219)

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 219; forwards 14. Previous complete-token replay was skipped due to ambiguous tuple extraction; corrected to prelude return[0], as used by preludeE185. Comparing complete candidate dose changes to its own factual baseline. No weights changed.\n")
```