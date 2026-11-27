At ordinary amplitudes, the isolated insulin response is nearly symmetric. Norepinephrine saturates and also shifts glucose. These results do not explain the early wrong-sign insulin half-dose response. I’ll now replay complete changed-dose tokens, including the model’s pre-existing action pathway. A wrong-sign one-step response would locate the problem before rollout feedback.

```python
print("=== PRELUDE RETURN STRUCTURES ===")
def summary218(x):
    if torch.is_tensor(x):
        return ("Tensor", tuple(x.shape), str(x.dtype))
    if isinstance(x, np.ndarray):
        return ("ndarray", x.shape, str(x.dtype))
    if isinstance(x, dict):
        return {str(k): summary218(v) for k, v in list(x.items())[:12]}
    if isinstance(x, (tuple, list)):
        return [summary218(v) for v in x]
    return (type(x).__name__, str(x)[:100])

print("r214_base", summary218(r214_base))
print("r214_results example", summary218(r214_results[(0, .5)]))
print("r214_tokens example", summary218(r214_tokens[(0, .5)]))
for name218 in ["prelude155", "r214_parts", "preludeE185"]:
    fn218 = globals()[name218]
    print(name218, "locals", fn218.__code__.co_varnames,
          "globals", fn218.__code__.co_names,
          "constants", [x for x in fn218.__code__.co_consts
                        if isinstance(x, (str, int, float))])

print("\n=== RETAINED TOKEN / CONTEXT OBJECTS ===")
for name218, obj218 in sorted(list(globals().items())):
    if any(s in name218.lower() for s in ["tokens", "ctx169", "context169", "213"]):
        if hasattr(obj218, "shape"):
            print(name218, summary218(obj218))
        elif isinstance(obj218, dict):
            print(name218, "dict keys", list(obj218)[:8])
        elif callable(obj218):
            print(name218, "callable")

print("\n=== COMPLETE EDITED-TOKEN REPLAY ===")
replay218 = {}
for key218, tok218 in sorted(r214_tokens.items()):
    try:
        ep218 = prelude155(tensor153(tok218)[i169], overrides=candidate206)
        if isinstance(ep218, (tuple, list)):
            matches218 = [
                x for x in ep218
                if hasattr(x, "shape") and tuple(x.shape[-2:]) == (4, 192)
            ]
            if len(matches218) != 1:
                print(key218, "ambiguous prelude return", summary218(ep218))
                continue
            ep218 = matches218[0]
        if tuple(ep218.shape) != (len(i169), 4, 192):
            print(key218, "unexpected prelude return", summary218(ep218))
            continue
        hp218, op218 = state_update82(
            hf169, tensor153(ep218), mem169, mask169, age_emb169, 0
        )
        pred218 = arr183(phys171(view171(hp218, op218, dt169)))
        replay218[key218] = pred218
        print(key218, "mean predictions", np.round(pred218.mean(axis=0), 5).tolist())
    except Exception as e218:
        print(key218, type(e218).__name__, str(e218)[:180])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 218; forwards 14. Fixed-context raw insulin impulses are approximately symmetric at ordinary amplitudes; NE is saturating and cross-affects glucose. Testing complete changed-dose token/prelude/recurrent responses, including the pre-existing action pathway. No weights changed.\n")
```