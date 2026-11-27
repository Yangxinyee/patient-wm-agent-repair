The architecture confirms that a value-head edit leaves other heads’ weights unchanged, though rollout feedback can still affect their predictions. The likelihood discrepancy may come from how cached samples were averaged. I’ll inspect those dimensions and the existing likelihood helper; matching the trusted calculation is required before testing another edit.

```python
import dis, torch

def r261_summary(x):
    if isinstance(x, dict):
        return {k: r261_summary(v) for k, v in list(x.items())[:24]}
    if hasattr(x, "shape"):
        return (type(x).__name__, tuple(x.shape))
    if isinstance(x, (tuple, list)):
        return (type(x).__name__, len(x),
                [r261_summary(v) for v in x[:5]])
    return repr(x)[:130]

print("TRAIN PACK:", r261_summary(packs259["train"]))
print("TUNING PACK:", r261_summary(packs259["tune"]))
print("NE DOUBLE PACK:", r261_summary(packs259[("norepinephrine", 2.0)]))
print("DATASET TRAIN:", r261_summary(datasets259["train"]))

print("\n=== LIKELIHOOD HELPER ===")
dis.dis(mixture_nll190)
print("\n=== HEAD HELPER METADATA ===")
for r261_name in ["head177_eval", "head177_make", "scalar_nll190",
                  "cal170_nll", "reconstruct_reading_head"]:
    r261_fn = globals()[r261_name]
    print(r261_name, "names:", r261_fn.__code__.co_names,
          "constants:", repr(r261_fn.__code__.co_consts)[:650])

print("\n=== FUNCTION NAMES ===")
print([k for k, v in list(globals().items())
       if inspect.isfunction(v) and not k.startswith("r26")])

print("\n=== CACHED LIKELIHOOD CHECK ===")
print("check_nll259 shape:", tuple(check_nll259.shape))
print("target259:", tuple(target259.shape), target259.dtype)
print("fit/cv counts:", int(np.sum(fit259)), int(np.sum(cv259)))
print("pred summary:", r261_summary(tr259["pred"]))
print("probability sums:",
      float(p259.sum(-1).min()), float(p259.sum(-1).max()))

print("\n=== CAUSAL EVIDENCE SUMMARY ===")
for r261_item in chosen257[:5]:
    print(str(r261_item)[:800])

with open("/work/notes.md", "a") as r261_f:
    r261_f.write(
        "\n## Run 261: cached likelihood dimensions\n"
        "- Inspecting cached sample axes and the trusted mixture-likelihood helper "
        "before fitting an output correction. No weights changed.\n"
    )
```