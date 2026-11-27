The retained decoder has a large intercept, making consistent feature centering especially important. I’ll compare its predictions with its fitted targets and inspect the graft formula. Small target errors but large graft outputs would point to an implementation error; large target errors would reject the decoder itself.

```python
import dis

def short_stats(x):
    x = np.asarray(x)
    return np.round(np.quantile(x[np.isfinite(x)], [0, .01, .1, .5, .9, .99, 1]), 5).tolist()

print("INSULIN DECODER RESULTS")
for key, result in decode_results.items():
    if key[0] == "insulin":
        print(key, str(result)[:450])

print("\nRAW GRAFT DECODER CHECK")
for split_name in ["first", "extra", "tuning"]:
    x0 = np.asarray(raw_input_sets[split_name])
    if hasattr(raw_input_sets[split_name], "detach"):
        x0 = raw_input_sets[split_name].detach().cpu().numpy()
    p0 = x0 @ np.asarray(graft_decoder_w) + float(graft_decoder_b)
    gm0 = np.asarray(glucose_masks[split_name], dtype=bool)
    vm0 = np.asarray(valid_glucose_masks[split_name], dtype=bool)
    target0 = {"first": unit_first, "extra": unit_extra, "tuning": unit_tuning}[split_name]
    print(split_name, "X", x0.shape,
          "all", short_stats(p0), "glucose", short_stats(p0[gm0]),
          "valid", short_stats(p0[vm0]), "target", short_stats(target0[vm0]),
          "valid RMSE", float(np.sqrt(np.mean((p0[vm0]-target0[vm0])**2))))
    gate0 = x0 @ np.asarray(gate_w) + float(gate_b)
    print("gate glucose/non-glucose",
          short_stats(gate0[gm0]), short_stats(gate0[~gm0]))

print("\nGRAFT FORMULAS")
for func_name in ["make_insulin_graft", "graft_probs"]:
    func0 = globals()[func_name]
    print("\n", func_name)
    try:
        source0 = inspect.getsource(func0)
    except Exception:
        source0 = "\n".join(f"{i.opname} {i.argrepr}" for i in dis.get_instructions(func0))
    print(source0[:3100])

print("\nFEATURE SET KEYS")
for k0, v0 in decode_feature_sets.items():
    print(k0, {k1: tuple(v1.shape) for k1, v1 in v0.items()})
```